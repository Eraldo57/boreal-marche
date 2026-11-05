"""Experiment (2026-11-02): which retrieval changes bring the answering passage to the model?

Not part of the product code: it is kept as the evidence behind the retrieval design. Run:
    python reports/current_run/retrieval_experiments/retrieval_lab.py "$PWD" > results.txt
 Builds in-memory variants over docs/corpus with the production
embedding model and scores each on the three question sets with the evaluator's own rule
(marker phrase in a returned passage of the right file, ranked by position).
"""

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(sys.argv[1])
sys.path.insert(0, str(REPO))
from src.rag.index import chunk_text, embedding_model  # noqa: E402

MODEL = embedding_model("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
norm = lambda s: re.sub(r"\s+", " ", s).lower()  # noqa: E731
tok = lambda s: re.findall(r"[a-zàâçéèêëîïôûùüÿœ0-9]+", s.lower())  # noqa: E731

golden = {q["id"]: q["text"] for q in json.loads((REPO / "tests/golden/golden_questions.json").read_text())["questions"]}
rubric = json.loads((REPO / "tests/golden/policy_answer_rubric.json").read_text())["questions"]
hold = json.loads((REPO / "tests/golden/retrieval_holdout.json").read_text())["questions"]
SETS = {
    "golden": [(f"Q{r['id']}", golden[r["id"]], r["source"], r["marker"]) for r in rubric if r["marker"]],
    "direct": [(q["id"], q["text"], q["source"], q["marker"]) for q in hold if q["kind"] == "direct"],
    "indirect": [(q["id"], q["text"], q["source"], q["marker"]) for q in hold if q["kind"] == "indirect"],
}


def sections(path: Path):
    text = path.read_text()
    title = re.match(r"# (.+)", text).group(1).strip()
    parts = re.split(r"(?m)^(?=## )", text)
    out = []
    for part in parts[1:]:
        heading = part.splitlines()[0].lstrip("# ").strip()
        out.append({"source": path.name, "title": title, "heading": heading, "text": part.strip()})
    return out


LIST_ITEM = re.compile(r"^\s*(?:[-*]|\d+\.)\s+")


def fine_units(body: str, size: int = 420):
    """One unit per list item, table row (with its header row) or paragraph."""
    units, cur, header = [], [], None

    def flush():
        if cur:
            units.append(" ".join(x.strip() for x in cur))
            cur.clear()

    for line in body.splitlines():
        if not line.strip():
            flush()
            header = None
        elif line.lstrip().startswith("|"):
            flush()
            if header is None:
                header = line.strip()
            elif not set(line.replace("|", "").strip()) <= set("-: "):
                units.append(header + "\n" + line.strip())
        elif LIST_ITEM.match(line):
            flush()
            cur.append(line)
        else:
            cur.append(line)
    flush()
    out = []
    for u in units:
        out += chunk_text(u, size, 0) if len(u) > size else [u]
    return out


def build(prefix: bool, size: int = 420, overlap: int = 64, fine: bool = False):
    """Children (what is embedded) and the parent section each came from."""
    parents, children = [], []
    for path in sorted((REPO / "docs/corpus").glob("*.md")):
        for sec in sections(path):
            pid = len(parents)
            parents.append(sec)
            body = sec["text"].split("\n", 1)[1] if "\n" in sec["text"] else ""
            for piece in (fine_units(body, size) if fine else chunk_text(body, size, overlap)) or [body]:
                head = f"{sec['title']} — {sec['heading']}\n" if prefix else ""
                children.append({"parent": pid, "source": sec["source"], "embed": head + piece, "piece": piece})
    vec = MODEL.encode([c["embed"] for c in children], normalize_embeddings=True)
    return parents, children, np.asarray(vec)


class Bm25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.tf = [Counter(tok(d)) for d in docs]
        self.len = np.array([sum(t.values()) for t in self.tf], float)
        self.avg, self.k1, self.b, n = self.len.mean(), k1, b, len(docs)
        df = Counter(w for t in self.tf for w in t)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, q):
        s = np.zeros(len(self.tf))
        for w in set(tok(q)):
            if w not in self.idf:
                continue
            for i, t in enumerate(self.tf):
                f = t.get(w, 0)
                if f:
                    s[i] += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


def rank_children(q, children, vec, bm, mode, floor=0.25, k=60):
    sims = vec @ MODEL.encode([q], normalize_embeddings=True)[0]
    order = list(np.argsort(-sims))
    if mode == "vector":
        return [i for i in order if sims[i] >= floor][:12]
    kw = bm.scores(q)
    kw_order = [i for i in np.argsort(-kw) if kw[i] > 0]
    fused = Counter()
    for r, i in enumerate(order[:30]):
        fused[i] += 1 / (k + r)
    for r, i in enumerate(kw_order[:30]):
        fused[i] += 1 / (k + r)
    keep = [i for i, _ in fused.most_common() if sims[i] >= floor or i in kw_order[:5]]
    return keep[:12]


def run(label, prefix, expand, mode, size=420, overlap=64, n_given=3, n_retrieved=5, bm_on="piece", fine=False):
    parents, children, vec = build(prefix, size, overlap, fine)
    bm = Bm25([c["embed"] if bm_on == "embed" else c["piece"] for c in children])
    result, chars = {}, []
    for name, questions in SETS.items():
        given = retrieved = 0
        for qid, q, src, marker in questions:
            hits = rank_children(q, children, vec, bm, mode)
            passages, seen = [], set()
            for i in hits:
                c = children[i]
                key = c["parent"] if expand else i
                if key in seen:
                    continue
                seen.add(key)
                passages.append((c["source"], parents[c["parent"]]["text"] if expand else c["embed"]))
                if len(passages) == n_retrieved:
                    break
            rank = next((r for r, (s, t) in enumerate(passages, 1) if s == src and norm(marker) in norm(t)), None)
            given += bool(rank and rank <= n_given)
            retrieved += rank is not None
            chars += [sum(len(t) for _, t in passages[:n_given])]
        result[name] = (given, retrieved, len(questions))
    g, d, i = result["golden"], result["direct"], result["indirect"]
    total = g[0] + d[0] + i[0]
    print(f"{label:46} golden {g[0]:>2}/{g[2]} ({g[1]:>2} in 5) | direct {d[0]:>2}/{d[2]} ({d[1]:>2}) | indirect {i[0]:>2}/{i[2]} ({i[1]:>2}) | "
          f"top-{n_given} total {total}/{g[2]+d[2]+i[2]} | avg chars given {int(np.mean(chars))}", flush=True)
    return result


if __name__ == "__main__":
    run("A0 plain chunks (the layout before the change)", False, False, "vector")
    run("A1 + title and heading on each chunk", True, False, "vector")
    run("A2 + the whole section handed over (adopted)", True, True, "vector")
    run("A3 = A2 + keyword fusion (RRF)", True, True, "hybrid")
    run("A3b = A3, keyword search over the titled text", True, True, "hybrid", bm_on="embed")
    run("A4 = A3, 5 excerpts given", True, True, "hybrid", n_given=5)
    run("A5 = A2, 5 excerpts given", True, True, "vector", n_given=5)
    run("B1 plain chunks, 5 excerpts given", False, False, "vector", n_given=5)
    run("F1 = A2 with one chunk per item, row or paragraph", True, True, "vector", fine=True)
    run("F2 = F1 + keyword fusion", True, True, "hybrid", fine=True)
    run("F3 = F1, 5 excerpts given", True, True, "vector", fine=True, n_given=5)
