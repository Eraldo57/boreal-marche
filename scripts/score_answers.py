"""Score the model-written policy answers against the corpus (brief 3.4: groundedness).

Reads tests/golden/policy_answer_rubric.json (facts from docs/corpus, and a hand-made verdict
for each saved answer) and the saved evaluations of each provider. It refuses a verdict whose
answer has changed since it was judged, checks that every fact is still in the corpus, and
measures whether the fact reached the model at all: the retrieval step hands the model only the
first few excerpts. Writes reports/current_run/answer_correctness.json.

    python -m scripts.score_answers
"""

import hashlib
import json
import re
import sys
from collections import Counter
from datetime import UTC, datetime

import yaml

from api.routers.chat import GENERATION_EXCERPTS
from src.contracts import PROJECT_ROOT
from src.rag.index import RagIndex

RUBRIC = PROJECT_ROOT / "tests/golden/policy_answer_rubric.json"
CORPUS = PROJECT_ROOT / "docs/corpus"
PROVIDERS = ("openai", "ollama")
OUTPUT = PROJECT_ROOT / "reports/current_run/answer_correctness.json"


def normal(text: str) -> str:
    """Whitespace-insensitive, case-insensitive form: chunks break lines mid-sentence."""
    return re.sub(r"\s+", " ", text).lower()


def counts(verdicts: list[str]) -> dict[str, int]:
    return dict(Counter(verdicts))


def main() -> None:
    rubric = json.loads(RUBRIC.read_text())
    golden = {
        q["id"]: q
        for q in json.loads((PROJECT_ROOT / "tests/golden/golden_questions.json").read_text())[
            "questions"
        ]
    }
    config = yaml.safe_load((PROJECT_ROOT / "configs/pipeline.yaml").read_text())["assistant"]
    saved = {}
    for provider in PROVIDERS:
        path = PROJECT_ROOT / f"reports/current_run/assistant_{provider}_metrics.json"
        report = json.loads(path.read_text())
        saved[provider] = {q["id"]: q["reply"] for q in report["streamed_generation"]["questions"]}
        if report["executed_at"] != rubric["judged_runs"][provider]:
            sys.exit(f"{provider}: the saved run is not the one the rubric judged; re-judge it")

    index = RagIndex()
    retrieval = {}
    for item in rubric["questions"]:
        qid = item["id"]
        for provider in PROVIDERS:
            reply = saved[provider][qid]
            if (
                hashlib.sha256(reply.encode()).hexdigest()
                != item["judgments"][provider]["reply_sha256"]
            ):
                sys.exit(f"Q{qid} {provider}: the answer changed since it was judged; re-judge it")
        if item["marker"] is None:
            continue
        source_text = normal((CORPUS / item["source"]).read_text())
        if normal(item["marker"]) not in source_text:
            sys.exit(f"Q{qid}: the marker is not in {item['source']}")
        hits = [
            h
            for h in index.query(golden[qid]["text"], top_k=config["max_chunks_per_query"])
            if h["score"] >= config["similarity_floor"]
        ]
        rank = next(
            (
                i
                for i, h in enumerate(hits, 1)
                if h["source"] == item["source"] and normal(item["marker"]) in normal(h["text"])
            ),
            None,
        )
        retrieval[qid] = {
            "rank": rank,
            "given_to_model": bool(rank and rank <= GENERATION_EXCERPTS),
        }

    answerable = [item for item in rubric["questions"] if item["answerable"]]
    scores = {}
    for provider in PROVIDERS:
        groups: dict[str, list[str]] = {"fact_given": [], "fact_not_given": [], "unanswerable": []}
        for item in rubric["questions"]:
            verdict = item["judgments"][provider]["verdict"]
            if not item["answerable"]:
                groups["unanswerable"].append(verdict)
            elif retrieval[item["id"]]["given_to_model"]:
                groups["fact_given"].append(verdict)
            else:
                groups["fact_not_given"].append(verdict)
        everything = [v for g in groups.values() for v in g]
        scores[provider] = {
            "questions": len(everything),
            "verdicts": counts(everything),
            "right_behaviour": sum(v in {"correct", "declined_appropriate"} for v in everything),
            "answered_wrongly": sum(v == "wrong" for v in everything),
            "by_retrieval": {name: counts(v) for name, v in groups.items()},
        }

    report = {
        "executed_at": datetime.now(UTC).isoformat(),
        "rater": "AI evaluator (single rater, not independent); see the rubric's 'about'",
        "judged_runs": rubric["judged_runs"],
        "verdict_meanings": rubric["verdicts"],
        "retrieval": {
            "answerable_questions": len(answerable),
            "fact_given_to_model": sum(r["given_to_model"] for r in retrieval.values()),
            "fact_in_retrieved_top_k": sum(r["rank"] is not None for r in retrieval.values()),
            "excerpts_given_to_model": GENERATION_EXCERPTS,
            "chunks_retrieved": config["max_chunks_per_query"],
            "similarity_floor": config["similarity_floor"],
            "per_question": retrieval,
            "note": "Searches with the golden question's own text. A provider may rephrase the "
            "question it sends to policy_search, so its excerpts can differ slightly.",
        },
        "providers": scores,
    }
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("retrieval", "providers")}, indent=2))


if __name__ == "__main__":
    main()
