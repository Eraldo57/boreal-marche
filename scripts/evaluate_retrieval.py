"""How often does the passage that answers a question reach the model? (brief 3.4, grounding)

For each question the index returns its best chunks; a question counts as found at rank k when
a chunk of the right policy file contains the fact's marker phrase. The model is handed the
first GENERATION_EXCERPTS chunks, so "found in the top 3" is what decides whether it can answer.
Three question sets: the answerable golden questions (the development set, whose rubric holds
the markers) and a separate hold-out set that retrieval changes are judged on, in two kinds:
"direct" questions use the document's own words, "indirect" ones paraphrase.

    python -m scripts.evaluate_retrieval   # writes reports/current_run/retrieval_metrics.json
"""

import json
import re
from datetime import UTC, datetime
from typing import Any

import yaml

from api.routers.chat import GENERATION_EXCERPTS
from src.contracts import PROJECT_ROOT
from src.rag.index import RagIndex

GOLDEN = PROJECT_ROOT / "tests/golden"
OUTPUT = PROJECT_ROOT / "reports/current_run/retrieval_metrics.json"
FLOOR_OUTPUT = PROJECT_ROOT / "reports/current_run/retrieval_floor.json"
SWEEP = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50)


def normal(text: str) -> str:
    """Whitespace- and case-insensitive: chunks break lines mid-sentence."""
    return re.sub(r"\s+", " ", text).lower()


def fact_rank(index: Any, question: str, source: str, marker: str, config: dict) -> int | None:
    """Rank (1-based) of the first retrieved chunk of `source` that holds the marker."""
    hits = [
        hit
        for hit in index.query(question, top_k=config["max_chunks_per_query"])
        if hit["score"] >= config["similarity_floor"]
    ]
    return next(
        (
            rank
            for rank, hit in enumerate(hits, 1)
            if hit["source"] == source and normal(marker) in normal(hit["text"])
        ),
        None,
    )


def summarise(ranks: dict[str, int | None]) -> dict[str, Any]:
    found = [r for r in ranks.values() if r is not None]
    return {
        "questions": len(ranks),
        "fact_in_excerpts_given_to_model": sum(r <= GENERATION_EXCERPTS for r in found),
        "fact_in_retrieved_chunks": len(found),
        "ranks": ranks,
    }


def relevance_bands(index: Any, questions: dict[str, list[str]], config: dict) -> dict[str, Any]:
    """Where the best passage's score falls for answerable and for undocumented questions.

    A question is declined below the similarity floor, answered with a caution between the
    floor and the confident score, and answered plainly above it. The sweep shows that no
    single cutoff separates the two groups.
    """
    scores = {
        group: sorted(index.query(text, top_k=1)[0]["score"] for text in texts)
        for group, texts in questions.items()
    }
    floor, confident = config["similarity_floor"], config["confident_score"]
    bands = {
        group: {
            "questions": len(values),
            "declined": sum(v < floor for v in values),
            "answered_with_caution": sum(floor <= v < confident for v in values),
            "answered_plainly": sum(v >= confident for v in values),
        }
        for group, values in scores.items()
    }
    sweep = [
        {
            "cutoff": cutoff,
            **{
                f"{group}_below": sum(v < cutoff for v in values)
                for group, values in scores.items()
            },
        }
        for cutoff in SWEEP
    ]
    return {
        "similarity_floor": floor,
        "confident_score": confident,
        "bands": bands,
        "single_cutoff_sweep": sweep,
        "lowest_answerable": scores["answerable"][:5],
        "highest_undocumented": scores["undocumented"][-5:],
    }


def main() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "configs/pipeline.yaml").read_text())["assistant"]
    index = RagIndex()
    texts = {
        q["id"]: q["text"]
        for q in json.loads((GOLDEN / "golden_questions.json").read_text())["questions"]
    }
    rubric = json.loads((GOLDEN / "policy_answer_rubric.json").read_text())["questions"]
    holdout = json.loads((GOLDEN / "retrieval_holdout.json").read_text())["questions"]
    sets = {
        "golden_answerable": {
            f"Q{item['id']}": (texts[item["id"]], item["source"], item["marker"])
            for item in rubric
            if item["marker"]
        },
        **{
            f"holdout_{kind}": {
                q["id"]: (q["text"], q["source"], q["marker"]) for q in holdout if q["kind"] == kind
            }
            for kind in ("direct", "indirect")
        },
    }
    report = {
        "executed_at": datetime.now(UTC).isoformat(),
        "index": index.metadata,
        "excerpts_given_to_model": GENERATION_EXCERPTS,
        "chunks_retrieved": config["max_chunks_per_query"],
        "similarity_floor": config["similarity_floor"],
        **{
            name: summarise(
                {
                    qid: fact_rank(index, question, source, marker, config)
                    for qid, (question, source, marker) in questions.items()
                }
            )
            for name, questions in sets.items()
        },
    }
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    negatives = json.loads((GOLDEN / "retrieval_negatives.json").read_text())["questions"]
    answerable = [text for questions in sets.values() for text, _, _ in questions.values()]
    floor_report = {
        "executed_at": report["executed_at"],
        "answerable_questions": "the golden answerable questions plus the whole hold-out set",
        "undocumented_questions": "tests/golden/retrieval_negatives.json",
        **relevance_bands(
            index,
            {"answerable": answerable, "undocumented": [q["text"] for q in negatives]},
            config,
        ),
    }
    FLOOR_OUTPUT.write_text(json.dumps(floor_report, indent=2, ensure_ascii=False) + "\n")
    for group, band in floor_report["bands"].items():
        print(f"{group:13} {band}")
    for name in sets:
        s = report[name]
        print(
            f"{name:18} {s['questions']:>3} questions | fact in the {GENERATION_EXCERPTS} excerpts "
            f"given to the model: {s['fact_in_excerpts_given_to_model']:>2} | "
            f"in all {config['max_chunks_per_query']} retrieved: {s['fact_in_retrieved_chunks']:>2}"
        )


if __name__ == "__main__":
    main()
