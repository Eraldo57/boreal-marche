"""Execute the preserved golden set through the actual app; save observations only."""

import argparse
import json
import time
from collections import Counter
from datetime import UTC, datetime

import numpy as np
import yaml
from fastapi.testclient import TestClient

from api.main import app
from src.contracts import PROJECT_ROOT
from src.rag.guardrails import detect_language
from src.rag.index import RagIndex
from src.rag.providers import RegexProvider


def token_cost(usage: dict | None, price: dict) -> float:
    """USD for one call's measured tokens, at the config's price per million tokens."""
    if not usage:
        return 0.0
    return (
        usage["prompt_tokens"] * price["input"] + usage["completion_tokens"] * price["output"]
    ) / 1e6


def generation_check(client, questions, provider):
    """Policy questions through /chat/stream: does the model write the answer, in the
    requested language, with citations? (Language models only; regex quotes sources.)"""
    rows = []
    for question in questions:
        if question["expected_tool"] != "policy_search":
            continue
        payload = {
            "message": question["text"],
            "language": question["language"],
            "provider": provider,
        }
        start = time.perf_counter()
        with client.stream("POST", "/chat/stream", json=payload) as response:
            events = [json.loads(line) for line in response.iter_lines() if line]
        done = next(e for e in events if e["type"] == "done")
        reply = done["reply"]
        rows.append(
            {
                "id": question["id"],
                "language": question["language"],
                "answer_written_by": done["answer_written_by"],
                "fell_back": any(e["type"] == "fallback" for e in events),
                "reply_language": detect_language(reply) if not done["refused"] else None,
                "cited": bool(done["citations"]),
                "figure_check": done.get("figure_check"),
                "token_usage": done.get("token_usage"),  # the routing call
                "generation_token_usage": done.get("generation_token_usage"),  # the answer
                "seconds": time.perf_counter() - start,
                "reply": reply,
            }
        )
    written = [r for r in rows if r["answer_written_by"] == provider]
    added = [r["id"] for r in written if r["figure_check"]["figures_not_in_sources"]]
    dropped = [r["id"] for r in written if r["figure_check"]["top_source_figures_missing"]]
    return {
        "policy_questions": len(rows),
        "written_by_model": len(written),
        "fallbacks": sum(r["fell_back"] for r in rows),
        "language_match_rate": sum(r["reply_language"] == r["language"] for r in written)
        / len(written)
        if written
        else None,
        "french_written": sum(r["language"] == "fr" for r in written),
        "all_written_answers_cited": all(r["cited"] for r in written),
        # Proxies for faithfulness, not proof: figures invented, and top-source figures omitted.
        "answers_adding_figures": added,
        "answers_omitting_top_source_figures": dropped,
        "seconds_p50": float(np.median([r["seconds"] for r in rows])) if rows else None,
        "questions": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["regex", "openai", "ollama"], default="regex")
    args = parser.parse_args()
    questions = json.loads((PROJECT_ROOT / "tests/golden/golden_questions.json").read_text())[
        "questions"
    ]
    rows = []
    with TestClient(app) as client:
        for question in questions:
            match = RegexProvider._CUSTOMER_ID_PAT.search(question["text"])
            payload = {
                "message": question["text"],
                "language": question["language"],
                "provider": args.provider,
                "customer_id": match.group(1) if match else None,
            }
            start = time.perf_counter()
            response = client.post("/chat", json=payload)
            elapsed = (time.perf_counter() - start) * 1000
            data = response.json()
            actual_tool = next((t["tool_name"] for t in data.get("tool_calls", [])), None)
            sources = [c["source"] for c in data.get("citations", [])]
            citations = data.get("citations", [])
            # Groundedness proxy is deliberately limited to extractive policy answers.
            extractive = (
                any(c["chunk"] in data.get("reply", "") for c in citations)
                if actual_tool == "policy_search"
                else None
            )
            rows.append(
                {
                    "id": question["id"],
                    "category": question["category"],
                    "language": question["language"],
                    "expected_tool": question["expected_tool"],
                    "actual_tool": actual_tool,
                    "tool_correct": actual_tool == question["expected_tool"],
                    "expected_source": question["expected_source"],
                    "sources": sources,
                    "source_correct": question["expected_source"] in sources
                    if question["expected_source"]
                    else None,
                    "refused": data.get("refused"),
                    "http_status": response.status_code,
                    "latency_ms": elapsed,
                    "provider_latency_ms": data.get("provider_latency_ms"),
                    "policy_excerpt_in_reply": extractive,
                    "cost_usd": data.get("cost_usd"),
                    "token_usage": data.get("token_usage"),
                    "reply": data.get("reply", data),
                }
            )
        generation = (
            generation_check(client, questions, args.provider) if args.provider != "regex" else None
        )
    oos = [r for r in rows if r["category"] == "out_of_scope"]
    in_scope = [r for r in rows if r["category"] != "out_of_scope"]
    sourced = [r for r in rows if r["expected_source"]]
    extracted = [r for r in rows if r["policy_excerpt_in_reply"] is not None]
    index = RagIndex()
    french_query = "Quel est le délai pour retourner un produit non périssable ?"
    hits = index.query(french_query, top_k=5)
    cost_per_query = rows[0]["cost_usd"] if rows else None
    cost_per_streamed_answer = None
    if args.provider == "openai":
        config = yaml.safe_load((PROJECT_ROOT / "configs/pipeline.yaml").read_text())
        price = config["assistant"]["openai"]["usd_per_million_tokens"]
        # Measured tokens priced at the list price: a /chat question costs its routing call;
        # a streamed policy answer costs its routing call plus the call that writes it.
        cost_per_query = sum(token_cost(r["token_usage"], price) for r in rows) / len(rows)
        written = [r for r in generation["questions"] if r["answer_written_by"] == args.provider]
        if written:
            cost_per_streamed_answer = sum(
                token_cost(r["token_usage"], price) + token_cost(r["generation_token_usage"], price)
                for r in written
            ) / len(written)
    report = {
        "executed_at": datetime.now(UTC).isoformat(),
        "provider": args.provider,
        "golden_count": len(rows),
        "category_counts": dict(Counter(q["category"] for q in questions)),
        "tool_choice_accuracy_all": sum(r["tool_correct"] for r in rows) / len(rows),
        "tool_choice_accuracy_in_scope": sum(r["tool_correct"] for r in in_scope) / len(in_scope),
        "out_of_scope_refusal_rate": sum(r["refused"] is True for r in oos) / len(oos),
        "in_scope_refusal_count": sum(r["refused"] is True for r in in_scope),
        "expected_source_accuracy": sum(r["source_correct"] for r in sourced) / len(sourced),
        "policy_extract_grounding_proxy": sum(r["policy_excerpt_in_reply"] for r in extracted)
        / len(extracted)
        if extracted
        else None,
        "latency_p50_ms": float(np.median([r["latency_ms"] for r in rows])),
        "latency_p95_ms": float(np.percentile([r["latency_ms"] for r in rows], 95)),
        "cost_usd_per_query": cost_per_query,
        "cost_usd_per_streamed_answer": cost_per_streamed_answer,
        "streamed_generation": generation,
        "multilingual_smoke": {
            "query_fr": french_query,
            "hits": hits,
            "english_return_policy_retrieved": any(
                h["source"] == "return_policy.md" and h["language"] == "en" for h in hits
            ),
        },
        "limitations": [
            "Golden set is a development regression set, not an unseen evaluation set",
            "Extractive grounding proxy does not measure answer relevance or completeness",
            "/chat policy answers quote the source text, which may be English for a French "
            "question; with OpenAI or Ollama, /chat/stream has the model write the answer "
            "(streamed_generation); this script does not score their correctness "
            "(scripts/score_answers.py does)",
            "OpenAI and Ollama results are absent unless explicitly executed with --provider",
            "Local providers cost nothing per query. OpenAI's cost is its measured tokens at the "
            "list price in configs/pipeline.yaml: per_query is the routing call; "
            "per_streamed_answer adds the call that writes the answer",
        ],
        "questions": rows,
    }
    path = PROJECT_ROOT / f"reports/current_run/assistant_{args.provider}_metrics.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"questions", "multilingual_smoke"}},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
