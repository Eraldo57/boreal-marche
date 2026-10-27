"""Execute the preserved golden set through the actual app; save observations only."""

import argparse
import json
import time
from collections import Counter
from datetime import UTC, datetime

import numpy as np
from fastapi.testclient import TestClient

from api.main import app
from src.contracts import PROJECT_ROOT
from src.rag.index import RagIndex
from src.rag.providers import RegexProvider


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
    oos = [r for r in rows if r["category"] == "out_of_scope"]
    in_scope = [r for r in rows if r["category"] != "out_of_scope"]
    sourced = [r for r in rows if r["expected_source"]]
    extracted = [r for r in rows if r["policy_excerpt_in_reply"] is not None]
    index = RagIndex()
    french_query = "Quel est le délai pour retourner un produit non périssable ?"
    hits = index.query(french_query, top_k=5)
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
            "French policy answers may quote English source text; "
            "fluent translation is not implemented",
            "OpenAI and Ollama results are absent unless explicitly executed with --provider",
            "Costs unknown for external providers; no pricing assumptions fabricated",
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
