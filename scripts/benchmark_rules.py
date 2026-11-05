"""How does recommendation lookup scale with the number of rules? (defence question 14)

The real export has 3,567 rules. Larger sets are SYNTHETIC: antecedents of one or two items
and a consequent, all drawn by each item's real frequency in the training baskets, with
random confidence and lift. They test the cost of the lookup, not the quality of the advice.
Queries are real September-to-November baskets.

    python -m scripts.benchmark_rules       # writes reports/current_run/rules_scaling.json
"""

import json
import resource
import time
from collections import Counter
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from src.contracts import PROJECT_ROOT
from src.ingest import load_config
from src.recommend import RecommendationEngine
from src.rules import baskets, temporal_baskets

SIZES = (3_567, 10_000, 100_000, 500_000)
QUERIES = 300


def synthetic_rules(count: int, items: list[str], weights: np.ndarray, rng: Any) -> list[dict]:
    antecedent_size = rng.integers(1, 3, size=count)
    picks = rng.choice(len(items), size=(count, 3), p=weights)
    return [
        {
            "antecedent": sorted(
                {items[picks[i, 0]], *([items[picks[i, 1]]] * (antecedent_size[i] - 1))}
            ),
            "consequent": [items[picks[i, 2]]],
            "confidence": float(rng.uniform(0.1, 0.9)),
            "lift": float(rng.uniform(1.2, 30)),
            "support": float(rng.uniform(0.005, 0.05)),
        }
        for i in range(count)
    ]


def main() -> None:
    cfg = load_config()["rules"]
    df = pd.read_parquet(PROJECT_ROOT / load_config()["data"]["parquet_path"])
    train, _, test = temporal_baskets(df, cfg["train_end"], cfg["validation_end"], cfg["test_end"])
    counts = Counter(i for b in baskets(train) for i in b)
    items = sorted(counts)
    weights = np.array([counts[i] for i in items], float)
    weights /= weights.sum()
    rng = np.random.default_rng(42)
    queries = [b for b in baskets(test) if len(b) > 1][:QUERIES]
    real = json.loads((PROJECT_ROOT / "artifacts/rules.json").read_text())["rules"]
    popularity = json.loads((PROJECT_ROOT / "artifacts/popularity.json").read_text())["items"]
    results = []
    for size in SIZES:
        rules = real if size == len(real) else synthetic_rules(size, items, weights, rng)
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        start = time.perf_counter()
        engine = RecommendationEngine.from_records(rules, popularity)
        build_seconds = time.perf_counter() - start
        memory_mb = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before
        ) / 1e6  # macOS: bytes
        times = []
        for basket in queries:
            start = time.perf_counter()
            engine.recommend(basket)
            times.append((time.perf_counter() - start) * 1000)
        results.append(
            {
                "rules": size,
                "rules_are": "real export" if size == len(real) else "synthetic",
                "build_seconds": round(build_seconds, 2),
                "memory_added_mb": round(memory_mb, 1),
                "recommend_ms_p50": round(float(np.percentile(times, 50)), 3),
                "recommend_ms_p95": round(float(np.percentile(times, 95)), 3),
                "recommend_ms_max": round(float(max(times)), 3),
            }
        )
        print(results[-1], flush=True)
    report = {
        "executed_at": datetime.now(UTC).isoformat(),
        "queries": len(queries),
        "query_source": "real September-to-November baskets with more than one item",
        "note": "Only the 3,567-rule row uses the real export; larger rows are synthetic.",
        "results": results,
    }
    path = PROJECT_ROOT / "reports/current_run/rules_scaling.json"
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
