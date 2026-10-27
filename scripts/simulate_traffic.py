"""Local demonstration traffic, not production observations or a load benchmark.

Samples actual historical feature rows. --shifted deliberately adds 365 days
to recency to exercise drift calculation; this is a synthetic perturbation.
"""

import argparse
import json
from datetime import UTC, datetime

import httpx
import pandas as pd

from src.contracts import FEATURE_NAMES, PROCESSED, PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--shifted", action="store_true")
    args = parser.parse_args()
    if not 100 <= args.count <= 1000:
        parser.error("--count must be between 100 and 1000 for this local demo")
    features = pd.read_parquet(PROCESSED / "customer_features.parquet")[FEATURE_NAMES]
    sampled = features.sample(args.count, random_state=42, replace=True)
    if args.shifted:
        sampled["recency_days"] += 365
    requests = [
        {"customer_id": "DEMO", "features": row.astype(float).to_dict()}
        for _, row in sampled.iterrows()
    ]
    with httpx.Client(base_url=args.base_url, timeout=120) as client:
        before = client.get("/metrics").json()
        for offset in range(0, len(requests), 30):
            response = client.post(
                "/predict/batch", json={"customers": requests[offset : offset + 30]}
            )
            response.raise_for_status()
            assert response.json()["count"] == len(requests[offset : offset + 30])
        client.post("/recommend", json={"basket": []}).raise_for_status()
        client.post("/chat", json={"message": "What is the return policy?"}).raise_for_status()
        after = client.get("/metrics").json()
    assert after["drift_status"] == "measured"
    report = {
        "executed_at": datetime.now(UTC).isoformat(),
        "purpose": "local demo traffic; not production drift or performance benchmarking",
        "prediction_rows_sent": len(requests),
        "synthetic_recency_shift_days": 365 if args.shifted else 0,
        "before": before,
        "after": after,
        "caveat": "Metrics include earlier local traffic in the durable log",
    }
    (PROJECT_ROOT / "reports/current_run/monitoring_demo.json").write_text(
        json.dumps(report, indent=2)
    )
    print(f"Sent {len(requests)} demo predictions; PSI/KS measured from logged feature values")


if __name__ == "__main__":
    main()
