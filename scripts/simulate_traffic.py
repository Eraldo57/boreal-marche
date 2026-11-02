"""Local demonstration traffic, not production observations or a load benchmark.

Real traffic: actual historical feature rows (customer_features.parquet) sent to
/predict/batch, and actual September-November 2011 baskets from the cleaned workbook
(the first half of each invoice's items, as in the offline evaluation) sent to /recommend.

--shifted adds deliberate synthetic perturbations, so each monitor has something to see:
+365 days of recency (input and prediction drift), baskets of unknown items (the rule
fire-rate falls) and ten out-of-range feature vectors (rejected: data quality).
The background monitor evaluates its checks every ALERT_INTERVAL_SECONDS (60 by default);
the script waits for it and records the alerts it raised.
"""

import argparse
import json
import time
from datetime import UTC, datetime

import httpx
import pandas as pd

from src.contracts import FEATURE_NAMES, PROCESSED, PROJECT_ROOT

SINGLE = 20  # sent one by one to /predict; the rest go to /predict/batch in groups of 30


def real_baskets(count: int, seed: int) -> list[list[str]]:
    """Half of each of *count* real later invoices (canonical descriptions, first-seen order)."""
    columns = ["Invoice", "Description", "InvoiceDate", "IsPurchase"]
    events = pd.read_parquet(PROCESSED / "transactions.parquet", columns=columns)
    window = (events["InvoiceDate"] >= "2011-09-01") & (events["InvoiceDate"] < "2011-12-01")
    late = events[events["IsPurchase"] & window]
    items = late.groupby("Invoice")["Description"].agg(lambda d: list(dict.fromkeys(d)))
    items = items[items.map(len) >= 2].sample(count, random_state=seed)
    return [basket[: len(basket) // 2] for basket in items]


def wait_for_alerts(client: httpx.Client, seconds: float) -> dict:
    """Wait until the background monitor has alerted on every currently breached check."""
    deadline = time.monotonic() + seconds
    while True:
        checks = client.get("/metrics").json()["checks"]
        breached = {c["check"] for c in checks if c["level"] in ("yellow", "red")}
        alerts = client.get("/alerts").json()
        if breached <= set(alerts["active"]) or time.monotonic() > deadline:
            return alerts
        time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--baskets", type=int, default=40)
    parser.add_argument("--shifted", action="store_true")
    parser.add_argument("--wait-alerts", type=float, default=75.0)
    args = parser.parse_args()
    if not 100 <= args.count <= 1000:
        parser.error("--count must be between 100 and 1000 for this local demo")
    features = pd.read_parquet(PROCESSED / "customer_features.parquet")[FEATURE_NAMES]
    sampled = features.sample(args.count, random_state=42, replace=True)
    baskets = real_baskets(args.baskets, seed=42)
    invalid = []
    if args.shifted:
        sampled["recency_days"] += 365
        half = len(baskets) // 2
        baskets = baskets[:half] + [[f"unknown item {i}"] for i in range(len(baskets) - half)]
        broken = sampled.head(10).copy()
        broken["recency_days"] = -1.0  # outside the valid range: the API rejects it
        invalid = [
            {"customer_id": "DEMO", "features": row.astype(float).to_dict()}
            for _, row in broken.iterrows()
        ]
    requests = [
        {"customer_id": "DEMO", "features": row.astype(float).to_dict()}
        for _, row in sampled.iterrows()
    ]
    with httpx.Client(base_url=args.base_url, timeout=120) as client:
        before = client.get("/metrics").json()
        for payload in requests[:SINGLE]:  # one-customer lookups: what the latency target is for
            client.post("/predict", json=payload).raise_for_status()
        for offset in range(SINGLE, len(requests), 30):
            response = client.post(
                "/predict/batch", json={"customers": requests[offset : offset + 30]}
            )
            response.raise_for_status()
            assert response.json()["count"] == len(requests[offset : offset + 30])
        for basket in baskets:
            client.post("/recommend", json={"basket": basket, "top_k": 5}).raise_for_status()
        rejected = sum(client.post("/predict", json=p).status_code == 422 for p in invalid)
        client.post("/chat", json={"message": "What is the return policy?"}).raise_for_status()
        alerts = wait_for_alerts(client, args.wait_alerts)
        after = client.get("/metrics").json()
    assert after["drift_status"] == "measured"
    report = {
        "executed_at": datetime.now(UTC).isoformat(),
        "purpose": "local demo traffic; not production drift or performance benchmarking",
        "prediction_rows_sent": len(requests),
        "synthetic_recency_shift_days": 365 if args.shifted else 0,
        "real_baskets_sent": sum(not b[0].startswith("unknown item") for b in baskets),
        "unknown_item_baskets_sent": sum(b[0].startswith("unknown item") for b in baskets),
        "invalid_requests_sent": len(invalid),
        "invalid_requests_rejected": rejected,
        "before": before,
        "after": after,
        "alerts": alerts,
        "caveat": "Metrics include earlier local traffic in the durable log",
    }
    (PROJECT_ROOT / "reports/current_run/monitoring_demo.json").write_text(
        json.dumps(report, indent=2)
    )
    print(
        f"Sent {len(requests)} predictions, {len(baskets)} baskets and {len(invalid)} invalid "
        f"requests; status {after['status']}; {len(alerts['alerts'])} alerts recorded"
    )


if __name__ == "__main__":
    main()
