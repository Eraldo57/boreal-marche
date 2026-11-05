"""Numbers behind the defence answers that no other report already holds.

Each section answers one question of the brief's defence bank (Appendix C) from the real
data and the saved model, so a figure quoted in docs/DEFENCE_PREPARATION.md can be traced:

    4   how long customers really wait between purchases, against the 90-day horizon
    8   which features the model relies on most, over all customers
    9   the model against a "most recent customers first" rule, at the same selection rate
    12  the rules the lift filter discards, and how popular their consequents are

    python -m scripts.defence_facts      # writes reports/current_run/defence_facts.json
"""

import json
from collections import Counter
from datetime import UTC, datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd

from api.routers.predict import _final_estimator
from src.contracts import ARTIFACTS, FEATURE_NAMES, PROCESSED, PROJECT_ROOT
from src.ingest import load_config
from src.rules import baskets, build_transaction_matrix, mine, temporal_baskets

OUTPUT = PROJECT_ROOT / "reports/current_run/defence_facts.json"


def purchase_gaps(config: dict[str, Any]) -> dict[str, Any]:
    """Days between a customer's consecutive purchase days (question 4)."""
    frame = pd.read_parquet(
        PROJECT_ROOT / config["data"]["parquet_path"],
        columns=["Customer ID", "InvoiceDate", "IsPurchase"],
    )
    days = frame.loc[frame["IsPurchase"]].assign(day=lambda f: f["InvoiceDate"].dt.normalize())
    days = days.drop_duplicates(["Customer ID", "day"]).sort_values(["Customer ID", "day"])
    gaps = days.groupby("Customer ID")["day"].diff().dt.days.dropna()
    horizon = config["labelling"]["label_window_days"]
    return {
        "horizon_days": horizon,
        "customers_with_a_purchase": int(days["Customer ID"].nunique()),
        "repeat_gaps": int(len(gaps)),
        "gap_days_median": float(np.median(gaps)),
        "gap_days_p75": float(np.percentile(gaps, 75)),
        "gap_days_p90": float(np.percentile(gaps, 90)),
        "share_of_gaps_within": {
            str(days_): round(float((gaps <= days_).mean()), 4) for days_ in (30, 60, horizon, 180)
        },
    }


def feature_importance() -> dict[str, Any]:
    """The classifier's global importances (question 8)."""
    estimator = _final_estimator(joblib.load(ARTIFACTS / "retention_model.joblib"))
    ranked = sorted(zip(FEATURE_NAMES, estimator.feature_importances_), key=lambda p: -p[1])
    return {
        "classifier": type(estimator).__name__,
        "ranking": [{"feature": n, "importance": round(float(v), 4)} for n, v in ranked],
    }


def model_versus_recency_rule() -> dict[str, Any]:
    """Recall and precision on the test snapshot at the model's own selection rate (question 9)."""
    features = pd.read_parquet(PROCESSED / "X.parquet")
    snapshots = pd.read_parquet(PROCESSED / "snapshots.parquet")
    predictions = pd.read_parquet(ARTIFACTS / "test_predictions.parquet")
    test = snapshots.index[snapshots["split"] == "test"]
    if not (
        predictions["customer_id"].astype(str).to_numpy()
        == snapshots.loc[test, "customer_id"].astype(str).to_numpy()
    ).all():
        raise ValueError("test_predictions.parquet is not in the test snapshot's order")
    probability = predictions["probability"].to_numpy()
    returned = predictions["actual_will_return"].astype(int).to_numpy() == 1
    recency = features.loc[test, "recency_days"].to_numpy()
    metadata = json.loads((ARTIFACTS / "retention_model_metadata.json").read_text())
    threshold = metadata["decision_threshold"]

    def scores(selected: Any) -> dict[str, float]:
        return {
            "selected_share": round(float(selected.mean()), 4),
            "recall": round(float((selected & returned).sum() / returned.sum()), 4),
            "precision": round(float((selected & returned).sum() / selected.sum()), 4),
        }

    model_selected = probability >= threshold
    count = int(round(model_selected.mean() * len(returned)))
    rule_selected = np.zeros(len(returned), bool)
    rule_selected[np.argsort(recency, kind="stable")[:count]] = True  # most recent first
    return {
        "snapshot": "test (2011-09-01)",
        "customers": int(len(returned)),
        "threshold": threshold,
        "model": scores(model_selected),
        "recency_rule_same_selection_rate": scores(rule_selected),
        "recency_rule_fixed_windows": {
            f"bought_within_{days}_days": scores(recency <= days) for days in (30, 60, 90, 120, 180)
        },
    }


def discarded_rules(config: dict[str, Any]) -> dict[str, Any]:
    """Rules removed by the lift filter, and what they look like (question 12)."""
    rules_config = config["rules"]
    frame = pd.read_parquet(PROJECT_ROOT / config["data"]["parquet_path"])
    train, _, _ = temporal_baskets(
        frame, rules_config["train_end"], rules_config["validation_end"], rules_config["test_end"]
    )
    rows = baskets(train)
    share = Counter(item for basket in rows for item in basket)
    matrix, _ = build_transaction_matrix(train)
    every, _, _ = mine(matrix, 0.005, 0.1, 0.0, "apriori", rules_config["max_itemset_length"])
    dropped = every[every["lift"] < rules_config["min_lift"]].sort_values(
        "confidence", ascending=False
    )
    name = lambda items: " + ".join(sorted(items))  # noqa: E731
    return {
        "training_baskets": len(rows),
        "min_support": 0.005,
        "min_confidence": 0.1,
        "min_lift": rules_config["min_lift"],
        "rules_without_a_lift_filter": int(len(every)),
        "rules_dropped_by_the_lift_filter": int(len(dropped)),
        "highest_confidence_among_dropped": round(float(dropped["confidence"].max()), 4),
        "dropped_rules_with_confidence_at_least_0.30": int((dropped["confidence"] >= 0.30).sum()),
        "most_common_items_share_of_baskets": [
            {"item": item, "share": round(count / len(rows), 4)}
            for item, count in share.most_common(3)
        ],
        "highest_confidence_dropped_rules": [
            {
                "antecedent": name(row["antecedents"]),
                "consequent": name(row["consequents"]),
                "confidence": round(float(row["confidence"]), 4),
                "lift": round(float(row["lift"]), 4),
                "consequent_share_of_baskets": round(
                    share[next(iter(row["consequents"]))] / len(rows), 4
                ),
            }
            for _, row in dropped.head(6).iterrows()
        ],
    }


def main() -> None:
    config = load_config()
    report = {
        "executed_at": datetime.now(UTC).isoformat(),
        "q4_purchase_gaps": purchase_gaps(config),
        "q8_feature_importance": feature_importance(),
        "q9_model_versus_recency_rule": model_versus_recency_rule(),
        "q12_lift_filter": discarded_rules(config),
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {OUTPUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
