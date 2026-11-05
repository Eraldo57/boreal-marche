"""Export test confusion counts and paired-bootstrap uncertainty without model selection.

Developed with AI assistance; see docs/AI_USE.md. This reads saved predictions,
never fits or tunes, and treats each unique customer in the single test snapshot as a unit.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.contracts import ARTIFACTS, PROCESSED, PROJECT_ROOT


def classification(y: np.ndarray, selected: np.ndarray) -> dict[str, Any]:
    tp = int(np.sum((y == 1) & selected))
    fp = int(np.sum((y == 0) & selected))
    fn = int(np.sum((y == 1) & ~selected))
    tn = int(np.sum((y == 0) & ~selected))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "selection_rate": float(selected.mean()),
    }


def uncertainty(
    y: np.ndarray,
    probability: np.ndarray,
    baseline: np.ndarray,
    threshold: float,
    draws: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    if len(y) < 2 or set(np.unique(y)) != {0, 1} or draws < 20:
        raise ValueError("Both classes and at least 20 bootstrap draws are required")
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {
        k: [] for k in ("pr_auc", "roc_auc", "brier_score", "recall", "precision", "ap_difference")
    }
    for _ in range(draws):
        ix = rng.integers(0, len(y), len(y))
        yy, pp, bb = y[ix], probability[ix], baseline[ix]
        if len(np.unique(yy)) < 2:
            continue
        ap = float(average_precision_score(yy, pp))
        samples["pr_auc"].append(ap)
        samples["roc_auc"].append(float(roc_auc_score(yy, pp)))
        samples["brier_score"].append(float(brier_score_loss(yy, pp)))
        rates = classification(yy, pp >= threshold)
        samples["recall"].append(rates["recall"])
        samples["precision"].append(rates["precision"])
        samples["ap_difference"].append(ap - float(average_precision_score(yy, bb)))
    if len(samples["pr_auc"]) < 20:
        raise ValueError("Too few bootstrap samples containing both classes")
    return {
        "draws_requested": draws,
        "draws_used": len(samples["pr_auc"]),
        "seed": seed,
        "unit": "unique customer in one historical test snapshot",
        "method": "paired nonparametric percentile bootstrap; 95% intervals",
        "ci95": {
            key: [float(v) for v in np.percentile(values, [2.5, 97.5])]
            for key, values in samples.items()
        },
        "limitation": "Conditional on the selected model and this test period; not uncertainty "
        "over training/model selection or future temporal/domain shifts.",
    }


def run(draws: int = 1000) -> dict[str, Any]:
    meta = json.loads((ARTIFACTS / "retention_model_metadata.json").read_text())
    report = json.loads((PROJECT_ROOT / "reports/current_run/retention_metrics.json").read_text())
    pred = pd.read_parquet(ARTIFACTS / "test_predictions.parquet")
    snapshots = pd.read_parquet(PROCESSED / "snapshots.parquet")
    test = snapshots.loc[snapshots["split"] == "test"]
    if pred["customer_id"].duplicated().any() or not np.array_equal(
        pred["customer_id"].astype(str).to_numpy(), test["customer_id"].astype(str).to_numpy()
    ):
        raise ValueError("Expected unique test customers aligned with the test snapshot")
    y = pred["actual_will_return"].to_numpy(dtype=int)
    p = pred["probability"].to_numpy(dtype=float)
    x = pd.read_parquet(PROCESSED / "X.parquet")
    baseline = 1 / (1 + x.loc[test.index, "recency_days"].to_numpy())
    threshold = float(meta["decision_threshold"])
    metrics = {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
    }
    if any(
        not np.isclose(value, report["final_test"][key], atol=1e-10)
        for key, value in metrics.items()
    ):
        raise ValueError("Saved predictions disagree with the training report")
    selected = p >= threshold
    baseline_selected = np.zeros(len(y), dtype=bool)
    baseline_selected[np.argsort(-baseline, kind="stable")[: int(selected.sum())]] = True
    result = {
        "executed_at": datetime.now(UTC).isoformat(),
        "model_name": meta["model_family"],
        "model_version": meta["model_version"],
        "model_sha256": meta["model_sha256"],
        "feature_count": len(meta["feature_names"]),
        "snapshot": ", ".join(sorted(test["cutoff"].unique())),
        "customers": len(y),
        "threshold": threshold,
        "positive_prevalence": float(y.mean()),
        "model": {**metrics, **classification(y, selected)},
        "recency_same_selection_count": {
            "pr_auc": float(average_precision_score(y, baseline)),
            "roc_auc": float(roc_auc_score(y, baseline)),
            **classification(y, baseline_selected),
        },
        "uncertainty": uncertainty(y, p, baseline, threshold, draws),
        "calibration_decision_period": report["calibration_decision_period"],
        "selected_calibration": report["selected_calibration"],
        "interpretation": "Purchase prediction only; coupon/control effects are not measured.",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=1000)
    args = parser.parse_args()
    result = run(args.draws)
    path = PROJECT_ROOT / "reports/current_run/retention_diagnostics.json"
    path.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {"path": str(path), "model": result["model"], "ci95": result["uncertainty"]["ci95"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
