"""Held-out PR/ROC plots from saved predictions; no fitting or threshold selection.

Run ``python -m src.ranking_plots`` to export the existing local test run's plots.
Training calls the same renderer with its already-computed test predictions.
"""
# Developed with AI assistance; verification/provenance: docs/AI_USE.md.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.contracts import ARTIFACTS, PROCESSED, PROJECT_ROOT

FIGURE_NAMES = ("precision_recall_curve.png", "roc_curve.png")


def ranking_figures(
    y_true: Any,
    probabilities: Any,
    recency_scores: Any,
    threshold: float,
    model_name: str,
    snapshot: str,
) -> dict[str, Figure]:
    """Compare two rankings on identical labels; mark the existing model threshold."""
    y = np.asarray(y_true, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    baseline = np.asarray(recency_scores, dtype=float)
    if (
        y.ndim != 1
        or probability.shape != y.shape
        or baseline.shape != y.shape
        or set(np.unique(y)) != {0, 1}
        or not np.isfinite(probability).all()
        or not np.isfinite(baseline).all()
    ):
        raise ValueError("Aligned finite scores and both binary classes are required")

    pr, roc = Figure(figsize=(8, 6)), Figure(figsize=(8, 6))
    pr_ax, roc_ax = pr.subplots(), roc.subplots()
    for name, scores, color in (
        (model_name, probability, "#2463a6"),
        ("Recency baseline", baseline, "#b85c00"),
    ):
        precision, recall, _ = precision_recall_curve(y, scores)
        pr_ax.step(
            recall,
            precision,
            where="post",
            color=color,
            label=f"{name}: AP {average_precision_score(y, scores):.3f}",
        )
        fpr, tpr, _ = roc_curve(y, scores)
        roc_ax.plot(fpr, tpr, color=color, label=f"{name}: AUC {roc_auc_score(y, scores):.3f}")

    pr_ax.axhline(
        y.mean(), color="grey", linestyle="--", label=f"Return prevalence: {y.mean():.1%}"
    )
    roc_ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Random ranking: AUC 0.500")
    selected = probability >= threshold
    tp = int(np.sum(selected & (y == 1)))
    fp = int(np.sum(selected & (y == 0)))
    recall_at_threshold = tp / int(y.sum())
    precision_at_threshold = tp / int(selected.sum()) if selected.any() else None
    marker = f"Model threshold {threshold:.1%}"
    if precision_at_threshold is not None:
        pr_ax.plot(recall_at_threshold, precision_at_threshold, "o", color="#212121", label=marker)
    roc_ax.plot(fp / int((y == 0).sum()), recall_at_threshold, "o", color="#212121", label=marker)

    for ax, title, xlabel, ylabel in (
        (pr_ax, "Precision–Recall", "Recall = TP / (TP + FN)", "Precision = TP / (TP + FP)"),
        (roc_ax, "ROC", "False-positive rate = FP / (FP + TN)", "Recall = TP / (TP + FN)"),
    ):
        ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel=xlabel, ylabel=ylabel)
        ax.set_title(f"{title} — held-out test\nCutoff: {snapshot}; {len(y):,} customers")
        ax.grid(alpha=0.2)
        ax.legend(loc="lower left" if ax is pr_ax else "lower right", fontsize=9)
    pr.tight_layout()
    roc.tight_layout()
    return dict(zip(FIGURE_NAMES, (pr, roc)))


def save_ranking_figures(directory: Path, **kwargs: Any) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fig in ranking_figures(**kwargs).items():
        path = directory / name
        fig.savefig(path, dpi=150)
        paths.append(path)
    return paths


def main() -> None:
    predictions = pd.read_parquet(ARTIFACTS / "test_predictions.parquet")
    snapshots = pd.read_parquet(PROCESSED / "snapshots.parquet")
    features = pd.read_parquet(PROCESSED / "X.parquet")
    test = snapshots.loc[snapshots["split"] == "test"]
    if not np.array_equal(
        predictions["customer_id"].astype(str).to_numpy(),
        test["customer_id"].astype(str).to_numpy(),
    ):
        raise ValueError("Saved predictions do not match the test snapshot's customer order")
    report = json.loads((PROJECT_ROOT / "reports/current_run/retention_metrics.json").read_text())
    y, scores = predictions["actual_will_return"], predictions["probability"]
    baseline = 1 / (1 + features.loc[test.index, "recency_days"].to_numpy())
    # Refuse mixed/stale artifacts instead of displaying curves with different headline scores.
    for values, expected in (
        (scores, report["final_test"]),
        (baseline, report["final_test_baselines"]["business_heuristic_recency"]),
    ):
        if not (
            np.isclose(average_precision_score(y, values), expected["pr_auc"], atol=1e-9)
            and np.isclose(roc_auc_score(y, values), expected["roc_auc"], atol=1e-9)
        ):
            raise ValueError("Saved predictions disagree with the reported test metrics")
    paths = save_ranking_figures(
        PROJECT_ROOT / "reports/current_run/figures",
        y_true=y,
        probabilities=scores,
        recency_scores=baseline,
        threshold=report["threshold_selection"]["best_threshold"],
        model_name=report["selected_model"],
        snapshot=", ".join(sorted(test["cutoff"].unique())),
    )
    for path in paths:
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
