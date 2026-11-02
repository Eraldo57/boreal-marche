"""Report figures, drawn from the numbers a training run saved.

Charts are rendered from retention_metrics.json, never recomputed, so a figure always
matches its report. src.train logs the same figures to MLflow for every run; for an
existing run, ``python -m src.plots`` redraws them into reports/current_run/figures/.
The object-oriented Figure API is used (not pyplot), so no display or backend is needed.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure

from src.contracts import PROJECT_ROOT

REPORTS = PROJECT_ROOT / "reports" / "current_run"


def _decision_period(metrics: dict[str, Any]) -> str:
    split = metrics["splits"]["decision"]
    return f"{', '.join(split['cutoffs'])} cut-off, {split['rows']:,} rows"


def reliability_figure(metrics: dict[str, Any]) -> Figure:
    """Reliability curves before calibration (uncalibrated) and after each method."""
    fig = Figure(figsize=(6.4, 6.0))
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfectly calibrated")
    selected = metrics["selected_calibration"]
    for method, data in metrics["calibration_decision_period"].items():
        label = f"{method}: Brier {data['brier_score']:.4f}"
        if method == selected:
            label += " (selected)"
        ax.plot(data["curve_pred"], data["curve_true"], marker="o", label=label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability of returning (up to 10 equal-width bins)")
    ax.set_ylabel("Observed share that returned")
    ax.set_title(
        f"Reliability before and after calibration\nDecision period: {_decision_period(metrics)}"
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def threshold_utility_figure(metrics: dict[str, Any]) -> Figure:
    """Assumed utility of every candidate threshold on the decision period."""
    analysis = metrics["threshold_selection"]
    costs = analysis["cost_assumptions"]
    fig = Figure(figsize=(7.6, 4.6))
    ax = fig.subplots()
    ax.plot(analysis["curve_thresholds"], analysis["curve_utility"])
    best = analysis["best_threshold"]
    ax.axvline(best, linestyle="--", color="grey", label=f"chosen threshold {best:.3f}")
    ax.set_xlabel("Decision threshold on P(return within 90 days)")
    ax.set_ylabel("Assumed utility (monetary units)")
    ax.set_title(
        f"Threshold selection, decision period: {_decision_period(metrics)}\n"
        f"Placeholder costs: coupon {costs['cost_coupon']:g}, "
        f"margin {costs['margin_retained']:g}, missed return {costs['cost_false_negative']:g}"
    )
    ax.legend()
    fig.tight_layout()
    return fig


FIGURES: dict[str, Callable[[dict[str, Any]], Figure]] = {
    "reliability_curve.png": reliability_figure,
    "threshold_utility.png": threshold_utility_figure,
}


def save_figures(metrics: dict[str, Any], directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, draw in FIGURES.items():
        path = directory / name
        draw(metrics).savefig(path, dpi=150)
        paths.append(path)
    return paths


def main() -> None:
    metrics = json.loads((REPORTS / "retention_metrics.json").read_text())
    for path in save_figures(metrics, REPORTS / "figures"):
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
