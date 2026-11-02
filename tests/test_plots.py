"""Report figures render from the committed retention report (real run outputs)."""

from __future__ import annotations

import json

from src.plots import FIGURES, REPORTS, reliability_figure, save_figures

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _metrics() -> dict:
    return json.loads((REPORTS / "retention_metrics.json").read_text())


def test_every_figure_is_saved_as_png(tmp_path):
    paths = save_figures(_metrics(), tmp_path)
    assert [p.name for p in paths] == list(FIGURES)
    assert all(p.read_bytes().startswith(PNG_SIGNATURE) for p in paths)


def test_reliability_figure_plots_each_method_and_marks_the_selected_one():
    metrics = _metrics()
    (ax,) = reliability_figure(metrics).axes
    labels = [line.get_label() for line in ax.get_lines()]
    assert labels[0] == "Perfectly calibrated"
    assert len(labels) == 1 + len(metrics["calibration_decision_period"])
    assert sum("(selected)" in label for label in labels) == 1
    assert any(label.startswith(metrics["selected_calibration"]) for label in labels[1:])
