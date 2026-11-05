"""Test curve orientation and threshold ties on labelled synthetic predictions."""

import pytest

from src.ranking_plots import FIGURE_NAMES, ranking_figures, save_ranking_figures


def _inputs():
    return {
        "y_true": [1, 0, 1, 0],
        "probabilities": [0.8, 0.22, 0.22, 0.1],
        "recency_scores": [0.1, 0.9, 0.2, 0.8],
        "threshold": 0.22,
        "model_name": "Synthetic model",
        "snapshot": "synthetic cutoff",
    }


def test_threshold_marker_includes_scores_equal_to_threshold():
    figures = ranking_figures(**_inputs())
    pr_ax = figures[FIGURE_NAMES[0]].axes[0]
    roc_ax = figures[FIGURE_NAMES[1]].axes[0]
    pr_marker = next(line for line in pr_ax.lines if line.get_label() == "Model threshold 22.0%")
    roc_marker = next(line for line in roc_ax.lines if line.get_label() == "Model threshold 22.0%")
    assert pr_marker.get_xdata()[0] == 1.0  # recall: both returners selected
    assert pr_marker.get_ydata()[0] == pytest.approx(2 / 3)  # precision: 2 of 3 selected
    assert roc_marker.get_xdata()[0] == 0.5  # FPR: 1 of 2 non-returners selected
    assert roc_marker.get_ydata()[0] == 1.0
    assert "Synthetic model: AP 0.833" in [line.get_label() for line in pr_ax.lines]
    assert "Synthetic model: AUC 0.875" in [line.get_label() for line in roc_ax.lines]


def test_exported_curves_are_standalone_pngs(tmp_path):
    paths = save_ranking_figures(tmp_path, **_inputs())
    assert tuple(p.name for p in paths) == FIGURE_NAMES
    assert all(p.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for p in paths)


@pytest.mark.parametrize("labels", [[1, 1, 1, 1], [0, 1]])
def test_invalid_or_single_class_evaluations_are_refused(labels):
    with pytest.raises(ValueError, match="both binary classes"):
        ranking_figures(**{**_inputs(), "y_true": labels})


def test_no_selected_customers_has_no_undefined_precision_marker():
    figures = ranking_figures(**{**_inputs(), "threshold": 1.0})
    labels = [line.get_label() for line in figures[FIGURE_NAMES[0]].axes[0].lines]
    assert "Model threshold 100.0%" not in labels
