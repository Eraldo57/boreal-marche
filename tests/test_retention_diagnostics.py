"""Paired uncertainty must preserve the identity comparison and confusion counts."""

import numpy as np
import pytest

from scripts.evaluate_retention import classification, uncertainty


def test_confusion_counts_keep_selection_and_outcome_separate():
    result = classification(np.array([1, 0, 1, 0]), np.array([True, True, False, False]))
    assert [result[k] for k in ["tp", "fp", "fn", "tn"]] == [1, 1, 1, 1]
    assert result["precision"] == result["recall"] == result["selection_rate"] == 0.5


def test_identical_rankings_have_zero_paired_ap_difference():
    y = np.tile([0, 1], 20)
    p = np.linspace(0.01, 0.99, len(y))
    report = uncertainty(y, p, p, 0.5, draws=30)
    assert report["ci95"]["ap_difference"] == [0.0, 0.0]
    assert report["draws_used"] == 30
    assert report == uncertainty(y, p, p, 0.5, draws=30)


def test_bootstrap_refuses_an_invalid_single_class_study():
    with pytest.raises(ValueError, match="Both classes"):
        uncertainty(np.ones(5), np.ones(5), np.ones(5), 0.5, draws=30)
