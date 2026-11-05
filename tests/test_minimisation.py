"""
Drop-column necessity test on SYNTHETIC data: one feature carries the signal, one is
pure noise. Generated numbers, not workbook rows. Plus one integration check on the real
model, skipped when the local artifacts are absent.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import minimisation
from src.contracts import FEATURE_NAMES, METADATA_PATH, PROCESSED

SPLITS = ["train", "validation", "calibration", "decision"]


def _synthetic():
    rng = np.random.default_rng(3)
    n = 400 * len(SPLITS)
    frame = pd.DataFrame({"signal": rng.normal(size=n), "noise": rng.normal(size=n)})
    y = pd.Series((frame["signal"] + rng.normal(scale=0.7, size=n) > 0).astype(int))
    splits = {name: frame.index[i * 400 : (i + 1) * 400] for i, name in enumerate(SPLITS)}
    return Pipeline([("classifier", LogisticRegression())]), frame, y, splits


def test_signal_is_necessary_and_noise_is_not_shown():
    pipeline, frame, y, splits = _synthetic()
    report = minimisation.audit(pipeline, frame, y, splits, "sigmoid", seed=0, draws=100)
    rows = {r["feature"]: r for r in report["features"]}
    assert rows["signal"]["necessary"] and rows["signal"]["ci95_low"] > 0
    assert not rows["noise"]["necessary"]
    joint = report["joint_removal"]
    assert joint["removed"] == ["noise"] and joint["kept"] == ["signal"]
    assert joint["removal_recommended"]


def test_identical_predictions_lose_nothing():
    y = np.array([0, 1, 1, 0, 1])
    p = np.array([0.1, 0.8, 0.6, 0.3, 0.9])
    draws = [np.arange(5), np.array([0, 1, 1, 2, 4])]
    assert minimisation.paired_bootstrap_loss(y, p, p, draws) == {
        "pr_auc_loss": 0.0,
        "ci95_low": 0.0,
        "ci95_high": 0.0,
    }


def test_markdown_table_marks_verdicts_and_joint_result():
    pipeline, frame, y, splits = _synthetic()
    report = minimisation.audit(pipeline, frame, y, splits, "sigmoid", seed=0, draws=50)
    report.update(decision_cutoff="2020-01-01", decision_rows=400)
    table = minimisation.markdown_table(report)
    assert "| signal |" in table and "| yes |" in table and "| not shown |" in table
    assert "Removing all 1 features" in table


@pytest.mark.integration
@pytest.mark.skipif(
    not METADATA_PATH.exists() or not (PROCESSED / "X.parquet").exists(),
    reason="Run the real local pipeline to generate the model and features",
)
def test_committed_minimisation_report_matches_the_saved_model():
    committed = json.loads((minimisation.REPORTS / "minimisation_audit.json").read_text())
    metadata = json.loads(METADATA_PATH.read_text())
    assert committed["model_sha256"] == metadata["model_sha256"], "re-run src.minimisation"
    assert {row["feature"] for row in committed["features"]} == set(FEATURE_NAMES)
    assert committed["note"].startswith("Recommendation only")
