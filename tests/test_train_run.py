"""
End-to-end test of the retention training run on SYNTHETIC data.

The rows below are generated random numbers, not workbook rows. They exist only to
exercise the training code path (spot-check, tuning, calibration, threshold, artifacts)
without needing the DVC-tracked data, so the test also runs on a clean CI checkout.
"""

from __future__ import annotations

import json

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.svm import LinearSVC

from src import train
from src.contracts import FEATURE_NAMES

ROWS_PER_SPLIT = 150


def _synthetic_inputs(directory):
    rng = np.random.default_rng(7)
    frames, snapshot_rows = [], []
    for position, split in enumerate(train.SPLITS):
        cutoff = pd.Timestamp("2020-01-01") + pd.Timedelta(days=100 * position)
        frame = pd.DataFrame(
            rng.normal(size=(ROWS_PER_SPLIT, len(FEATURE_NAMES))), columns=FEATURE_NAMES
        )
        frame["recency_days"] = frame["recency_days"].abs()  # real recency is never negative
        frame["is_one_time_buyer"] = rng.integers(0, 2, ROWS_PER_SPLIT)
        frame["country_code"] = rng.integers(0, 3, ROWS_PER_SPLIT)
        frames.append(frame)
        for customer in range(ROWS_PER_SPLIT):
            snapshot_rows.append(
                {
                    "split": split,
                    "cutoff": cutoff.strftime("%Y-%m-%d"),
                    "label_end": (cutoff + pd.Timedelta(days=90)).strftime("%Y-%m-%d"),
                    "customer_id": f"S{position}-{customer}",
                }
            )
    matrix = pd.concat(frames, ignore_index=True)
    y = pd.DataFrame(
        {
            "will_return": (
                matrix["recency_days"] + rng.normal(scale=0.5, size=len(matrix)) > 0.8
            ).astype(int)
        }
    )
    snapshots = pd.DataFrame(snapshot_rows)
    matrix.to_parquet(directory / "X.parquet")
    y.to_parquet(directory / "y.parquet")
    snapshots.to_parquet(directory / "snapshots.parquet")
    (directory / "feature_metadata.json").write_text(
        json.dumps({"source_data_sha256": "0" * 64, "country_map": {"United Kingdom": 0}})
    )


def test_run_produces_artifacts_on_synthetic_data(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    processed.mkdir()
    _synthetic_inputs(processed)
    monkeypatch.setattr(train, "PROCESSED", processed)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.chdir(tmp_path)  # MLflow writes its artifact store relative to cwd
    output = tmp_path / "out"

    metadata = train.run(n_trials=1, output_dir=output)

    assert 0 <= metadata["decision_threshold"] <= 1
    assert metadata["feature_names"] == FEATURE_NAMES
    assert (output / "retention_model.joblib").exists()
    assert (output / "drift_reference.parquet").exists()
    report = json.loads((output / "reports" / "retention_metrics.json").read_text())
    assert set(report["splits"]) == set(train.SPLITS)
    assert report["optuna_trials"], "tuning trials must be logged"
    stored = json.loads((output / "retention_model_metadata.json").read_text())
    assert stored["model_sha256"] == metadata["model_sha256"]


def test_run_rejects_misaligned_snapshots(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    processed.mkdir()
    _synthetic_inputs(processed)
    snapshots = pd.read_parquet(processed / "snapshots.parquet").iloc[:-1]
    snapshots.to_parquet(processed / "snapshots.parquet")
    monkeypatch.setattr(train, "PROCESSED", processed)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    with pytest.raises(ValueError, match="contract mismatch"):
        train.run(n_trials=0, output_dir=tmp_path / "out")


@pytest.mark.parametrize(
    "name",
    [
        "LogisticRegression",
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        "HistGradientBoostingClassifier",
    ],
)
def test_trial_model_builds_each_tunable_family(name):
    trial = optuna.create_study().ask()
    model = train.trial_model(name, trial, seed=0)
    assert hasattr(model, "fit")


def test_get_probas_uses_decision_function_for_margin_models():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(60, 3)), columns=["a", "b", "c"])
    labels = (frame["a"] > 0).astype(int)
    model = LinearSVC().fit(frame, labels)
    probabilities = train._get_probas(model, frame)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_recency_heuristic_orders_recent_customers_first():
    heuristic = train.RecencyHeuristic().fit(None, None)
    scores = heuristic.predict_proba(pd.DataFrame({"recency_days": [1, 100]}))[:, 1]
    assert scores[0] > scores[1]
