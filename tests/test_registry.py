"""
Model registry lifecycle against a throwaway MLflow database, with SYNTHETIC data and
small stand-in models: register, refuse, promote, roll back, and serve with fallback.
The rows and scores are generated, not workbook data.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import api.lifespan as life
from src import registry
from src.contracts import CONTRACT_VERSION, FEATURE_NAMES, PROJECT_ROOT
from src.ingest import _sha256

RULES = {"max_pr_auc_drop": 0.01, "max_brier_increase": 0.01, "max_fairness_gap_widening_pp": 2}


def _frame(seed: int, rows: int = 400) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.normal(size=(rows, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    frame["recency_days"] = frame["recency_days"].abs()
    frame["country_code"] = 0
    frame["is_one_time_buyer"] = rng.integers(0, 2, rows)
    frame["days_since_first_purchase"] = rng.integers(0, 700, rows)
    frame["monetary_total"] = rng.gamma(2.0, 300.0, rows)
    y = (frame["frequency"] + rng.normal(scale=0.5, size=rows) > 0).astype(int)
    return frame, y


@pytest.fixture
def tracking(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.chdir(tmp_path)  # MLflow writes its artifact store relative to cwd
    mlflow.set_tracking_uri(uri)
    # MLflow keeps the active experiment process-wide; another test's id would not exist here.
    mlflow.set_experiment("registry-tests")
    monkeypatch.setattr(registry, "EVIDENCE_DIR", tmp_path / "evidence")
    processed = tmp_path / "processed"
    processed.mkdir()
    frame, y = _frame(seed=1)
    frame.to_parquet(processed / "X.parquet")
    y.to_frame("will_return").to_parquet(processed / "y.parquet")
    pd.DataFrame({"split": "test", "cutoff": "2020-01-01"}, index=frame.index).to_parquet(
        processed / "snapshots.parquet"
    )
    monkeypatch.setattr(registry, "PROCESSED", processed)
    return tmp_path


def _good_model() -> Pipeline:
    frame, y = _frame(seed=0)
    return Pipeline([("classifier", LogisticRegression(max_iter=500))]).fit(frame, y)


def _constant_model() -> Pipeline:
    frame, y = _frame(seed=0)
    return Pipeline([("classifier", DummyClassifier(strategy="prior"))]).fit(frame, y)


def _register(tmp_path, model, label: str) -> str:
    folder = tmp_path / label
    folder.mkdir()
    model_path = folder / registry.MODEL_FILE
    joblib.dump(model, model_path)
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, "model")
    metadata = {
        "contract_version": CONTRACT_VERSION,
        "feature_names": FEATURE_NAMES,
        "decision_threshold": 0.5,
        "target": "will_return",
        "model_sha256": _sha256(model_path),
        "country_map": {"United Kingdom": 0},
        "model_family": label,
        "trained_at": "2026-11-01T00:00:00+00:00",
        "git_commit": "test",
        "mlflow_run_id": run.info.run_id,
    }
    (folder / registry.METADATA_FILE).write_text(json.dumps(metadata))
    return registry.register_run(run.info.run_id, model_path, folder / registry.METADATA_FILE)


def _approve(version: str, tmp_path):
    return registry.promote(
        version,
        approver="Test Reviewer",
        reason="synthetic test",
        acknowledge_fairness_flags=True,
        cache_dir=tmp_path / "cache",
    )


def test_registering_creates_a_candidate_and_moves_no_alias(tracking):
    version = _register(tracking, _good_model(), "good")
    (entry,) = registry.status()
    assert (version, entry["status"], entry["aliases"]) == ("1", "candidate", [])
    assert registry.champion_version() is None


def test_promotion_requires_a_named_approver_and_a_reason(tracking):
    version = _register(tracking, _good_model(), "good")
    with pytest.raises(registry.PromotionRefusedError, match="approver"):
        registry.promote(version, approver=" ", reason="x", cache_dir=tracking / "cache")
    assert registry.champion_version() is None


def test_full_lifecycle_promote_refuse_promote_rollback(tracking):
    first = _register(tracking, _good_model(), "good")
    report = _approve(first, tracking)
    assert registry.champion_version() == first
    assert report["gates"][0]["gate"] == "beats_recency_baseline"
    assert (tracking / report["evidence_file"]).exists()

    worse = _register(tracking, _constant_model(), "constant")
    with pytest.raises(registry.PromotionRefusedError, match="pr_auc_not_worse"):
        _approve(worse, tracking)
    assert registry.champion_version() == first, "a refused promotion changes nothing"

    same = _register(tracking, _good_model(), "good-again")  # identical scores pass every gate
    _approve(same, tracking)
    tags = {entry["version"]: entry for entry in registry.status()}
    assert registry.champion_version() == same
    assert tags[same]["approved_by"] == "Test Reviewer"
    assert tags[same]["previous_champion"] == first and tags[first]["status"] == "retired"

    record = registry.rollback(approver="Test Reviewer", reason="synthetic incident")
    assert registry.champion_version() == first
    assert (record["from_version"], record["to_version"]) == (same, first)
    assert {entry["version"]: entry for entry in registry.status()}[same]["status"] == "rolled_back"


def test_rollback_without_a_previous_champion_is_refused(tracking):
    _approve(_register(tracking, _good_model(), "good"), tracking)
    with pytest.raises(registry.PromotionRefusedError, match="previous champion"):
        registry.rollback(approver="Test Reviewer", reason="nothing to go back to")


def test_api_serves_the_champion_and_falls_back_to_local_files(tracking, monkeypatch):
    version = _register(tracking, _good_model(), "good")
    _approve(version, tracking)
    monkeypatch.setenv("MODEL_SOURCE", "registry")
    monkeypatch.setattr(life, "CACHE_DIR", tracking / "api-cache")
    model, metadata, source = life._load_retention_model()
    assert (source["source"], source["version"], source["alias"]) == (
        "registry",
        version,
        "champion",
    )
    assert metadata["model_family"] == "good"

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tracking / 'empty.db'}")
    monkeypatch.setattr(life, "_MODEL_PATH", tracking / "good" / registry.MODEL_FILE)
    monkeypatch.setattr(life, "_META_PATH", tracking / "good" / registry.METADATA_FILE)
    _, _, fallback = life._load_retention_model()
    assert fallback["source"] == "local" and fallback["fallback_reason"]


def test_resolution_works_in_a_fresh_process_without_mlflow_settings(tracking):
    """Regression: with MLFLOW_TRACKING_URI never set, MLflow's own default store is
    ./mlruns, so resolving a version through it found nothing. A clean subprocess is the
    only faithful reproduction: inside pytest the fixture has already set MLflow up."""
    version = _register(tracking, _good_model(), "good")
    _approve(version, tracking)
    script = (
        "import sys; from pathlib import Path; import src.registry as r; "
        f"r.tracking_uri = lambda: {registry.tracking_uri()!r}; "
        "print(r.resolve_champion(Path(sys.argv[1]))[2]['version'])"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("MLFLOW_")}
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", script, str(tracking / "fresh-cache")],
        cwd=tracking,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-1500:]
    assert result.stdout.strip().splitlines()[-1] == version


def test_api_skips_the_registry_when_told_to(tracking, monkeypatch):
    _register(tracking, _good_model(), "good")
    monkeypatch.setattr(life, "_MODEL_PATH", tracking / "good" / registry.MODEL_FILE)
    monkeypatch.setattr(life, "_META_PATH", tracking / "good" / registry.METADATA_FILE)
    _, _, source = life._load_retention_model()  # conftest sets MODEL_SOURCE=local
    assert source == {
        "source": "local",
        "path": "artifacts/",
        "fallback_reason": "MODEL_SOURCE=local",
    }


class _Scores:
    """Stand-in model that returns fixed scores, to put a fairness gap where we want it."""

    feature_names_in_ = np.array(FEATURE_NAMES)

    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=float)

    def predict_proba(self, frame):
        return np.column_stack([1 - self.scores, self.scores])


def test_gates_refuse_a_new_fairness_flag_and_a_changed_encoding():
    frame, _ = _frame(seed=2, rows=400)
    y = np.r_[np.ones(200), np.zeros(200)]
    frame["is_one_time_buyer"] = np.r_[np.zeros(100), np.ones(100), np.zeros(200)]
    fair = np.r_[np.full(200, 0.9), np.full(200, 0.1)]  # every returner found
    unfair = fair.copy()
    unfair[100:200] = 0.1  # one-time returners missed: a new strong flag
    meta = {"decision_threshold": 0.5, "country_map": {"United Kingdom": 0}}
    report = registry.check_gates((_Scores(unfair), meta), (_Scores(fair), meta), frame, y, RULES)
    failed = {g["gate"] for g in report["gates"] if not g["passed"]}
    assert "no_new_fairness_flags" in failed and not report["passed"]
    other_map = {**meta, "country_map": {"France": 0}}
    report = registry.check_gates(
        (_Scores(fair), other_map), (_Scores(fair), meta), frame, y, RULES
    )
    assert not next(g for g in report["gates"] if g["gate"] == "same_feature_encoding")["passed"]
