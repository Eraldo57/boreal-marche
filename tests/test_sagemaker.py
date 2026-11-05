"""
SageMaker pieces that run without AWS: the container's train contract, the endpoint app
and the request bodies the launcher sends. Synthetic data and stand-in models only.
"""

from __future__ import annotations

import json
import re

import joblib
import pytest
from fastapi.testclient import TestClient

from scripts import sagemaker_entry, sagemaker_jobs
from src import registry
from src.contracts import CONTRACT_VERSION, FEATURE_NAMES
from tests.test_registry import _good_model
from tests.test_train_run import _synthetic_inputs


def test_container_train_contract_runs_the_same_training_file(tmp_path, monkeypatch):
    base = tmp_path / "opt_ml"
    channel = base / "input" / "data" / "train"
    channel.mkdir(parents=True)
    _synthetic_inputs(channel)
    (base / "input" / "config").mkdir(parents=True)
    hyperparameters = {"mode": '"hpo-trial"', "n-estimators": "20", "max-depth": "3"}
    (base / "input" / "config" / "hyperparameters.json").write_text(json.dumps(hyperparameters))
    for name in ("SM_CHANNEL_TRAIN", "SM_MODEL_DIR"):
        monkeypatch.delenv(name, raising=False)
    sagemaker_entry.train(base)
    result = json.loads((base / "model" / "hpo_trial.json").read_text())
    assert result["params"]["n_estimators"] == 20


def test_container_failure_is_written_where_sagemaker_reads_it(tmp_path, monkeypatch):
    base = tmp_path / "opt_ml"
    (base / "input" / "data" / "train").mkdir(parents=True)  # empty channel: training fails
    monkeypatch.setenv("SM_CHANNEL_TRAIN", str(base / "input" / "data" / "train"))
    monkeypatch.setenv("SM_MODEL_DIR", str(base / "model"))
    with pytest.raises(SystemExit):
        sagemaker_entry.train(base)
    assert "Error" in (base / "output" / "failure").read_text()


def _model_dir(tmp_path):
    folder = tmp_path / "model"
    folder.mkdir()
    joblib.dump(_good_model(), folder / registry.MODEL_FILE)
    metadata = {
        "contract_version": CONTRACT_VERSION,
        "feature_names": FEATURE_NAMES,
        "decision_threshold": 0.5,
        "target": "will_return",
        "model_sha256": registry._sha256(folder / registry.MODEL_FILE),
        "model_version": "test",
    }
    (folder / registry.METADATA_FILE).write_text(json.dumps(metadata))
    return folder


def test_endpoint_pings_and_scores_valid_instances(tmp_path, monkeypatch, sample_features):
    monkeypatch.setenv("SM_MODEL_DIR", str(_model_dir(tmp_path)))
    from api.sagemaker_serve import app

    with TestClient(app) as client:
        assert client.get("/ping").status_code == 200
        scored = client.post("/invocations", json={"instances": [sample_features] * 2})
        assert scored.status_code == 200
        predictions = scored.json()["predictions"]
        assert len(predictions) == 2 and 0 <= predictions[0]["probability"] <= 1
        broken = {**sample_features, "recency_days": -5.0}
        assert client.post("/invocations", json={"instances": [broken]}).status_code == 422


def test_endpoint_without_a_model_is_not_healthy(tmp_path, monkeypatch):
    monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path / "empty"))
    from api.sagemaker_serve import app

    with TestClient(app) as client:
        assert client.get("/ping").status_code == 503


def test_tuning_request_is_bayesian_spot_and_reads_the_printed_metric():
    request = sagemaker_jobs.tuning_job_request(
        "bm-hpo-202611010000", "image", "role", "s3://b/d/", "s3://b/o/"
    )
    config = request["HyperParameterTuningJobConfig"]
    assert config["Strategy"] == "Bayesian"
    assert config["HyperParameterTuningJobObjective"]["MetricName"] == "validation_pr_auc"
    assert len(request["HyperParameterTuningJobName"]) <= 32
    definition = request["TrainingJobDefinition"]
    assert definition["EnableManagedSpotTraining"] is True
    assert definition["StoppingCondition"]["MaxWaitTimeInSeconds"] >= 1800
    assert definition["StaticHyperParameters"] == {"mode": "hpo-trial"}
    names = {r["Name"] for r in config["ParameterRanges"]["IntegerParameterRanges"]}
    assert names == {"n-estimators", "max-depth", "min-samples-leaf"}
    regex = definition["AlgorithmSpecification"]["MetricDefinitions"][0]["Regex"]
    assert re.search(regex, "validation_pr_auc=0.712345").group(1) == "0.712345"
    assert sagemaker_jobs.REGION == "ca-central-1"


def test_training_and_endpoint_requests():
    job = sagemaker_jobs.training_job_request("bm-train-1", "image", "role", "s3://d", "s3://o")
    assert job["HyperParameters"] == {"mode": "full", "trials": "12"}
    assert job["InputDataConfig"][0]["ChannelName"] == "train"
    parts = sagemaker_jobs.endpoint_requests(
        "bm-ep", "image", "role", "s3://m.tar.gz", "ml.t2.medium"
    )
    assert parts["config"]["ProductionVariants"][0]["ModelName"] == parts["model"]["ModelName"]
    assert parts["endpoint"]["EndpointConfigName"] == "bm-ep"


def test_crossover_compares_sequential_local_and_parallel_managed_trials():
    fast_local = sagemaker_jobs.crossover(0.5, 30, 180, parallel=3, trials=12)
    assert fast_local["managed_faster"] is False
    assert fast_local["local_trial_seconds_at_crossover"] == pytest.approx(4 * 210 / 12)
    slow_local = sagemaker_jobs.crossover(600, 30, 180, parallel=3, trials=12)
    assert slow_local["managed_faster"] is True
