"""Managed-job requests enforce data separation and bounded cloud spending."""

import pytest

from scripts.vertex_jobs import job_spec, tuning_request


def configuration():
    return {
        "project": "test",
        "region": "northamerica-northeast1",
        "bucket": "test-data",
        "images": {"vertex": "test-image@sha256:abc"},
        "mlflow_url": "https://private.run.app",
        "git_sha": "abc",
        "accounts": {"train": "train@test.iam.gserviceaccount.com"},
    }


def test_training_has_spot_and_finite_compute_and_queue_budgets():
    request = job_spec(configuration(), "gs://test-data/output")
    assert request["scheduling"]["strategy"] == "SPOT"
    assert request["scheduling"]["timeout"] == "600s"
    assert request["scheduling"]["maxWaitDuration"] == "600s"
    worker = request["workerPoolSpecs"][0]
    assert worker["replicaCount"] == "1"
    assert "gs://test-data/processed" in worker["containerSpec"]["args"]
    assert request["serviceAccount"] == "train@test.iam.gserviceaccount.com"


def test_hpo_never_optimises_the_final_test_snapshot():
    request = tuning_request(configuration(), "gs://test-data/tuning")
    assert request["studySpec"]["metrics"] == [
        {"metricId": "validation_pr_auc", "goal": "MAXIMIZE"}
    ]
    assert request["maxTrialCount"] == 12
    assert request["parallelTrialCount"] == 2
    assert "hpo-trial" in request["trialJobSpec"]["workerPoolSpecs"][0]["containerSpec"]["args"]


def test_trial_budget_cannot_be_accidentally_exceeded():
    with pytest.raises(ValueError):
        tuning_request(configuration(), "gs://test-data/tuning", trials=100)
