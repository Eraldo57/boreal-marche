"""Bounded Vertex AI training/HPO and a temporary endpoint with guaranteed cleanup."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from datetime import UTC, datetime

import requests

from scripts.gcp import EVIDENCE, ROOT, api, gcloud, save


def job_spec(cfg: dict, output_uri: str, mode: str = "full") -> dict:
    """The worker uses the same training implementation and chronological splits as local."""
    return {
        "workerPoolSpecs": [
            {
                "machineSpec": {"machineType": "n1-standard-4"},
                "replicaCount": "1",
                "containerSpec": {
                    "imageUri": cfg["images"]["vertex"],
                    "command": ["python", "-m", "scripts.vertex_entry"],
                    "args": [
                        "train",
                        "--mode",
                        mode,
                        "--trials",
                        "12",
                        "--data-uri",
                        f"gs://{cfg['bucket']}/processed",
                        "--output-uri",
                        output_uri,
                    ],
                    "env": [
                        {"name": key, "value": value}
                        for key, value in {
                            "MLFLOW_TRACKING_URI": cfg["mlflow_url"],
                            "MLFLOW_GOOGLE_IAM_AUTH": "true",
                            "GIT_SHA": cfg["git_sha"],
                            "MLFLOW_HTTP_REQUEST_TIMEOUT": "60",
                        }.items()
                    ],
                },
            }
        ],
        "serviceAccount": cfg["accounts"]["train"],
        "scheduling": {
            "strategy": "SPOT",
            "timeout": "600s",
            "restartJobOnWorkerRestart": False,
        },
        "baseOutputDirectory": {"outputUriPrefix": output_uri.rsplit("/", 1)[0]},
    }


def tuning_request(cfg: dict, output_uri: str, trials: int = 6) -> dict:
    if not 1 <= trials <= 12:
        raise ValueError("The capstone tuning budget permits 1–12 trials")
    trial_spec = job_spec(cfg, output_uri, "hpo-trial")
    # Allow startup/tracking delays and bounded Spot failures while the launcher
    # still cancels the entire study after one hour with at most two workers.
    trial_spec["scheduling"]["timeout"] = "1800s"
    return {
        "displayName": "boreal-bayesian-spot",
        # Omitted algorithm selects Vertex's default Bayesian optimisation.
        "studySpec": {
            "metrics": [{"metricId": "validation_pr_auc", "goal": "MAXIMIZE"}],
            "parameters": [
                {
                    "parameterId": name,
                    "integerValueSpec": {"minValue": low, "maxValue": high},
                    "scaleType": "UNIT_LINEAR_SCALE",
                }
                for name, low, high in (
                    ("n-estimators", 100, 250),
                    ("max-depth", 4, 12),
                    ("min-samples-leaf", 2, 25),
                )
            ],
        },
        "maxTrialCount": trials,
        "parallelTrialCount": min(2, trials),
        "maxFailedTrialCount": min(4, trials),
        "trialJobSpec": trial_spec,
    }


def wait_job(cfg: dict, name: str, timeout: int) -> dict:
    url = f"https://{cfg['region']}-aiplatform.googleapis.com/v1/{name}"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = api(cfg, "GET", url)
        state = job["state"]
        save("last_vertex_job", job)
        print(f"{name.rsplit('/', 1)[-1]}: {state}", flush=True)
        if state == "JOB_STATE_SUCCEEDED":
            return job
        if state in {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}:
            raise RuntimeError(json.dumps(job.get("error", job)))
        time.sleep(20)
    api(cfg, "POST", url + ":cancel", {})
    raise TimeoutError("Vertex job cancelled after its wall-clock budget expired")


def upload_inputs(cfg: dict) -> None:
    for name in ("X.parquet", "y.parquet", "snapshots.parquet", "feature_metadata.json"):
        gcloud(
            cfg,
            "storage",
            "cp",
            str(ROOT / "data/processed" / name),
            f"gs://{cfg['bucket']}/processed/{name}",
        )


def train(cfg: dict, *, tune: bool = False, trials: int = 6) -> dict:
    upload_inputs(cfg)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    name = "hpo" if tune else "full"
    output = f"gs://{cfg['bucket']}/training/{name}-{stamp}/model"
    base = (
        f"https://{cfg['region']}-aiplatform.googleapis.com/v1/projects/{cfg['project']}"
        f"/locations/{cfg['region']}"
    )
    body = (
        tuning_request(cfg, output, trials)
        if tune
        else {
            "displayName": f"boreal-full-{stamp}",
            "jobSpec": job_spec(cfg, output),
        }
    )
    resource = "hyperparameterTuningJobs" if tune else "customJobs"
    job = api(cfg, "POST", f"{base}/{resource}", body)
    save(f"vertex_{name}_created", job)
    result = wait_job(cfg, job["name"], 3600 if tune else 1500)
    result["artifact_uri"] = output
    save(f"vertex_{name}", result)
    if not tune:
        cfg["trained_artifact_uri"] = output
        save("settings", cfg)
    return result


def endpoint_demo(cfg: dict) -> None:
    model = gcloud(
        cfg,
        "ai",
        "models",
        "upload",
        f"--region={cfg['region']}",
        "--display-name=boreal-retention",
        f"--artifact-uri={cfg['trained_artifact_uri']}",
        f"--container-image-uri={cfg['images']['vertex']}",
        "--container-command=python,-m,scripts.vertex_entry",
        "--container-args=serve",
        "--container-health-route=/ping",
        "--container-predict-route=/invocations",
        "--container-ports=8080",
        "--version-aliases=candidate",
        as_json=True,
    )
    save("vertex_model", model)
    endpoint = gcloud(
        cfg,
        "ai",
        "endpoints",
        "create",
        f"--region={cfg['region']}",
        "--display-name=boreal-temporary-evidence",
        as_json=True,
    )
    name = endpoint["name"]
    save("vertex_endpoint_created", endpoint)
    evidence = {"endpoint": name, "model": model.get("model", model.get("name"))}
    try:
        resource = model.get("model", model.get("name"))
        gcloud(
            cfg,
            "ai",
            "endpoints",
            "deploy-model",
            name,
            f"--region={cfg['region']}",
            f"--model={resource}",
            "--display-name=boreal-demo",
            "--machine-type=n1-standard-2",
            "--min-replica-count=1",
            "--max-replica-count=1",
            "--traffic-split=0=100",
            f"--service-account={cfg['accounts']['train']}",
        )
        record = requests.get(f"{cfg['api_url']}/customers/12347/features", timeout=60)
        record.raise_for_status()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as file:
            json.dump({"instances": [record.json()["features"]]}, file)
            file.flush()
            evidence["prediction"] = gcloud(
                cfg,
                "ai",
                "endpoints",
                "predict",
                name,
                f"--region={cfg['region']}",
                f"--json-request={file.name}",
                as_json=True,
            )
        save("vertex_endpoint_invocation", evidence)
    finally:
        current = gcloud(
            cfg, "ai", "endpoints", "describe", name, f"--region={cfg['region']}", as_json=True
        )
        for deployed in current.get("deployedModels", []):
            gcloud(
                cfg,
                "ai",
                "endpoints",
                "undeploy-model",
                name,
                f"--region={cfg['region']}",
                f"--deployed-model-id={deployed['id']}",
            )
        gcloud(cfg, "ai", "endpoints", "delete", name, f"--region={cfg['region']}")
        remaining = gcloud(
            cfg, "ai", "endpoints", "list", f"--region={cfg['region']}", as_json=True
        )
        save("vertex_endpoint_teardown", {"deleted": name, "remaining_endpoints": remaining})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "tune", "endpoint-demo"])
    parser.add_argument("--trials", type=int, default=6)
    args = parser.parse_args()
    cfg = json.loads((EVIDENCE / "settings.json").read_text())
    cfg.update(
        json.loads(gcloud(cfg, "storage", "cat", f"gs://{cfg['bucket']}/deployment/settings.json"))
    )
    if args.command == "endpoint-demo":
        endpoint_demo(cfg)
    else:
        train(cfg, tune=args.command == "tune", trials=args.trials)


if __name__ == "__main__":
    main()
