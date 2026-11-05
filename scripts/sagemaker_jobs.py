"""SageMaker work for brief section 3.5: launch, keep the evidence, tear down. Needs AWS.

  upload-data  copy data/processed (X, y, snapshots, feature metadata) to S3
  train        one training job running the full pipeline (script mode, managed spot)
  tune         a Bayesian hyperparameter tuning job of the random forest (managed spot)
  deploy       a real-time endpoint serving a training job's model
  invoke       one real historical customer sent to the endpoint
  teardown     delete the endpoint, its configuration and its model
  compare      the tuning job against the local Optuna study: best score, wall-clock,
               cost and the crossover point at which managed tuning becomes worthwhile

Region ca-central-1. Settings come from the environment: SAGEMAKER_ROLE_ARN,
SAGEMAKER_IMAGE_URI (the ECR image built from docker/Dockerfile, target "sagemaker") and
SAGEMAKER_BUCKET. Every command saves what AWS returned in reports/current_run/sagemaker/.
Request bodies come from pure functions, so they are unit-tested without an AWS account.
Prices are never assumed: compare takes the hourly price you read on the AWS pricing page.
"""

import argparse
import json
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from src.contracts import FEATURE_NAMES, PROCESSED, PROJECT_ROOT
from src.envfile import load_env_file

REGION = "ca-central-1"
PREFIX = "boreal-marche"
EVIDENCE = PROJECT_ROOT / "reports" / "current_run" / "sagemaker"
METRIC = {"Name": "validation_pr_auc", "Regex": r"validation_pr_auc=([0-9.]+)"}
# The same ranges as the random-forest Optuna study in src/train.py (trial_model).
RF_RANGES = {
    "IntegerParameterRanges": [
        {"Name": "n-estimators", "MinValue": "100", "MaxValue": "250", "ScalingType": "Auto"},
        {"Name": "max-depth", "MinValue": "4", "MaxValue": "12", "ScalingType": "Auto"},
        {"Name": "min-samples-leaf", "MinValue": "2", "MaxValue": "25", "ScalingType": "Auto"},
    ]
}
DATA_FILES = ["X.parquet", "y.parquet", "snapshots.parquet", "feature_metadata.json"]


def _job_resources(
    role_arn: str, data_s3: str, output_s3: str, instance_type: str, max_seconds: int, spot: bool
) -> dict:
    stopping = {"MaxRuntimeInSeconds": max_seconds}
    if spot:
        stopping["MaxWaitTimeInSeconds"] = 2 * max_seconds  # time allowed to wait for capacity
    source = {
        "S3DataType": "S3Prefix",
        "S3Uri": data_s3,
        "S3DataDistributionType": "FullyReplicated",
    }
    return {
        "RoleArn": role_arn,
        "InputDataConfig": [{"ChannelName": "train", "DataSource": {"S3DataSource": source}}],
        "OutputDataConfig": {"S3OutputPath": output_s3},
        "ResourceConfig": {"InstanceType": instance_type, "InstanceCount": 1, "VolumeSizeInGB": 10},
        "StoppingCondition": stopping,
        "EnableManagedSpotTraining": spot,
    }


def _algorithm(image_uri: str) -> dict:
    return {"TrainingImage": image_uri, "TrainingInputMode": "File", "MetricDefinitions": [METRIC]}


def training_job_request(
    name: str,
    image_uri: str,
    role_arn: str,
    data_s3: str,
    output_s3: str,
    *,
    instance_type: str = "ml.m5.large",
    trials: int = 12,
    spot: bool = True,
    max_seconds: int = 3600,
) -> dict:
    return {
        "TrainingJobName": name,
        "AlgorithmSpecification": _algorithm(image_uri),
        "HyperParameters": {"mode": "full", "trials": str(trials)},
        **_job_resources(role_arn, data_s3, output_s3, instance_type, max_seconds, spot),
    }


def tuning_job_request(
    name: str,
    image_uri: str,
    role_arn: str,
    data_s3: str,
    output_s3: str,
    *,
    instance_type: str = "ml.m5.large",
    max_jobs: int = 12,
    max_parallel: int = 3,
    spot: bool = True,
    max_seconds: int = 1800,
) -> dict:
    return {
        "HyperParameterTuningJobName": name,  # at most 32 characters
        "HyperParameterTuningJobConfig": {
            "Strategy": "Bayesian",
            "HyperParameterTuningJobObjective": {"Type": "Maximize", "MetricName": METRIC["Name"]},
            "ResourceLimits": {
                "MaxNumberOfTrainingJobs": max_jobs,
                "MaxParallelTrainingJobs": max_parallel,
            },
            "ParameterRanges": RF_RANGES,
            "TrainingJobEarlyStoppingType": "Off",
        },
        "TrainingJobDefinition": {
            "AlgorithmSpecification": _algorithm(image_uri),
            "StaticHyperParameters": {"mode": "hpo-trial"},
            **_job_resources(role_arn, data_s3, output_s3, instance_type, max_seconds, spot),
        },
    }


def endpoint_requests(
    name: str, image_uri: str, role_arn: str, model_data_s3: str, instance_type: str
) -> dict[str, dict]:
    return {
        "model": {
            "ModelName": name,
            "ExecutionRoleArn": role_arn,
            "PrimaryContainer": {"Image": image_uri, "ModelDataUrl": model_data_s3},
        },
        "config": {
            "EndpointConfigName": name,
            "ProductionVariants": [
                {
                    "VariantName": "AllTraffic",
                    "ModelName": name,
                    "InstanceType": instance_type,
                    "InitialInstanceCount": 1,
                }
            ],
        },
        "endpoint": {"EndpointName": name, "EndpointConfigName": name},
    }


def crossover(
    local_trial_seconds: float,
    cloud_trial_seconds: float,
    overhead_seconds: float,
    parallel: int,
    trials: int,
) -> dict:
    """Wall-clock of *trials* sequential local trials against parallel managed trials.

    Managed jobs pay a start-up overhead each, but run *parallel* at a time. The crossover
    is the local per-trial duration above which the managed study finishes first.
    """
    waves = math.ceil(trials / parallel)
    local = trials * local_trial_seconds
    managed = waves * (cloud_trial_seconds + overhead_seconds)
    breakeven = waves * (cloud_trial_seconds + overhead_seconds) / trials
    return {
        "trials": trials,
        "local_wall_clock_s": local,
        "managed_wall_clock_s": managed,
        "managed_faster": managed < local,
        "local_trial_seconds_at_crossover": breakeven,
    }


# ---------------------------------------------------------------------------
# Commands (need AWS credentials; boto3 is in the "sagemaker" extra)
# ---------------------------------------------------------------------------
def _settings() -> dict:
    keys = ["SAGEMAKER_ROLE_ARN", "SAGEMAKER_IMAGE_URI", "SAGEMAKER_BUCKET"]
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        raise SystemExit(f"Set {', '.join(missing)} (see docs/SAGEMAKER.md)")
    bucket = os.environ["SAGEMAKER_BUCKET"]
    return {
        "role": os.environ["SAGEMAKER_ROLE_ARN"],
        "image": os.environ["SAGEMAKER_IMAGE_URI"],
        "data": f"s3://{bucket}/{PREFIX}/processed/",
        "output": f"s3://{bucket}/{PREFIX}/output/",
        "bucket": bucket,
    }


def _client(service: str):
    import boto3

    return boto3.client(service, region_name=REGION)


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d%H%M")


def _evidence(name: str, payload: dict) -> Path:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"evidence: {path.relative_to(PROJECT_ROOT)}")
    return path


def upload_data(args) -> None:
    cfg, s3 = _settings(), _client("s3")
    for name in DATA_FILES:
        s3.upload_file(str(PROCESSED / name), cfg["bucket"], f"{PREFIX}/processed/{name}")
    _evidence(f"upload_{_stamp()}", {"uploaded": DATA_FILES, "to": cfg["data"]})


def train(args) -> None:
    cfg, sm = _settings(), _client("sagemaker")
    name = f"bm-train-{_stamp()}"
    request = training_job_request(
        name, cfg["image"], cfg["role"], cfg["data"], cfg["output"], instance_type=args.instance
    )
    sm.create_training_job(**request)
    sm.get_waiter("training_job_completed_or_stopped").wait(TrainingJobName=name)
    _evidence(name, sm.describe_training_job(TrainingJobName=name))


def tune(args) -> None:
    cfg, sm = _settings(), _client("sagemaker")
    name = f"bm-hpo-{_stamp()}"
    request = tuning_job_request(
        name,
        cfg["image"],
        cfg["role"],
        cfg["data"],
        cfg["output"],
        instance_type=args.instance,
        max_jobs=args.max_jobs,
        max_parallel=args.parallel,
    )
    sm.create_hyper_parameter_tuning_job(**request)
    while True:
        job = sm.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=name)
        if job["HyperParameterTuningJobStatus"] in {"Completed", "Stopped", "Failed"}:
            break
        time.sleep(60)
    trials = sm.list_training_jobs_for_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=name, MaxResults=100
    )["TrainingJobSummaries"]
    details = [sm.describe_training_job(TrainingJobName=t["TrainingJobName"]) for t in trials]
    _evidence(name, {"tuning_job": job, "training_jobs": details})


def deploy(args) -> None:
    cfg, sm = _settings(), _client("sagemaker")
    job = sm.describe_training_job(TrainingJobName=args.training_job)
    name = f"bm-endpoint-{_stamp()}"
    requests = endpoint_requests(
        name, cfg["image"], cfg["role"], job["ModelArtifacts"]["S3ModelArtifacts"], args.instance
    )
    sm.create_model(**requests["model"])
    sm.create_endpoint_config(**requests["config"])
    sm.create_endpoint(**requests["endpoint"])
    sm.get_waiter("endpoint_in_service").wait(EndpointName=name)
    _evidence(name, {"requests": requests, "endpoint": sm.describe_endpoint(EndpointName=name)})


def invoke(args) -> None:
    import pandas as pd

    store = pd.read_parquet(PROCESSED / "customer_features.parquet")
    instance = store.loc[args.customer, FEATURE_NAMES].astype(float).to_dict()
    runtime = _client("sagemaker-runtime")
    start = time.perf_counter()
    response = runtime.invoke_endpoint(
        EndpointName=args.endpoint,
        ContentType="application/json",
        Body=json.dumps({"instances": [instance]}),
    )
    latency = (time.perf_counter() - start) * 1000
    body = json.loads(response["Body"].read())
    _evidence(
        f"invoke_{args.endpoint}",
        {"customer": args.customer, "latency_ms": latency, "response": body},
    )


def teardown(args) -> None:
    sm = _client("sagemaker")
    sm.delete_endpoint(EndpointName=args.endpoint)
    sm.get_waiter("endpoint_deleted").wait(EndpointName=args.endpoint)
    sm.delete_endpoint_config(EndpointConfigName=args.endpoint)
    sm.delete_model(ModelName=args.endpoint)
    remaining = sm.list_endpoints(NameContains=args.endpoint)["Endpoints"]
    _evidence(f"teardown_{args.endpoint}", {"deleted": args.endpoint, "remaining": remaining})


def compare(args) -> None:
    from src import train as training

    metrics = json.loads((PROJECT_ROOT / "reports/current_run/retention_metrics.json").read_text())
    forest = [t for t in metrics["optuna_trials"] if t["family"] == "RandomForestClassifier"]
    best = max(forest, key=lambda t: t["validation_pr_auc"])
    params = {k: int(v) for k, v in best["params"].items()}
    timing_dir = PROJECT_ROOT / "data" / "registry_cache" / "local_timing"
    seconds = []
    for _ in range(args.repeat):  # measured on this machine, the same split as the cloud
        training.hpo_trial(PROCESSED, timing_dir, params)
        seconds.append(json.loads((timing_dir / "hpo_trial.json").read_text())["fit_seconds"])
    tuning = json.loads(Path(args.tuning_evidence).read_text())
    jobs = tuning["training_jobs"]
    billable = sum(j.get("BillableTimeInSeconds", 0) for j in jobs)
    training_seconds = [j["TrainingTimeInSeconds"] for j in jobs]
    best_cloud = tuning["tuning_job"]["BestTrainingJob"]
    created, ended = (
        tuning["tuning_job"][k] for k in ("CreationTime", "HyperParameterTuningEndTime")
    )
    wall = (
        datetime.fromisoformat(str(ended)) - datetime.fromisoformat(str(created))
    ).total_seconds()
    local_trial = sum(seconds) / len(seconds)
    report = {
        "local_optuna": {
            "trials": len(forest),
            "best_validation_pr_auc": best["validation_pr_auc"],
            "best_params": params,
            "measured_seconds_per_trial": local_trial,
            "cost": "no additional spend (existing laptop)",
        },
        "sagemaker": {
            "trials": len(jobs),
            "best_validation_pr_auc": best_cloud["FinalHyperParameterTuningJobObjectiveMetric"][
                "Value"
            ],
            "best_params": best_cloud["TunedHyperParameters"],
            "wall_clock_seconds": wall,
            "billable_seconds": billable,
            "spot_saving": 1 - billable / sum(training_seconds) if sum(training_seconds) else None,
            "price_per_hour_usd": args.price_per_hour,
            "price_source": args.price_source,
            "cost_usd": billable / 3600 * args.price_per_hour,
        },
        "crossover": crossover(
            local_trial,
            sum(training_seconds) / len(training_seconds),
            args.overhead_seconds,
            tuning["tuning_job"]["HyperParameterTuningJobConfig"]["ResourceLimits"][
                "MaxParallelTrainingJobs"
            ],
            len(jobs),
        ),
    }
    _evidence("comparison_optuna_vs_sagemaker", report)


def main() -> None:
    load_env_file()  # SAGEMAKER_ROLE_ARN and friends may live in .env
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("upload-data")
    for name, instance in (("train", "ml.m5.large"), ("tune", "ml.m5.large")):
        command = commands.add_parser(name)
        command.add_argument("--instance", default=instance)
        if name == "tune":
            command.add_argument("--max-jobs", type=int, default=12)
            command.add_argument("--parallel", type=int, default=3)
    deploy_cmd = commands.add_parser("deploy")
    deploy_cmd.add_argument("--training-job", required=True)
    deploy_cmd.add_argument("--instance", default="ml.t2.medium")
    for name in ("invoke", "teardown"):
        command = commands.add_parser(name)
        command.add_argument("--endpoint", required=True)
        if name == "invoke":
            command.add_argument("--customer", default="12347")
    compare_cmd = commands.add_parser("compare")
    compare_cmd.add_argument("--tuning-evidence", required=True, help="JSON written by tune")
    compare_cmd.add_argument("--price-per-hour", type=float, required=True)
    compare_cmd.add_argument("--price-source", required=True, help="URL and date of the price")
    compare_cmd.add_argument("--overhead-seconds", type=float, default=180.0)
    compare_cmd.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    handlers = {
        "upload-data": upload_data,
        "train": train,
        "tune": tune,
        "deploy": deploy,
        "invoke": invoke,
        "teardown": teardown,
        "compare": compare,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
