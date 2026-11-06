"""Run the existing retention pipeline in Vertex AI or serve its verified artifact."""

import argparse
import json
import os
import tempfile
from pathlib import Path


def split_gs(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://") or "/" not in uri[5:]:
        raise ValueError("Expected gs://bucket/prefix")
    bucket, prefix = uri[5:].split("/", 1)
    return bucket, prefix.strip("/")


def download_inputs(uri: str, destination: Path) -> None:
    from google.cloud import storage

    bucket, prefix = split_gs(uri)
    client = storage.Client()
    for name in ("X.parquet", "y.parquet", "snapshots.parquet", "feature_metadata.json"):
        client.bucket(bucket).blob(f"{prefix}/{name}").download_to_filename(destination / name)


def upload_outputs(uri: str, source: Path) -> None:
    from google.cloud import storage

    bucket, prefix = split_gs(uri)
    client = storage.Client()
    for path in source.rglob("*"):
        if path.is_file():
            client.bucket(bucket).blob(f"{prefix}/{path.relative_to(source)}").upload_from_filename(
                path
            )


def train(args) -> None:
    import mlflow

    from src.cloud_auth import configure_tracking_auth
    from src.train import hpo_trial, run

    configure_tracking_auth()
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    with tempfile.TemporaryDirectory(prefix="vertex-training-") as temporary:
        root = Path(temporary)
        inputs, outputs = root / "inputs", root / "model"
        inputs.mkdir()
        outputs.mkdir()
        download_inputs(args.data_uri, inputs)
        output_uri = args.output_uri.rstrip("/")
        if args.mode == "hpo-trial":
            import hypertune

            mlflow.set_experiment("boreal-marche-managed-hpo")
            params = {
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "min_samples_leaf": args.min_samples_leaf,
            }
            with mlflow.start_run(run_name="vertex-bayesian-trial") as tracking:
                score = hpo_trial(inputs, outputs, params)
                mlflow.log_params(params)
                mlflow.log_metric("validation_pr_auc", score)
                mlflow.log_artifact(str(outputs / "hpo_trial.json"))
                output_uri += "/" + tracking.info.run_id
            hypertune.HyperTune().report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag="validation_pr_auc",
                metric_value=score,
                global_step=1,
            )
        else:
            run(n_trials=args.trials, output_dir=outputs, data_dir=inputs, register=True)
        upload_outputs(output_uri, outputs)
        print(json.dumps({"output_uri": output_uri, "mode": args.mode}), flush=True)


def serve() -> None:
    import uvicorn
    from google.cloud import storage

    bucket, prefix = split_gs(os.environ["AIP_STORAGE_URI"])
    destination = Path(tempfile.mkdtemp(prefix="vertex-serving-"))
    client = storage.Client()
    for name in ("retention_model.joblib", "retention_model_metadata.json"):
        client.bucket(bucket).blob(f"{prefix}/{name}").download_to_filename(destination / name)
    os.environ["SM_MODEL_DIR"] = str(destination)
    uvicorn.run(
        "api.sagemaker_serve:app", host="0.0.0.0", port=int(os.getenv("AIP_HTTP_PORT", "8080"))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "serve"], default="train", nargs="?")
    parser.add_argument("--mode", choices=["full", "hpo-trial"], default="full")
    parser.add_argument("--data-uri")
    parser.add_argument("--output-uri")
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    args = parser.parse_args()
    if args.command == "serve":
        serve()
    else:
        if not args.data_uri or not args.output_uri:
            parser.error("training needs --data-uri and --output-uri")
        train(args)


if __name__ == "__main__":
    main()
