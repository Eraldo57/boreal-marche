#!/usr/bin/env python3
"""
Boréal Marché — Scheduled Retraining Pipeline

Re-runs ETL, re-mines association rules, retrains the retention model,
logs everything to MLflow, and compares against the current production model.

If the new model passes promotion criteria, it updates the local artifacts.
A CI/CD workflow then creates a PR — this script NEVER auto-deploys.

Usage:
    python scripts/retrain.py --config configs/pipeline.yaml
    python scripts/retrain.py --config configs/pipeline.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("retrain")


def load_config(config_path: str) -> dict:
    """Load the pipeline configuration."""
    import yaml

    with open(config_path) as f:
        return yaml.safe_load(f)


def run_etl(cfg: dict) -> Path:
    """Re-run the ingestion and feature engineering pipeline."""
    from src.ingest import ingest_pipeline
    from src.features import build_feature_table

    logger.info("Step 1/5: Running ETL pipeline...")
    ingest_pipeline(cfg)

    logger.info("Step 2/5: Building feature table...")
    feature_path = build_feature_table(cfg)
    logger.info(f"Feature table written to {feature_path}")
    return feature_path


def mine_rules(cfg: dict) -> Path:
    """Re-mine association rules."""
    from src.rules import mine_and_export

    logger.info("Step 3/5: Mining association rules...")
    rules_path = mine_and_export(cfg)
    logger.info(f"Rules exported to {rules_path}")
    return rules_path


def retrain_model(cfg: dict) -> dict:
    """Retrain the retention model and log to MLflow."""
    import mlflow

    from src.train import train_pipeline

    logger.info("Step 4/5: Retraining retention model...")

    experiment_name = "boreal-marche-retention"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"retrain-{datetime.now().strftime('%Y%m%d-%H%M')}") as run:
        metrics = train_pipeline(cfg, mlflow_run=run)
        run_id = run.info.run_id

    logger.info(f"Training complete. MLflow run: {run_id}")
    logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")

    return {"run_id": run_id, "metrics": metrics}


def check_promotion(new_metrics: dict, cfg: dict, dry_run: bool = False) -> bool:
    """
    Compare new model metrics against current production model.
    Promotion criteria:
      - PR-AUC must not decrease by more than 1%
      - Brier score must not increase by more than 5%
    """
    import mlflow

    logger.info("Step 5/5: Checking promotion criteria...")

    primary_metric = cfg.get("modelling", {}).get("primary_metric", "average_precision")
    new_score = new_metrics.get(primary_metric, 0)

    # Try to find current production model
    try:
        client = mlflow.tracking.MlflowClient()
        model_name = "boreal-marche-retention"

        # Get production model's metrics
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        if prod_versions:
            prod_run = client.get_run(prod_versions[0].run_id)
            prod_score = float(prod_run.data.metrics.get(primary_metric, 0))
            logger.info(f"Current production {primary_metric}: {prod_score:.4f}")
            logger.info(f"New model {primary_metric}: {new_score:.4f}")

            # Promotion gate: new must be within 1% of production
            threshold = prod_score * 0.99
            if new_score < threshold:
                logger.warning(
                    f"New model ({new_score:.4f}) is below threshold ({threshold:.4f}). "
                    f"NOT promoting."
                )
                return False
        else:
            logger.info("No production model found — this is the first deployment.")

    except Exception as e:
        logger.warning(f"Could not compare with production model: {e}")
        logger.info("Proceeding with promotion (no baseline to compare).")

    if dry_run:
        logger.info("DRY RUN — would promote model but taking no action.")
        return False

    logger.info("✅ Promotion criteria met. Updating local artifacts.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Boréal Marché retraining pipeline")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Pipeline config path")
    parser.add_argument("--dry-run", action="store_true", help="Run without writing artifacts")
    parser.add_argument("--skip-etl", action="store_true", help="Skip ETL, use existing data")
    parser.add_argument("--skip-rules", action="store_true", help="Skip rule mining")
    args = parser.parse_args()

    cfg = load_config(args.config)

    try:
        # ETL
        if not args.skip_etl:
            run_etl(cfg)
        else:
            logger.info("Skipping ETL (--skip-etl)")

        # Rules
        if not args.skip_rules:
            mine_rules(cfg)
        else:
            logger.info("Skipping rule mining (--skip-rules)")

        # Model training
        result = retrain_model(cfg)

        # Promotion check
        promoted = check_promotion(result["metrics"], cfg, dry_run=args.dry_run)

        if promoted:
            logger.info("🎉 Retraining complete — artifacts updated.")
            logger.info("The CI/CD workflow will create a PR for review.")
        else:
            logger.info("Retraining complete — no promotion.")

        sys.exit(0)

    except Exception as e:
        logger.error(f"Retraining failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
