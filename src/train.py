"""Retention training with mature chronological snapshots and an untouched holdout.

Preprocessing is fitted inside each estimator. Positive class = will purchase
in the next 90 days. Run python -m src.train after ingestion and features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from scipy.special import expit
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.contracts import (
    ARTIFACTS,
    CONTRACT_VERSION,
    ENGINEERED_FEATURE_NAMES,
    FEATURE_NAMES,
    PROCESSED,
    PROJECT_ROOT,
)
from src.ingest import _sha256, load_config
from src.plots import FIGURES
from src.ranking_plots import save_ranking_figures
from src.registry import register_run

REPORTS = PROJECT_ROOT / "reports" / "current_run"
SPLITS = ["train", "validation", "calibration", "decision", "test"]


def temporal_partitions(snapshots: pd.DataFrame) -> dict[str, np.ndarray]:
    """Check label maturity and partition whole chronological snapshots."""
    result, previous_end = {}, None
    for split in SPLITS:
        rows = snapshots[snapshots["split"] == split]
        if rows.empty:
            raise ValueError(f"Missing temporal partition: {split}")
        cutoff = pd.to_datetime(rows["cutoff"]).min()
        end = pd.to_datetime(rows["label_end"]).max()
        if previous_end is not None and previous_end > cutoff:
            raise ValueError(f"Labels from preceding split are not mature before {split}")
        if end <= cutoff:
            raise ValueError("Label end must follow cutoff")
        result[split], previous_end = rows.index.to_numpy(), end
    return result


def recall_at_top_decile(y_true: Any, y_prob: Any) -> float:
    """Fraction of actual returners in the highest-scored 10%."""
    selected = np.argsort(-y_prob, kind="stable")[: max(1, len(y_true) // 10)]
    return float(y_true[selected].sum() / y_true.sum()) if y_true.sum() else 0.0


def evaluate(y_true: Any, y_prob: Any, fit_seconds: float | None = None) -> dict[str, Any]:
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "recall_top_decile": recall_at_top_decile(y_true, y_prob),
        "fit_time_s": fit_seconds,
    }


class RecencyHeuristic:
    """Unlearned business ranking baseline: 1 / (1 + recency days)."""

    def fit(self, X: Any, y: Any) -> RecencyHeuristic:
        return self

    def predict_proba(self, X: Any) -> Any:
        score = 1 / (1 + X["recency_days"].to_numpy())
        return np.column_stack([1 - score, score])


def pipeline(estimator: Any, scale: bool = False) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps + [("classifier", estimator)])


def _get_models(seed: int) -> dict:
    return {
        "DummyClassifier_stratified": pipeline(
            DummyClassifier(strategy="stratified", random_state=seed)
        ),
        "business_heuristic_recency": RecencyHeuristic(),
        "LogisticRegression": pipeline(LogisticRegression(max_iter=2000, random_state=seed), True),
        "RidgeClassifier": pipeline(RidgeClassifier(), True),
        "DecisionTreeClassifier": pipeline(DecisionTreeClassifier(max_depth=6, random_state=seed)),
        "RandomForestClassifier": pipeline(
            RandomForestClassifier(n_estimators=150, max_depth=8, random_state=seed, n_jobs=2)
        ),
        "GradientBoostingClassifier": pipeline(
            GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=seed)
        ),
        # No internal random validation split.
        "HistGradientBoostingClassifier": pipeline(
            HistGradientBoostingClassifier(
                max_iter=150, max_depth=6, early_stopping=False, random_state=seed
            )
        ),
        "KNeighborsClassifier": pipeline(KNeighborsClassifier(n_neighbors=15), True),
        "LinearSVC": pipeline(LinearSVC(max_iter=5000, random_state=seed), True),
    }


def _get_probas(model: Any, X: Any) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1])
    # Ranking transform for spot-check only; not calibrated probabilities.
    return np.asarray(expit(model.decision_function(X)))


def trial_model(name: str, trial: Any, seed: int) -> Pipeline:
    if name == "LogisticRegression":
        return pipeline(
            LogisticRegression(
                C=trial.suggest_float("C", 0.001, 20, log=True), max_iter=2000, random_state=seed
            ),
            True,
        )
    if name == "RandomForestClassifier":
        return pipeline(
            RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 250),
                max_depth=trial.suggest_int("max_depth", 4, 12),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 25),
                random_state=seed,
                n_jobs=2,
            )
        )
    if name == "GradientBoostingClassifier":
        return pipeline(
            GradientBoostingClassifier(
                n_estimators=trial.suggest_int("n_estimators", 80, 200),
                max_depth=trial.suggest_int("max_depth", 2, 5),
                learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                random_state=seed,
            )
        )
    return pipeline(
        HistGradientBoostingClassifier(
            max_iter=trial.suggest_int("max_iter", 80, 250),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 8, 40),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            l2_regularization=trial.suggest_float("l2_regularization", 0.001, 10, log=True),
            early_stopping=False,
            random_state=seed,
        )
    )


def utility(y_true: Any, y_prob: Any, threshold: float, costs: dict[str, Any]) -> float:
    """Assumed targeting utility; NOT causal coupon uplift or observed profit."""
    chosen = y_prob >= threshold
    return float(
        np.sum(chosen * (y_true * costs["margin_retained"] - costs["cost_coupon"]))
        - np.sum(~chosen * y_true * costs["cost_false_negative"])
    )


def select_threshold(
    y_true: Any, y_prob: Any, cost_cfg: dict[str, Any]
) -> tuple[float, dict[str, Any]]:
    thresholds = np.linspace(0, 1, 201)
    values = [utility(y_true, y_prob, t, cost_cfg) for t in thresholds]
    best = int(np.argmax(values))
    return float(thresholds[best]), {
        "best_threshold": float(thresholds[best]),
        "validation_utility": values[best],
        "cost_assumptions": cost_cfg,
        "interpretation": "assumed utility of targeting likely returners, not coupon uplift",
        "curve_thresholds": thresholds.tolist(),
        "curve_utility": values,
    }


def error_analysis(y: Any, p: Any, X: Any, threshold: float) -> dict[str, Any]:
    result = {}
    for col in ["is_one_time_buyer", "is_domestic"]:
        if col not in X.columns:
            continue  # A synthetic/serving feature matrix may omit offline audit-only columns.
        for val in sorted(X[col].unique()):
            mask = (X[col] == val).to_numpy()
            truth, pred = y[mask].astype(bool), p[mask] >= threshold
            tp, fp = int((truth & pred).sum()), int((~truth & pred).sum())
            fn, tn = int((truth & ~pred).sum()), int((~truth & ~pred).sum())
            result[f"{col}={val:g}"] = {
                "n": int(mask.sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "returner_recall": tp / (tp + fn) if tp + fn else None,
                "precision": tp / (tp + fp) if tp + fp else None,
            }
    return result


def _load(data_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    X = pd.read_parquet(data_dir / "X.parquet")
    y = pd.read_parquet(data_dir / "y.parquet")["will_return"]
    snapshots = pd.read_parquet(data_dir / "snapshots.parquet")
    if (
        not set(FEATURE_NAMES).issubset(X.columns)
        or not set(X.columns).issubset(ENGINEERED_FEATURE_NAMES)
        or not X.index.equals(y.index)
        or not X.index.equals(snapshots.index)
    ):
        raise ValueError("Feature/target/snapshot contract mismatch; rerun features")
    return X[FEATURE_NAMES], y, snapshots


def _git_state() -> tuple[str, bool | None]:
    """Commit and dirty flag; outside a git checkout (e.g. SageMaker) GIT_SHA, else unknown."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        )
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        return os.getenv("GIT_SHA", "unknown"), None
    return commit.decode().strip(), bool(status.strip())


def hpo_trial(data_dir: Path, model_dir: Path, params: dict[str, int]) -> float:
    """One managed tuning trial (SageMaker): the selected family, a random forest, with the
    given hyperparameters, fitted on the training snapshots and scored on the model-selection
    validation snapshot, the same split the local Optuna study used. The printed line is what
    the tuning job's metric regex reads."""
    seed = load_config()["data"]["seed"]
    X, y, snapshots = _load(data_dir)
    parts = temporal_partitions(snapshots)
    model = pipeline(RandomForestClassifier(**params, random_state=seed, n_jobs=-1))
    start = time.perf_counter()
    model.fit(X.loc[parts["train"]], y.loc[parts["train"]])
    seconds = time.perf_counter() - start
    validation = parts["validation"]
    score = float(average_precision_score(y.loc[validation], _get_probas(model, X.loc[validation])))
    model_dir.mkdir(parents=True, exist_ok=True)
    result = {"params": params, "validation_pr_auc": score, "fit_seconds": seconds}
    (model_dir / "hpo_trial.json").write_text(json.dumps(result, indent=2))
    print(f"validation_pr_auc={score:.6f}", flush=True)
    return score


def run(
    *,
    n_trials: int | None = None,
    output_dir: Path = ARTIFACTS,
    data_dir: Path | None = None,
    register: bool = True,
) -> dict:
    """Lock selection/calibration/threshold decisions before scoring final test."""
    cfg = load_config()
    seed = cfg["data"]["seed"]
    n_trials = n_trials if n_trials is not None else cfg["modelling"]["tuning"]["n_trials"]
    data_dir = data_dir or PROCESSED
    X, y, snapshots = _load(data_dir)
    parts = temporal_partitions(snapshots)
    tr, va, ca, de, te = [parts[s] for s in SPLITS]
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    )
    mlflow.set_experiment("boreal-marche-temporal")
    reports_dir = REPORTS if output_dir.resolve() == ARTIFACTS.resolve() else output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_log, rows, fitted = [], [], {}
    models = _get_models(seed)
    with mlflow.start_run(run_name="chronological-retention") as run_info:
        for name, model in models.items():
            start = time.perf_counter()
            model.fit(X.loc[tr], y.loc[tr])
            elapsed = time.perf_counter() - start
            score = evaluate(y.loc[va].to_numpy(), _get_probas(model, X.loc[va]), elapsed)
            rows.append({"model": name, **score})
            fitted[name] = model
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_metrics({k: v for k, v in score.items() if v is not None})
            print(f"[train] validation {name}: AP={score['pr_auc']:.4f}", flush=True)
        comparison = sorted(rows, key=lambda r: -r["pr_auc"])
        families = [
            r
            for r in comparison
            if r["model"] not in ("DummyClassifier_stratified", "business_heuristic_recency")
        ]
        best_name, best_score = families[0]["model"], families[0]["pr_auc"]
        selected = fitted[best_name]
        tunable = {
            "LogisticRegression",
            "RandomForestClassifier",
            "GradientBoostingClassifier",
            "HistGradientBoostingClassifier",
        }
        for row in [r for r in families if r["model"] in tunable][:2]:
            name = row["model"]

            def objective(trial: Any) -> float:
                # Actual learning-curve resources, not fabricated intermediate scores.
                # All subsets are earlier training rows; the fixed validation period is unchanged.
                steps = []
                with mlflow.start_run(run_name=f"tune-{name}-{trial.number}", nested=True):
                    for step, fraction in enumerate((0.5, 0.75, 1.0)):
                        subset = tr[: max(2, int(len(tr) * fraction))]
                        model = trial_model(name, trial, seed)
                        model.fit(X.loc[subset], y.loc[subset])
                        score = float(
                            average_precision_score(y.loc[va], _get_probas(model, X.loc[va]))
                        )
                        trial.report(score, step)
                        steps.append({"training_rows": len(subset), "validation_pr_auc": score})
                        mlflow.log_metric("validation_pr_auc", score, step=step)
                        if step < 2 and trial.should_prune():
                            trials_log.append(
                                {
                                    "family": name,
                                    "trial": trial.number,
                                    "state": "pruned",
                                    "validation_pr_auc": score,
                                    "params": trial.params,
                                    "learning_curve": steps,
                                }
                            )
                            mlflow.log_params(trial.params)
                            mlflow.set_tag("trial_state", "pruned")
                            raise optuna.TrialPruned()
                    trials_log.append(
                        {
                            "family": name,
                            "trial": trial.number,
                            "state": "complete",
                            "validation_pr_auc": score,
                            "params": trial.params,
                            "learning_curve": steps,
                        }
                    )
                    mlflow.log_params(trial.params)
                    mlflow.set_tag("trial_state", "complete")
                return score

            if n_trials:
                study = optuna.create_study(
                    direction="maximize",
                    sampler=optuna.samplers.TPESampler(seed=seed),
                    pruner=optuna.pruners.MedianPruner(n_startup_trials=4),
                )
                study.optimize(objective, n_trials=n_trials)
                if study.best_value > best_score:
                    best_name, best_score = name, study.best_value
                    selected = trial_model(name, study.best_trial, seed)
        fit_rows = np.concatenate([tr, va])
        refit_start = time.perf_counter()
        selected = clone(selected).fit(X.loc[fit_rows], y.loc[fit_rows])
        refit_seconds = time.perf_counter() - refit_start
        calibrators, calibration_data = {"uncalibrated": selected}, {}
        for method in cfg["modelling"]["calibration"]["methods"]:
            calibrators[method] = CalibratedClassifierCV(
                FrozenEstimator(selected), method=method
            ).fit(X.loc[ca], y.loc[ca])
        for method, model in calibrators.items():
            p = _get_probas(model, X.loc[de])
            observed, predicted = calibration_curve(y.loc[de], p, n_bins=10)
            calibration_data[method] = {
                "brier_score": float(brier_score_loss(y.loc[de], p)),
                "curve_true": observed.tolist(),
                "curve_pred": predicted.tolist(),
            }
        eligible = [k for k, m in calibrators.items() if hasattr(m, "predict_proba")]
        cal_method = min(eligible, key=lambda k: calibration_data[k]["brier_score"])
        final = calibrators[cal_method]
        threshold, analysis = select_threshold(
            y.loc[de].to_numpy(),
            _get_probas(final, X.loc[de]),
            cfg["modelling"]["threshold"]["cost_assumptions"],
        )
        # First and only final-test evaluation: all decisions above are locked.
        p_test = _get_probas(final, X.loc[te])
        final_metrics = evaluate(y.loc[te].to_numpy(), p_test, refit_seconds)
        final_metrics["assumed_test_utility"] = utility(
            y.loc[te].to_numpy(), p_test, threshold, analysis["cost_assumptions"]
        )
        baseline_test = {}
        for name in ["DummyClassifier_stratified", "business_heuristic_recency"]:
            baseline = models[name].fit(X.loc[fit_rows], y.loc[fit_rows])
            baseline_test[name] = evaluate(y.loc[te].to_numpy(), _get_probas(baseline, X.loc[te]))
        split_report = {
            s: {
                "rows": len(idx),
                "cutoffs": sorted(snapshots.loc[idx, "cutoff"].unique()),
                "label_end": snapshots.loc[idx, "label_end"].max(),
                "positive_rate": float(y.loc[idx].mean()),
            }
            for s, idx in parts.items()
        }
        report = {
            "protocol": "chronological snapshots with mature labels; repeat customers allowed",
            "splits": split_report,
            "model_comparison_validation": comparison,
            "optuna_trials": trials_log,
            "pruning": {
                "method": "MedianPruner",
                "resource": "earlier training rows",
                "fractions": [0.5, 0.75, 1.0],
                "startup_trials": 4,
            },
            "selected_model": best_name,
            "selected_calibration": cal_method,
            "calibration_decision_period": calibration_data,
            "threshold_selection": analysis,
            "final_test": final_metrics,
            "final_test_baselines": baseline_test,
            "error_analysis": error_analysis(
                y.loc[te].to_numpy(),
                p_test,
                pd.read_parquet(data_dir / "X.parquet").loc[te],
                threshold,
            ),
        }
        (reports_dir / "retention_metrics.json").write_text(json.dumps(report, indent=2))
        feature_meta = json.loads((data_dir / "feature_metadata.json").read_text())
        model_path = output_dir / "retention_model.joblib"
        joblib.dump(final, model_path)
        git_commit, git_dirty = _git_state()
        metadata = {
            "contract_version": CONTRACT_VERSION,
            "model_version": "0.3.0",
            "trained_at": datetime.now(UTC).isoformat(),
            "model_family": best_name,
            "calibration_method": cal_method,
            "target": "will_return",
            "label_window_days": 90,
            "feature_names": FEATURE_NAMES,
            "feature_count": len(FEATURE_NAMES),
            "engineered_feature_count": len(ENGINEERED_FEATURE_NAMES),
            "minimisation_basis": "June 2011 decision-period audit; original model archived",
            "training_row_count": len(fit_rows),
            "calibration_row_count": len(ca),
            "decision_threshold": threshold,
            "evaluation_metric": "average_precision",
            "source_data_sha256": feature_meta["source_data_sha256"],
            "feature_sha256": _sha256(data_dir / "X.parquet"),
            "model_sha256": _sha256(model_path),
            "config_sha256": _sha256(PROJECT_ROOT / "configs/pipeline.yaml"),
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "source_code_sha256": hashlib.sha256(
                b"".join(p.read_bytes() for p in sorted((PROJECT_ROOT / "src").rglob("*.py")))
            ).hexdigest(),
            "library_versions": {
                p: version(p) for p in ["scikit-learn", "numpy", "pandas", "optuna", "mlflow"]
            },
            "country_map": feature_meta["country_map"],
            "metrics": final_metrics,
            "mlflow_run_id": run_info.info.run_id,
            "cost_assumptions": analysis["cost_assumptions"],
            "evaluation_protocol": report["protocol"],
        }
        (output_dir / "retention_model_metadata.json").write_text(json.dumps(metadata, indent=2))
        X.loc[fit_rows].to_parquet(output_dir / "drift_reference.parquet", index=False)
        pd.DataFrame(
            {
                "customer_id": snapshots.loc[te, "customer_id"].to_numpy(),
                "actual_will_return": y.loc[te].to_numpy(),
                "probability": p_test,
            }
        ).to_parquet(output_dir / "test_predictions.parquet", index=False)
        mlflow.log_params(
            {
                "selected_model": best_name,
                "calibration": cal_method,
                "decision_threshold": threshold,
                "optuna_trials_per_family": n_trials,
            }
        )
        mlflow.log_metrics({k: v for k, v in final_metrics.items() if v is not None})
        mlflow.set_tags(
            {
                "git_commit": metadata["git_commit"],
                "git_dirty": str(metadata["git_dirty"]),
                "source_data_sha256": metadata["source_data_sha256"],
                "feature_sha256": metadata["feature_sha256"],
                "model_sha256": metadata["model_sha256"],
            }
        )
        for name, draw in FIGURES.items():
            mlflow.log_figure(draw(report), f"plots/{name}")
        for path in save_ranking_figures(
            reports_dir / "figures",
            y_true=y.loc[te].to_numpy(),
            probabilities=p_test,
            recency_scores=_get_probas(models["business_heuristic_recency"], X.loc[te]),
            threshold=threshold,
            model_name=best_name,
            snapshot=", ".join(split_report["test"]["cutoffs"]),
        ):
            mlflow.log_artifact(str(path), "plots")
        mlflow.log_artifact(str(reports_dir / "retention_metrics.json"))
        mlflow.log_artifact(str(output_dir / "retention_model_metadata.json"))
        mlflow.sklearn.log_model(final, "model", input_example=X.loc[te].head(1))
        print(
            json.dumps(
                {"selected_model": best_name, "calibration": cal_method, "test": final_metrics},
                indent=2,
            ),
            flush=True,
        )
    if register:
        # A new registry version is only a candidate; promotion is a separate, human step.
        registered = register_run(
            run_info.info.run_id, model_path, output_dir / "retention_model_metadata.json"
        )
        print(f"[train] registered version {registered} as a candidate (not promoted)", flush=True)
    return metadata


def main() -> None:
    """Local runs and SageMaker jobs share this entry point (SageMaker script mode).

    SageMaker passes hyperparameters as arguments and the data and model folders as
    SM_CHANNEL_TRAIN and SM_MODEL_DIR; locally the defaults are data/processed and
    artifacts/. ``--mode hpo-trial`` is one tuning trial of a managed tuning job.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["full", "hpo-trial"], default="full")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument(
        "--train-dir", type=Path, default=Path(os.getenv("SM_CHANNEL_TRAIN", str(PROCESSED)))
    )
    parser.add_argument(
        "--model-dir",
        "--output-dir",
        dest="model_dir",
        type=Path,
        default=Path(os.getenv("SM_MODEL_DIR", str(ARTIFACTS))),
    )
    parser.add_argument("--no-register", action="store_true", help="skip the MLflow registry")
    # Random-forest hyperparameters for --mode hpo-trial (same ranges as the Optuna study).
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    args = parser.parse_args()
    if args.mode == "hpo-trial":
        params = {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
        }
        hpo_trial(args.train_dir, args.model_dir, params)
        return
    run(
        n_trials=args.trials,
        output_dir=args.model_dir,
        data_dir=args.train_dir,
        register=not args.no_register,
    )


if __name__ == "__main__":
    main()
