"""
Boréal Marché — Model Training (Head A: Retention Prediction)
==============================================================

End-to-end training pipeline:

1. Load feature matrix and labels.
2. Train / test split with stratification.
3. Spot-check 8 model families + 2 baselines (DummyClassifier, recency heuristic).
4. Report PR-AUC (primary), ROC-AUC, Brier score, recall@top-decile, fit time.
5. Optuna HPO on the top 2 models, all trials logged to MLflow.
6. Probability calibration (isotonic + sigmoid) with reliability-curve data.
7. Decision-threshold selection by maximising expected profit.
8. Error analysis.
9. Save final model Pipeline + metadata sidecar via joblib.

SageMaker compatibility
-----------------------
The script honours the ``SM_CHANNEL_TRAIN`` and ``SM_MODEL_DIR`` environment
variables so that the same file runs locally **and** as a SageMaker training
job without modification.

Usage::

    python -m src.train
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data loading (SageMaker-aware)
# ---------------------------------------------------------------------------
def _data_dir() -> Path:
    """Return the directory containing X.parquet and y.parquet.

    If running inside SageMaker, ``SM_CHANNEL_TRAIN`` points to the data.
    Otherwise, we use ``data/processed/``.
    """
    sm = os.environ.get("SM_CHANNEL_TRAIN")
    if sm:
        return Path(sm)
    return PROJECT_ROOT / "data" / "processed"


def _model_dir() -> Path:
    """Return the directory where the model artifact should be saved.

    If running inside SageMaker, ``SM_MODEL_DIR`` is used.
    """
    sm = os.environ.get("SM_MODEL_DIR")
    if sm:
        return Path(sm)
    return PROJECT_ROOT / "artifacts"


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load features and labels from Parquet."""
    d = _data_dir()
    X = pd.read_parquet(d / "X.parquet")
    y = pd.read_parquet(d / "y.parquet").squeeze()
    print(f"[train] Loaded {len(X):,} customers, {X.shape[1]} features")
    print(f"[train] Positive rate: {y.mean():.2%}")
    return X, y


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def recall_at_top_decile(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Recall when targeting the top 10 % of predicted probabilities."""
    n = len(y_true)
    top_k = max(1, n // 10)
    top_idx = np.argsort(y_prob)[::-1][:top_k]
    return float(y_true[top_idx].sum() / y_true.sum()) if y_true.sum() > 0 else 0.0


def evaluate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fit_seconds: float = 0.0,
) -> dict[str, float]:
    """Compute the full metric suite."""
    return {
        "pr_auc": average_precision_score(y_true, y_prob),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "recall_top_decile": recall_at_top_decile(y_true, y_prob),
        "fit_time_s": fit_seconds,
    }


# ---------------------------------------------------------------------------
# Spot-check
# ---------------------------------------------------------------------------
class RecencyHeuristic:
    """Business baseline: predict return probability as 1/recency_days.

    Customers who bought more recently get a higher score.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RecencyHeuristic":
        self.classes_ = np.array([False, True])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        recency = X["recency_days"].values.astype(float)
        score = 1.0 / (1.0 + recency)  # simple monotone transform
        return np.column_stack([1 - score, score])


def _get_models(seed: int) -> dict[str, Any]:
    """Return the named dict of model instances for the spot-check."""
    return {
        # --- Baselines ---
        "DummyClassifier_stratified": DummyClassifier(strategy="stratified", random_state=seed),
        "business_heuristic_recency": RecencyHeuristic(),
        # --- Families ---
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]),
        "RidgeClassifier": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RidgeClassifier(random_state=seed)),
        ]),
        "DecisionTreeClassifier": DecisionTreeClassifier(max_depth=6, random_state=seed),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=seed, n_jobs=-1,
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, random_state=seed,
        ),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
            max_iter=200, max_depth=6, random_state=seed,
        ),
        "KNeighborsClassifier": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=15)),
        ]),
        "LinearSVC": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(max_iter=2000, random_state=seed)),
        ]),
    }


def _get_probas(model: Any, X: np.ndarray) -> np.ndarray:
    """Extract probabilities from a model, handling those without predict_proba."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # Normalise to [0, 1] via sigmoid
        return 1.0 / (1.0 + np.exp(-scores))
    else:
        # Fallback: predict and treat as probability
        return model.predict(X).astype(float)


def spot_check(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
) -> pd.DataFrame:
    """Run all spot-check models under cross-validation.

    Returns a DataFrame sorted by ``pr_auc`` descending.
    """
    seed = cfg["data"]["seed"]
    cv = StratifiedKFold(n_splits=cfg["modelling"]["cv_folds"], shuffle=True, random_state=seed)
    models = _get_models(seed)
    results = []

    mlflow.set_experiment("boreal-marche-spot-check")

    for name, model in models.items():
        print(f"[train]   {name} …", end="", flush=True)
        t0 = time.time()

        with mlflow.start_run(run_name=f"spot-check-{name}"):
            # CV probabilities
            if name == "business_heuristic_recency":
                # This heuristic doesn't need fitting — just score directly
                model.fit(X, y)
                y_prob = _get_probas(model, X)
            else:
                try:
                    y_prob = cross_val_predict(
                        model, X, y, cv=cv, method="predict_proba", n_jobs=-1,
                    )[:, 1]
                except AttributeError:
                    y_prob = cross_val_predict(
                        model, X, y, cv=cv, method="decision_function", n_jobs=-1,
                    )
                    y_prob = 1.0 / (1.0 + np.exp(-y_prob))

            elapsed = time.time() - t0
            metrics = evaluate(y.values, y_prob, elapsed)
            results.append({"model": name, **metrics})

            mlflow.log_params({"model_name": name})
            mlflow.log_metrics(metrics)

        print(f"  PR-AUC={metrics['pr_auc']:.4f}  ({elapsed:.1f}s)")

    df = pd.DataFrame(results).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    print("\n[train] Spot-check results:")
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Optuna HPO
# ---------------------------------------------------------------------------
def _build_hgb_trial(trial: optuna.Trial, seed: int) -> HistGradientBoostingClassifier:
    """Build a HistGradientBoosting with Optuna-sampled hyperparameters."""
    return HistGradientBoostingClassifier(
        max_iter=trial.suggest_int("max_iter", 100, 500),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 50),
        max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 127),
        l2_regularization=trial.suggest_float("l2_reg", 1e-6, 10.0, log=True),
        random_state=seed,
    )


def _build_gb_trial(trial: optuna.Trial, seed: int) -> GradientBoostingClassifier:
    """Build a GradientBoosting with Optuna-sampled hyperparameters."""
    return GradientBoostingClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 500),
        max_depth=trial.suggest_int("max_depth", 3, 8),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 50),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        random_state=seed,
    )


def _build_rf_trial(trial: optuna.Trial, seed: int) -> RandomForestClassifier:
    """Build a RandomForest with Optuna-sampled hyperparameters."""
    return RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 600),
        max_depth=trial.suggest_int("max_depth", 4, 15),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 30),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        random_state=seed,
        n_jobs=-1,
    )


def _build_lr_trial(trial: optuna.Trial, seed: int) -> Pipeline:
    """Build a LogisticRegression pipeline with Optuna-sampled hyperparameters."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=trial.suggest_float("C", 1e-4, 100.0, log=True),
            penalty=trial.suggest_categorical("penalty", ["l1", "l2"]),
            solver="saga",
            max_iter=2000,
            random_state=seed,
        )),
    ])


TUNABLE_BUILDERS: dict[str, Any] = {
    "HistGradientBoostingClassifier": _build_hgb_trial,
    "GradientBoostingClassifier": _build_gb_trial,
    "RandomForestClassifier": _build_rf_trial,
    "LogisticRegression": _build_lr_trial,
}


def tune_model(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
) -> tuple[Any, dict[str, float]]:
    """Run Optuna Bayesian HPO for one model family.

    Returns
    -------
    best_model
        Re-fitted on full training data with the best hyperparameters.
    best_metrics
        Evaluation metrics from the best trial.
    """
    seed = cfg["data"]["seed"]
    n_trials = cfg["modelling"]["tuning"]["n_trials"]
    cv = StratifiedKFold(n_splits=cfg["modelling"]["cv_folds"], shuffle=True, random_state=seed)

    builder = TUNABLE_BUILDERS.get(model_name)
    if builder is None:
        raise ValueError(f"No tunable builder registered for {model_name}")

    mlflow.set_experiment("boreal-marche-tuning")

    def objective(trial: optuna.Trial) -> float:
        model = builder(trial, seed)
        try:
            y_prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        except AttributeError:
            y_prob = cross_val_predict(model, X, y, cv=cv, method="decision_function", n_jobs=-1)
            y_prob = 1.0 / (1.0 + np.exp(-y_prob))
        score = average_precision_score(y.values, y_prob)

        with mlflow.start_run(run_name=f"tune-{model_name}-{trial.number}", nested=True):
            mlflow.log_params(trial.params)
            mlflow.log_metric("pr_auc", score)

        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner() if cfg["modelling"]["tuning"]["pruning"] else optuna.pruners.NopPruner(),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"[train] Best {model_name} trial: PR-AUC={study.best_value:.4f}")
    print(f"[train]   params: {study.best_params}")

    # Re-fit with best params on full training data
    best_model = builder(study.best_trial, seed)
    best_model.fit(X, y)
    y_prob = _get_probas(best_model, X)
    best_metrics = evaluate(y.values, y_prob)

    return best_model, best_metrics


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def calibrate(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    methods: list[str],
) -> tuple[Any, dict[str, Any]]:
    """Calibrate the model and return the best calibrator + curve data.

    Returns
    -------
    best_calibrated
        The calibrated model with the lowest Brier score.
    calibration_data
        Dict with reliability-curve data for each method.
    """
    results: dict[str, Any] = {}
    best_model = model
    best_brier = 1.0

    for method in methods:
        cal = CalibratedClassifierCV(model, method=method, cv=3)
        cal.fit(X_train, y_train)
        y_prob = cal.predict_proba(X_test)[:, 1]
        brier = brier_score_loss(y_test.values, y_prob)

        prob_true, prob_pred = calibration_curve(y_test.values, y_prob, n_bins=10)
        results[method] = {
            "brier_score": brier,
            "curve_true": prob_true.tolist(),
            "curve_pred": prob_pred.tolist(),
        }
        print(f"[train]   Calibration ({method}): Brier={brier:.4f}")

        if brier < best_brier:
            best_brier = brier
            best_model = cal

    # Also capture the uncalibrated curve
    y_prob_raw = _get_probas(model, X_test)
    prob_true_raw, prob_pred_raw = calibration_curve(y_test.values, y_prob_raw, n_bins=10)
    results["uncalibrated"] = {
        "brier_score": brier_score_loss(y_test.values, y_prob_raw),
        "curve_true": prob_true_raw.tolist(),
        "curve_pred": prob_pred_raw.tolist(),
    }

    return best_model, results


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------
def select_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_cfg: dict,
) -> tuple[float, dict[str, Any]]:
    """Select the decision threshold that maximises expected profit.

    Expected profit per customer at threshold *t*:
        TP: margin_retained − cost_coupon
        FP: −cost_coupon
        FN: −cost_false_negative
        TN: 0

    Parameters
    ----------
    y_true : ndarray
        True binary labels.
    y_prob : ndarray
        Predicted probabilities.
    cost_cfg : dict
        Keys: ``cost_coupon``, ``margin_retained``, ``cost_false_negative``.

    Returns
    -------
    best_threshold : float
    analysis : dict
    """
    cost_coupon = cost_cfg["cost_coupon"]
    margin = cost_cfg["margin_retained"]
    cost_fn = cost_cfg["cost_false_negative"]

    thresholds = np.linspace(0.05, 0.95, 181)
    profits = []

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        tp = ((preds == 1) & (y_true == 1)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()
        profit = tp * (margin - cost_coupon) + fp * (-cost_coupon) + fn * (-cost_fn)
        profits.append(float(profit))

    best_idx = int(np.argmax(profits))
    best_t = float(thresholds[best_idx])

    analysis = {
        "best_threshold": best_t,
        "max_expected_profit": profits[best_idx],
        "cost_assumptions": cost_cfg,
        "curve_thresholds": thresholds.tolist(),
        "curve_profits": profits,
    }
    print(f"[train] Optimal threshold: {best_t:.3f}  (expected profit: ${profits[best_idx]:,.0f})")
    return best_t, analysis


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------
def error_analysis(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
    feature_names: list[str],
) -> dict[str, Any]:
    """Analyse where the model makes mistakes.

    Returns a dict with segment-level error rates and top confusion features.
    """
    y_prob = _get_probas(model, X_test)
    y_pred = (y_prob >= threshold).astype(int)
    y_true = y_test.values.astype(int)

    # Per-segment breakdown
    segments: dict[str, Any] = {}
    for col in ["is_one_time_buyer", "is_domestic"]:
        if col in X_test.columns:
            for val in X_test[col].unique():
                mask = X_test[col].values == val
                if mask.sum() < 5:
                    continue
                seg_true = y_true[mask]
                seg_pred = y_pred[mask]
                seg_prob = y_prob[mask]
                n = int(mask.sum())
                tp = int(((seg_pred == 1) & (seg_true == 1)).sum())
                fp = int(((seg_pred == 1) & (seg_true == 0)).sum())
                fn = int(((seg_pred == 0) & (seg_true == 1)).sum())
                tn = int(((seg_pred == 0) & (seg_true == 0)).sum())
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                segments[f"{col}={int(val)}"] = {
                    "n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "recall": round(recall, 4),
                    "precision": round(precision, 4),
                    "pr_auc": round(float(average_precision_score(seg_true, seg_prob)), 4),
                }

    # Feature importance (if the model exposes it)
    importances: list[dict[str, Any]] = []
    estimator = model
    if hasattr(estimator, "named_steps"):
        estimator = estimator.steps[-1][1]
    if hasattr(estimator, "feature_importances_"):
        imp = estimator.feature_importances_
        for name, val in sorted(zip(feature_names, imp), key=lambda x: -x[1]):
            importances.append({"feature": name, "importance": round(float(val), 4)})

    return {"segment_errors": segments, "feature_importances": importances}


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def _git_sha() -> str | None:
    """Return the current git commit SHA, or None if not in a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def save_model(
    model: Any,
    threshold: float,
    feature_names: list[str],
    metrics: dict[str, float],
    calibration_data: dict[str, Any],
    threshold_analysis: dict[str, Any],
    error_data: dict[str, Any],
) -> Path:
    """Save the model artifact and metadata sidecar.

    Returns
    -------
    Path
        Path to the saved model file.
    """
    out_dir = _model_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "retention_model.joblib"
    meta_path = out_dir / "retention_model_metadata.json"

    joblib.dump(model, model_path)

    metadata = {
        "model_version": "0.1.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "decision_threshold": threshold,
        "metrics": metrics,
        "calibration": calibration_data,
        "threshold_analysis": {
            "best_threshold": threshold_analysis["best_threshold"],
            "max_expected_profit": threshold_analysis["max_expected_profit"],
            "cost_assumptions": threshold_analysis["cost_assumptions"],
        },
        "error_analysis": error_data,
        "pipeline_steps": (
            [step[0] for step in model.steps] if hasattr(model, "steps")
            else [type(model).__name__]
        ),
        "libraries": {
            "scikit-learn": __import__("sklearn").__version__,
            "optuna": optuna.__version__,
            "mlflow": mlflow.__version__,
        },
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"[train] Model saved to {model_path}")
    print(f"[train] Metadata saved to {meta_path}")
    return model_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run() -> None:
    """Execute the full training pipeline."""
    cfg = load_config()
    seed = cfg["data"]["seed"]

    print("=" * 70)
    print("Boréal Marché — Model Training (Head A)")
    print("=" * 70)

    # 1. Load
    X, y = load_data()

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["modelling"]["test_size"],
        random_state=seed, stratify=y,
    )
    print(f"[train] Train: {len(X_train):,}  Test: {len(X_test):,}")

    # 3. Spot-check
    print("\n[train] === Spot-check ===")
    spot_df = spot_check(X_train, y_train, cfg)

    # 4. Select top 2 for tuning (exclude baselines)
    non_baseline = spot_df[~spot_df["model"].isin([
        "DummyClassifier_stratified", "business_heuristic_recency",
    ])]
    top2 = non_baseline.head(2)["model"].tolist()
    print(f"\n[train] Top 2 for tuning: {top2}")

    # 5. Optuna HPO
    best_model = None
    best_score = -1.0
    best_metrics: dict[str, float] = {}

    for name in top2:
        if name not in TUNABLE_BUILDERS:
            print(f"[train] Skipping {name} (no tunable builder)")
            continue
        print(f"\n[train] === Tuning {name} ===")
        model, metrics = tune_model(name, X_train, y_train, cfg)
        if metrics["pr_auc"] > best_score:
            best_score = metrics["pr_auc"]
            best_model = model
            best_metrics = metrics
            best_name = name

    if best_model is None:
        # Fallback: use the top spot-check model
        best_name = non_baseline.iloc[0]["model"]
        print(f"[train] No tunable models available, using {best_name}")
        models = _get_models(seed)
        best_model = models[best_name]
        best_model.fit(X_train, y_train)
        y_prob = _get_probas(best_model, X_train)
        best_metrics = evaluate(y_train.values, y_prob)

    print(f"\n[train] Best tuned model: {best_name}  PR-AUC={best_score:.4f}")

    # 6. Calibration
    print("\n[train] === Calibration ===")
    calibrated_model, cal_data = calibrate(
        best_model, X_train, y_train, X_test, y_test,
        methods=cfg["modelling"]["calibration"]["methods"],
    )

    # 7. Threshold selection
    print("\n[train] === Threshold Selection ===")
    y_prob_test = _get_probas(calibrated_model, X_test)
    threshold, thresh_analysis = select_threshold(
        y_test.values, y_prob_test,
        cfg["modelling"]["threshold"]["cost_assumptions"],
    )

    # 8. Final held-out evaluation
    print("\n[train] === Final Held-out Evaluation ===")
    final_metrics = evaluate(y_test.values, y_prob_test)
    final_metrics["threshold"] = threshold
    print(f"[train] Test PR-AUC:     {final_metrics['pr_auc']:.4f}")
    print(f"[train] Test ROC-AUC:    {final_metrics['roc_auc']:.4f}")
    print(f"[train] Test Brier:      {final_metrics['brier_score']:.4f}")
    print(f"[train] Recall@top10%:   {final_metrics['recall_top_decile']:.4f}")

    # Log final to MLflow
    mlflow.set_experiment("boreal-marche-final")
    with mlflow.start_run(run_name=f"final-{best_name}"):
        mlflow.log_metrics(final_metrics)
        mlflow.log_param("model_name", best_name)
        mlflow.log_param("threshold", threshold)
        mlflow.sklearn.log_model(calibrated_model, "model")

    # 9. Error analysis
    print("\n[train] === Error Analysis ===")
    from src.features import FEATURE_NAMES
    err_data = error_analysis(calibrated_model, X_test, y_test, threshold, FEATURE_NAMES)
    for seg, info in err_data["segment_errors"].items():
        print(f"[train]   {seg}: n={info['n']}  recall={info['recall']:.2f}  precision={info['precision']:.2f}")

    # 10. Save
    print("\n[train] === Saving ===")
    save_model(
        calibrated_model, threshold, FEATURE_NAMES,
        final_metrics, cal_data, thresh_analysis, err_data,
    )

    print("=" * 70)
    print("[train] Done.")


def main() -> None:
    """CLI wrapper."""
    parser = argparse.ArgumentParser(description="Boréal Marché model training")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
