"""
Head A — Retention-prediction endpoints.

POST /predict       single customer
POST /predict/batch up to 500 customers
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import structlog
from fastapi import APIRouter, HTTPException, Request

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    FeatureEffect,
    PredictionRequest,
    PredictionResponse,
)

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["prediction"])


def _risk_label(probability: float) -> str:
    """Display bands only; 0.65 / 0.35 are judgement cut-offs, not tied to the profit threshold."""
    if probability >= 0.65:
        return "low"
    if probability >= 0.35:
        return "medium"
    return "high"


def _final_estimator(model: Any) -> Any | None:
    """Walk through wrappers (calibration, freezing, pipelines) to the fitted classifier."""
    estimator = model
    for _ in range(5):
        if hasattr(estimator, "feature_importances_"):
            return estimator
        if hasattr(estimator, "calibrated_classifiers_"):
            estimator = estimator.calibrated_classifiers_[0].estimator
        elif hasattr(estimator, "named_steps"):
            estimator = list(estimator.named_steps.values())[-1]
        elif hasattr(estimator, "estimator"):
            estimator = estimator.estimator
        else:
            return None
    return None


def _top_factors(
    features: dict[str, float], feature_order: list[str], model: Any
) -> list[dict[str, Any]]:
    """Global feature importances of the underlying classifier, not per-customer explanations."""
    try:
        estimator = _final_estimator(model)
        if estimator is None:
            return []
        names = feature_order or list(features.keys())
        pairs = sorted(zip(names, estimator.feature_importances_), key=lambda p: p[1], reverse=True)
        return [
            {"feature": name, "importance": round(float(imp), 4), "value": features.get(name)}
            for name, imp in pairs[:5]
        ]
    except Exception:
        logger.warning("top_factors_failed", exc_info=True)
        return []


TOP_EFFECTS = 5
MIN_EFFECT = 0.0005  # smaller moves (0.05 percentage points) are not worth telling a customer


def _score_with_effects(
    model: Any, row: pd.DataFrame, names: list[str], typical: dict[str, float] | None
) -> tuple[float, list[FeatureEffect]]:
    """Score one customer and explain the score from the same single prediction call.

    The call scores the customer once, then once more per feature with that feature set to
    its typical value. A feature's effect is the probability with the customer's value minus
    the probability with the typical one: how much this customer's own value of it moves the
    score. Features are correlated (orders, spend), so the effects do not add up to the gap
    between this customer and a typical one, and this is not a causal or coupon-uplift claim.
    """
    if not typical or any(name not in typical for name in names):
        return float(model.predict_proba(row)[0, 1]), []
    variants = pd.concat([row] * len(names), ignore_index=True)
    for position, name in enumerate(names):
        variants.loc[position, name] = typical[name]
    scores = model.predict_proba(pd.concat([row, variants], ignore_index=True))[:, 1]
    probability = float(scores[0])
    effects = [
        FeatureEffect(
            feature=name,
            value=float(row.iloc[0][name]),
            typical_value=typical[name],
            effect=round(probability - float(score), 4),
        )
        for name, score in zip(names, scores[1:])
    ]
    largest = sorted(effects, key=lambda e: abs(e.effect), reverse=True)
    return probability, [e for e in largest[:TOP_EFFECTS] if abs(e.effect) >= MIN_EFFECT]


def _predict_one(
    req: PredictionRequest,
    model: Any,
    threshold: float,
    feature_order: list[str],
    metadata: dict[str, Any],
    typical: dict[str, float] | None = None,
) -> PredictionResponse:
    """Run the sklearn pipeline on a single feature dict."""
    if model is None:
        raise HTTPException(status_code=503, detail="Retention model not loaded.")

    # Build a DataFrame row in the exact feature order the model expects.
    if feature_order:
        missing = [f for f in feature_order if f not in req.features]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Missing features: {missing}",
            )
        row = pd.DataFrame([{f: req.features[f] for f in feature_order}])
    else:
        row = pd.DataFrame([req.features])

    proba, explanation = _score_with_effects(model, row, list(row.columns), typical)

    return PredictionResponse(
        customer_id=req.customer_id,
        probability=round(proba, 6),
        will_return=proba >= threshold,
        threshold=threshold,
        risk_label=_risk_label(proba),
        top_factors=_top_factors(req.features, feature_order, model),
        explanation=explanation,
        model_version=metadata.get("model_version"),
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest, request: Request) -> PredictionResponse:
    """Predict whether a single customer will return within 90 days."""
    t0 = time.perf_counter()

    try:
        result = _predict_one(
            req,
            model=request.app.state.model,
            threshold=request.app.state.threshold,
            feature_order=request.app.state.feature_order,
            metadata=request.app.state.metadata,
            typical=getattr(request.app.state, "typical_values", None),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("prediction_failed", customer_id=req.customer_id, error=str(exc))
        raise HTTPException(status_code=500, detail="Internal prediction error.") from exc

    latency_ms = (time.perf_counter() - t0) * 1000
    request.state.prediction_records = [
        {
            "probability": result.probability,
            "will_return": result.will_return,
            "latency_ms": round(latency_ms, 2),
            "features": req.features,
        }
    ]
    return result


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(req: BatchPredictionRequest, request: Request) -> BatchPredictionResponse:
    """Predict retention for up to 500 customers in one call."""
    results: list[PredictionResponse] = []
    request.state.prediction_records = []
    for customer in req.customers:
        result = _predict_one(
            customer,
            model=request.app.state.model,
            threshold=request.app.state.threshold,
            feature_order=request.app.state.feature_order,
            metadata=request.app.state.metadata,
            typical=getattr(request.app.state, "typical_values", None),
        )
        results.append(result)
        request.state.prediction_records.append(
            {"features": customer.features, "probability": result.probability}
        )

    return BatchPredictionResponse(predictions=results, count=len(results))
