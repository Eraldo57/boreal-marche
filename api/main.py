"""
Boréal Marché API — the single entry point for all three model heads.

Run locally:
    uvicorn api.main:app --reload --port 8000

Endpoints:
    GET  /health         liveness probe
    GET  /ready          readiness probe (are models loaded?)
    GET  /version        model and application version
    POST /predict        single retention prediction
    POST /predict/batch  batch retention predictions
    POST /recommend      basket recommendations
    POST /chat           grounded bilingual assistant
    GET  /metrics        monitoring dashboard data, with each check's rule
    GET  /alerts         alerts raised by the background monitor
    POST /decisions      a person approves or rejects a proposed action
    GET  /decisions      the decision audit trail
    GET  /docs           auto-generated OpenAPI documentation
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Request, Response

from api.lifespan import lifespan
from api.routers import chat, decisions, monitor, predict, recommend
from api.schemas import HealthResponse, ReadyResponse, VersionResponse
from src.contracts import CONTRACT_VERSION, FEATURE_NAMES

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
        if os.getenv("APP_ENV") != "production"  # documented in .env.example; set in the image
        else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Boréal Marché API",
    description=(
        "Customer Intelligence Platform — retention prediction, "
        "product recommendations, and a grounded bilingual assistant."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Mount routers
app.include_router(predict.router)
app.include_router(recommend.router)
app.include_router(chat.router)
app.include_router(monitor.router)
app.include_router(decisions.router)


def _log_business_request(request: Request, status_code: int, start: float) -> None:
    if request.url.path in {"/predict", "/predict/batch", "/recommend", "/chat"}:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "path": request.url.path,
            "status_code": status_code,
            "latency_ms": (time.perf_counter() - start) * 1000,
            "predictions": getattr(request.state, "prediction_records", []),
        }
        recommendation = getattr(request.state, "recommendation", None)
        if recommendation is not None:
            record["recommendation"] = recommendation
        request.app.state.prediction_log.append(record)


@app.middleware("http")
async def record_traffic(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Measure all business routes, including validation failures, crashes and batch traffic."""
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        # An unhandled error reaches the caller as Starlette's 500: log that, then re-raise.
        _log_business_request(request, 500, start)
        raise
    _log_business_request(request, response.status_code, start)
    return response


# ---------------------------------------------------------------------------
# Core endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["infra"])
async def health() -> HealthResponse:
    """Liveness probe — always returns 200 if the process is running."""
    return HealthResponse()


@app.get("/ready", response_model=ReadyResponse, tags=["infra"])
async def ready(request: Request, response: Response) -> ReadyResponse:
    """Readiness probe — reports which models are loaded."""
    loaded: list[str] = []
    state = request.app.state
    if state.model is not None:
        loaded.append("retention_model")
    if state.rule_engine is not None:
        loaded.append("recommendation_rules")
    if state.rag_index is not None:
        loaded.append("rag_index")

    errors = getattr(state, "startup_errors", {})
    is_ready = len(loaded) == 3 and not errors
    if not is_ready:
        response.status_code = 503
    return ReadyResponse(ready=is_ready, models_loaded=loaded, errors=errors)


@app.get("/version", response_model=VersionResponse, tags=["infra"])
async def version(request: Request) -> VersionResponse:
    """Application and model version information."""
    meta = getattr(request.app.state, "metadata", {})

    # Try to read the git SHA
    git_sha = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_sha = result.stdout.strip()
    except Exception:
        pass

    return VersionResponse(
        version=app.version,
        git_sha=os.getenv("GIT_SHA") or git_sha,
        model_version=meta.get("model_version"),
        trained_at=meta.get("trained_at"),
        model_family=meta.get("model_family"),
        calibration_method=meta.get("calibration_method"),
        metrics=meta.get("metrics", {}),
        decision_threshold=meta.get("decision_threshold"),
        model_source=getattr(request.app.state, "model_source", None),
    )


@app.get("/feature-contract", tags=["prediction"])
async def feature_contract() -> dict[str, Any]:
    return {
        "feature_names": FEATURE_NAMES,
        "contract_version": CONTRACT_VERSION,
        "target": "will_return",
        "positive_means": "purchase within the next 90 days",
        "currency": "GBP",
        "risk_label_means": "risk of NOT returning",
    }


@app.get("/customers/{customer_id}/features", tags=["prediction"])
async def customer_features(customer_id: str, request: Request) -> dict[str, Any]:
    """Historical public-data demonstration only; no authentication is implemented."""
    store = getattr(request.app.state, "customer_features", None)
    if store is None:
        raise HTTPException(503, "Historical feature store missing; run python -m src.features")
    customer_id = customer_id.removeprefix("BM-")
    if customer_id not in store.index:
        raise HTTPException(404, "Customer absent from historical demonstration snapshot")
    return {
        "customer_id": customer_id,
        "features": store.loc[customer_id, FEATURE_NAMES].to_dict(),
        "snapshot": "2011-09-01",
        "demonstration_only": True,
    }
