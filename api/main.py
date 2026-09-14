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
    GET  /metrics        monitoring dashboard data
    GET  /docs           auto-generated OpenAPI documentation
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

import structlog
from fastapi import FastAPI

from api.lifespan import lifespan
from api.routers import chat, monitor, predict, recommend
from api.schemas import HealthResponse, ReadyResponse, VersionResponse

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer() if os.getenv("ENV") != "production"
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


# ---------------------------------------------------------------------------
# Core endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["infra"])
async def health() -> HealthResponse:
    """Liveness probe — always returns 200 if the process is running."""
    return HealthResponse()


@app.get("/ready", response_model=ReadyResponse, tags=["infra"])
async def ready() -> ReadyResponse:
    """Readiness probe — reports which models are loaded."""
    loaded: list[str] = []
    if app.state.model is not None:
        loaded.append("retention_model")
    if app.state.rule_engine and app.state.rule_engine.rules:
        loaded.append("recommendation_rules")
    if app.state.rag_index is not None:
        loaded.append("rag_index")

    return ReadyResponse(ready=len(loaded) > 0, models_loaded=loaded)


@app.get("/version", response_model=VersionResponse, tags=["infra"])
async def version() -> VersionResponse:
    """Application and model version information."""
    meta = getattr(app.state, "metadata", {})

    # Try to read the git SHA
    git_sha = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
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
        git_sha=git_sha,
        model_version=meta.get("version"),
        trained_at=meta.get("trained_at"),
    )
