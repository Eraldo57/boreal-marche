"""Inference for a SageMaker real-time endpoint: GET /ping and POST /invocations (port 8080).

SageMaker unpacks the training job's model.tar.gz into /opt/ml/model (SM_MODEL_DIR) and
starts the container with ``serve`` (scripts/sagemaker_entry.py). The model is loaded
once, with the same checksum and feature-contract checks as the API, and every instance
goes through the API's own input validation.

Request:  {"instances": [{<the 25 features>}, ...]}  (1 to 500 instances)
Response: {"predictions": [{"probability": p, "will_return": bool}, ...],
           "model_version": ..., "threshold": ...}
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from api.schemas import PredictionRequest
from src.contracts import FEATURE_NAMES
from src.registry import METADATA_FILE, MODEL_FILE, load_checked

state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    folder = Path(os.getenv("SM_MODEL_DIR", "/opt/ml/model"))
    try:
        state["model"], state["metadata"] = load_checked(
            folder / MODEL_FILE, folder / METADATA_FILE
        )
    except Exception as exc:  # /ping reports it; SageMaker then marks the endpoint failed
        state["error"] = f"{type(exc).__name__}: {exc}"
    yield
    state.clear()


app = FastAPI(title="Boréal Marché retention endpoint", lifespan=lifespan)


class Invocation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instances: list[dict[str, Any]] = Field(..., min_length=1, max_length=500)


def _loaded() -> tuple[Any, dict[str, Any]]:
    if "model" not in state:
        raise HTTPException(503, state.get("error", "Model not loaded"))
    return state["model"], state["metadata"]


@app.get("/ping")
async def ping() -> dict[str, str]:
    _loaded()
    return {"status": "ok"}


@app.post("/invocations")
async def invocations(body: Invocation) -> dict[str, Any]:
    model, metadata = _loaded()
    try:
        rows = [
            PredictionRequest(customer_id="endpoint", features=instance).features
            for instance in body.instances
        ]
    except ValidationError as exc:
        raise HTTPException(422, str(exc)[:1000]) from exc
    probabilities = model.predict_proba(pd.DataFrame(rows)[FEATURE_NAMES])[:, 1]
    threshold = metadata["decision_threshold"]
    return {
        "predictions": [
            {"probability": round(float(p), 6), "will_return": bool(p >= threshold)}
            for p in probabilities
        ],
        "model_version": metadata.get("model_version"),
        "threshold": threshold,
    }
