"""Load validated artifacts once. Missing components cause readiness failures."""

from __future__ import annotations

import json
import os
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

import joblib
import pandas as pd
import structlog

from src.contracts import (
    ARTIFACTS,
    FEATURE_NAMES,
    METADATA_PATH,
    MODEL_PATH,
    PROCESSED,
    PROJECT_ROOT,
    validate_metadata,
)
from src.ingest import _sha256
from src.recommend import RecommendationEngine

logger = structlog.get_logger(__name__)
_MODEL_PATH = MODEL_PATH
_META_PATH = METADATA_PATH
_RULES_PATH = ARTIFACTS / "rules.json"
_POPULARITY_PATH = ARTIFACTS / "popularity.json"


class PredictionLog:
    """Bounded metrics window plus durable JSONL; no IDs or message text recorded.

    Single-process local service only. Rotation/purge and multi-worker aggregation
    are explicitly future work. Tests pass no path and only retain memory.
    """

    def __init__(self, path: Path | None = None):
        self.path, self._lock = path, Lock()
        self._buffer = deque(maxlen=50000)
        self.request_count = self.error_count = 0
        if path and path.exists():
            with path.open() as stream:
                for line in stream:
                    try:
                        self._buffer.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("incomplete_log_line")
            self.request_count = len(self._buffer)
            self.error_count = sum(r.get("status_code", 200) >= 400 for r in self._buffer)

    def append(self, record):
        with self._lock:
            self.request_count += 1
            self.error_count += record.get("status_code", 200) >= 400
            self._buffer.append(record)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a") as stream:
                    stream.write(json.dumps(record) + "\n")

    def record_error(self):
        self.append({"status_code": 500})

    @property
    def entries(self):
        with self._lock:
            return list(self._buffer)

    @property
    def latencies_ms(self):
        return [r["latency_ms"] for r in self.entries if "latency_ms" in r]


class RuleEngine:
    """Compatibility adapter; all matching lives in src.recommend."""

    def __init__(self, rules=None, popularity_fallback=None):
        self.rules = rules or []
        self.popularity_fallback = popularity_fallback or []

    def build_index(self):
        self.engine = RecommendationEngine.from_records(self.rules, self.popularity_fallback)
        self._index = self.engine._by_item

    def recommend(self, basket, top_k=5):
        result = self.engine.recommend(basket, top_k)
        return result["recommendations"], result["because"], result["strategy"]


@asynccontextmanager
async def lifespan(app):
    app.state.model = app.state.rule_engine = app.state.rag_index = None
    app.state.customer_features = app.state.drift_reference = None
    app.state.metadata, app.state.startup_errors = {}, {}
    app.state.threshold, app.state.feature_order = None, FEATURE_NAMES
    log_path = Path(os.getenv("BOREAL_LOG_PATH", PROJECT_ROOT / "data/logs/requests.jsonl"))
    app.state.prediction_log = PredictionLog(log_path)
    try:
        metadata = json.loads(_META_PATH.read_text())
        validate_metadata(metadata)
        if _sha256(_MODEL_PATH) != metadata["model_sha256"]:
            raise ValueError("Model checksum does not match metadata")
        model = joblib.load(_MODEL_PATH)  # trusted locally-generated artifacts only
        if list(model.feature_names_in_) != FEATURE_NAMES:
            raise ValueError("Serialized estimator feature order mismatch")
        app.state.model, app.state.metadata = model, metadata
        app.state.threshold = metadata["decision_threshold"]
    except Exception as exc:
        app.state.startup_errors["retention_model"] = str(exc)
        logger.error("retention_startup_failed", error=str(exc))
    try:
        app.state.rule_engine = RecommendationEngine.load(_RULES_PATH, _POPULARITY_PATH)
    except Exception as exc:
        app.state.startup_errors["recommendation_rules"] = str(exc)
        logger.error("recommendation_startup_failed", error=str(exc))
    try:
        from src.rag.index import RagIndex

        app.state.rag_index = RagIndex()
    except Exception as exc:
        app.state.startup_errors["rag_index"] = str(exc)
        logger.error("rag_startup_failed", error=str(exc))
    try:
        app.state.customer_features = pd.read_parquet(PROCESSED / "customer_features.parquet")
        if list(app.state.customer_features.columns) != FEATURE_NAMES:
            raise ValueError("Customer feature store contract mismatch")
    except Exception as exc:
        app.state.customer_features = None
        app.state.startup_errors["customer_features"] = str(exc)
    ref = ARTIFACTS / "drift_reference.parquet"
    if ref.exists():
        app.state.drift_reference = pd.read_parquet(ref)
    logger.info("startup_complete", errors=app.state.startup_errors)
    yield
    logger.info("shutdown")


def log_prediction(prediction_log, record):
    prediction_log.append(record)
