"""Load validated artifacts once. Missing components cause readiness failures."""

from __future__ import annotations

import json
import os
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

import joblib
import pandas as pd
import structlog
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    func,
    insert,
    make_url,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

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
DEFAULT_LOG_DATABASE = f"sqlite:///{PROJECT_ROOT / 'data/logs/predictions.db'}"

_LOG_METADATA = MetaData()
PREDICTION_LOG_TABLE = Table(
    "prediction_log",
    _LOG_METADATA,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("logged_at", DateTime(timezone=True), nullable=False),
    Column("path", String(64), nullable=False),
    Column("status_code", Integer, nullable=False),
    Column("latency_ms", Float),
    Column("record", JSON, nullable=False),  # the full structured JSON record
)


def _log_engine(database_url: str) -> Engine:
    url = make_url(database_url)
    if url.get_backend_name() == "sqlite" and url.database not in (None, "", ":memory:"):
        Path(url.database).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(database_url, pool_pre_ping=True)


def _log_row(record: dict[str, Any]) -> dict[str, Any]:
    stamp = record.get("timestamp")
    return {
        "logged_at": datetime.fromisoformat(stamp) if stamp else datetime.now(UTC),
        "path": record.get("path", "unknown"),
        "status_code": record.get("status_code", 200),
        "latency_ms": record.get("latency_ms"),
        "record": record,
    }


class PredictionLog:
    """Bounded metrics window plus a durable prediction_log table; no IDs or message text.

    One row per business request: path, status and latency as columns, plus the full
    structured JSON record. Any SQLAlchemy URL works: SQLite locally, PostgreSQL in the
    Compose stack. Single-process local service only. Rotation/purge and multi-worker
    aggregation are explicitly future work. Tests pass no URL and only retain memory.
    """

    def __init__(self, database_url: str | None = None) -> None:
        self._lock = Lock()
        self._buffer: deque[dict[str, Any]] = deque(maxlen=50000)
        self.request_count = self.error_count = self.write_failures = 0
        self._engine: Engine | None = None
        if database_url:
            self._engine = _log_engine(database_url)
            _LOG_METADATA.create_all(self._engine)
            table = PREDICTION_LOG_TABLE
            with self._engine.connect() as conn:
                self.request_count = conn.scalar(select(func.count()).select_from(table)) or 0
                self.error_count = (
                    conn.scalar(
                        select(func.count()).select_from(table).where(table.c.status_code >= 400)
                    )
                    or 0
                )
                newest = conn.scalars(
                    select(table.c.record).order_by(table.c.id.desc()).limit(self._buffer.maxlen)
                ).all()
            self._buffer.extend(reversed(newest))

    def append(self, record: dict[str, Any]) -> None:
        with self._lock:
            self.request_count += 1
            self.error_count += record.get("status_code", 200) >= 400
            self._buffer.append(record)
            if self._engine is None:
                return
            try:
                with self._engine.begin() as conn:
                    conn.execute(insert(PREDICTION_LOG_TABLE).values(**_log_row(record)))
            except SQLAlchemyError as exc:
                # The request itself succeeded; the lost row is counted and shown in /metrics.
                self.write_failures += 1
                logger.error("prediction_log_write_failed", error=str(exc).splitlines()[0])

    @property
    def entries(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._buffer)

    @property
    def latencies_ms(self) -> list[float]:
        return [r["latency_ms"] for r in self.entries if "latency_ms" in r]


class RuleEngine:
    """Compatibility adapter; all matching lives in src.recommend."""

    def __init__(
        self,
        rules: list[dict[str, Any]] | None = None,
        popularity_fallback: list[Any] | None = None,
    ) -> None:
        self.rules = rules or []
        self.popularity_fallback = popularity_fallback or []

    def build_index(self) -> None:
        self.engine = RecommendationEngine.from_records(self.rules, self.popularity_fallback)
        self._index = self.engine._by_item

    def recommend(self, basket: list[str], top_k: int = 5) -> tuple[Any, Any, Any]:
        result = self.engine.recommend(basket, top_k)
        return result["recommendations"], result["because"], result["strategy"]


@asynccontextmanager
async def lifespan(app: Any) -> AsyncIterator[None]:
    app.state.model = app.state.rule_engine = app.state.rag_index = None
    app.state.customer_features = app.state.drift_reference = None
    app.state.metadata, app.state.startup_errors = {}, {}
    app.state.threshold, app.state.feature_order = None, FEATURE_NAMES
    try:
        app.state.prediction_log = PredictionLog(os.getenv("DATABASE_URL", DEFAULT_LOG_DATABASE))
    except Exception as exc:  # e.g. database unreachable: keep serving, report not ready
        app.state.prediction_log = PredictionLog()
        app.state.startup_errors["prediction_log"] = str(exc).splitlines()[0]
        logger.error("prediction_log_startup_failed", error=str(exc).splitlines()[0])
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
