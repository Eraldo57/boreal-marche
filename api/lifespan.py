"""
Application lifespan — load models once at startup, release at shutdown.

Everything expensive (joblib, JSON rules, RAG index) is loaded here and
stored on ``app.state`` so every request reads from memory.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import joblib
import structlog
from fastapi import FastAPI

logger = structlog.get_logger(__name__)

# Default paths — overridable via environment or tests
_MODEL_PATH = Path("artifacts/retention_model.joblib")
_META_PATH = Path("artifacts/model_metadata.json")
_RULES_PATH = Path("artifacts/rules.json")
_CORPUS_DIR = Path("docs/corpus")

# Maximum prediction log entries kept in memory for the monitoring dashboard
_MAX_LOG_SIZE = 50_000


# ---------------------------------------------------------------------------
# In-memory prediction log — one record per /predict call
# ---------------------------------------------------------------------------
@dataclass
class PredictionLog:
    """Thread-safe circular buffer that stores structured prediction records."""

    _buffer: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=_MAX_LOG_SIZE))
    _lock: Lock = field(default_factory=Lock)
    error_count: int = 0
    request_count: int = 0

    def append(self, record: dict[str, Any]) -> None:
        with self._lock:
            self.request_count += 1
            self._buffer.append(record)

    def record_error(self) -> None:
        with self._lock:
            self.error_count += 1
            self.request_count += 1

    @property
    def entries(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._buffer)

    @property
    def latencies_ms(self) -> list[float]:
        with self._lock:
            return [r["latency_ms"] for r in self._buffer if "latency_ms" in r]


# ---------------------------------------------------------------------------
# Recommendation rule index — antecedent → list of rules
# ---------------------------------------------------------------------------
@dataclass
class RuleEngine:
    """Pre-indexed association rules for O(1) antecedent lookup."""

    rules: list[dict[str, Any]] = field(default_factory=list)
    _index: dict[frozenset[str], list[dict[str, Any]]] = field(default_factory=dict)
    popularity_fallback: list[str] = field(default_factory=list)

    def build_index(self) -> None:
        """Build a lookup from frozenset(antecedent) → matching rules."""
        self._index.clear()
        for rule in self.rules:
            key = frozenset(rule["antecedent"])
            self._index.setdefault(key, []).append(rule)
        logger.info("rule_index_built", n_rules=len(self.rules), n_keys=len(self._index))

    def recommend(
        self, basket: list[str], top_k: int = 5
    ) -> tuple[list[str], list[dict[str, Any]], str]:
        """Return (recommendations, firing_rules, strategy).

        Strategy is one of: 'rule_based', 'popularity_fallback', 'empty_basket'.
        """
        if not basket:
            return self.popularity_fallback[:top_k], [], "empty_basket"

        basket_set = frozenset(item.lower().strip() for item in basket)

        # Collect all rules whose antecedent is a subset of the basket.
        matched: list[dict[str, Any]] = []
        for key, rules in self._index.items():
            if key.issubset(basket_set):
                matched.extend(rules)

        if not matched:
            return self.popularity_fallback[:top_k], [], "popularity_fallback"

        # Sort by lift (highest first), then deduplicate consequents
        matched.sort(key=lambda r: r.get("lift", 0), reverse=True)

        seen: set[str] = set()
        recommendations: list[str] = []
        firing_rules: list[dict[str, Any]] = []

        for rule in matched:
            for item in rule["consequent"]:
                normed = item.lower().strip()
                if normed not in basket_set and normed not in seen:
                    seen.add(normed)
                    recommendations.append(item)
                    firing_rules.append(rule)
                    if len(recommendations) >= top_k:
                        break
            if len(recommendations) >= top_k:
                break

        if not recommendations:
            return self.popularity_fallback[:top_k], [], "popularity_fallback"

        return recommendations, firing_rules, "rule_based"


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Load every artefact once, serve for the lifetime of the process."""
    log = logger.bind(event="startup")

    # --- Retention model ---------------------------------------------------
    model = None
    metadata: dict[str, Any] = {}
    threshold = 0.5
    feature_order: list[str] = []

    if _MODEL_PATH.exists():
        model = joblib.load(_MODEL_PATH)
        log.info("retention_model_loaded", path=str(_MODEL_PATH))
    else:
        log.warning("retention_model_missing", path=str(_MODEL_PATH))

    if _META_PATH.exists():
        with open(_META_PATH) as f:
            metadata = json.load(f)
        threshold = metadata.get("threshold", 0.5)
        feature_order = metadata.get("feature_order", [])
        log.info("model_metadata_loaded", version=metadata.get("version"))
    else:
        log.warning("model_metadata_missing", path=str(_META_PATH))

    # --- Recommendation rules ---------------------------------------------
    rule_engine = RuleEngine()
    if _RULES_PATH.exists():
        with open(_RULES_PATH) as f:
            rules_data = json.load(f)
        rule_engine.rules = rules_data.get("rules", [])
        rule_engine.popularity_fallback = rules_data.get("popularity_fallback", [])
        rule_engine.build_index()
        log.info("rules_loaded", n_rules=len(rule_engine.rules))
    else:
        log.warning("rules_file_missing", path=str(_RULES_PATH))

    # --- RAG index ---------------------------------------------------------
    rag_index = None
    try:
        from src.rag.index import build_index

        if _CORPUS_DIR.exists():
            rag_index = build_index(str(_CORPUS_DIR))
            log.info("rag_index_built", corpus_dir=str(_CORPUS_DIR))
        else:
            log.warning("corpus_dir_missing", path=str(_CORPUS_DIR))
    except ImportError:
        log.warning("rag_module_not_available")

    # --- Prediction log ----------------------------------------------------
    prediction_log = PredictionLog()

    # --- Expose on app.state -----------------------------------------------
    app.state.model = model
    app.state.metadata = metadata
    app.state.threshold = threshold
    app.state.feature_order = feature_order
    app.state.rule_engine = rule_engine
    app.state.rag_index = rag_index
    app.state.prediction_log = prediction_log

    log.info("startup_complete")

    yield  # ---- application serves requests here ----

    log.info("shutdown", predictions_served=prediction_log.request_count)


def log_prediction(prediction_log: PredictionLog, record: dict[str, Any]) -> None:
    """Append a structured prediction record and log it."""
    prediction_log.append(record)
    structlog.get_logger("predictions").info("prediction_logged", **record)
