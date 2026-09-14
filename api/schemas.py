"""
Pydantic v2 schemas — the contract between every component.

Every request and response passes through these models.
Extra fields are FORBIDDEN so a misspelled key fails loudly.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = "ok"
    message: str = "Boréal Marché API is running."

class ReadyResponse(BaseModel):
    ready: bool
    models_loaded: list[str] = []

class VersionResponse(BaseModel):
    version: str
    git_sha: str | None = None
    model_version: str | None = None
    trained_at: str | None = None


# ---------------------------------------------------------------------------
# Head A — Retention prediction
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    """Single-customer retention prediction."""
    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(..., description="Boréal Marché customer identifier")
    features: dict[str, float] = Field(
        ...,
        description="Pre-computed feature vector keyed by feature name",
    )

class PredictionResponse(BaseModel):
    customer_id: str
    probability: float = Field(..., ge=0.0, le=1.0)
    will_return: bool
    threshold: float
    risk_label: str  # "high", "medium", "low"
    top_factors: list[dict[str, Any]] = []
    model_version: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customers: list[PredictionRequest] = Field(..., max_length=500)

class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int


# ---------------------------------------------------------------------------
# Head B — Recommendation
# ---------------------------------------------------------------------------
class RecommendationRequest(BaseModel):
    """Given a basket of item descriptions, recommend next items."""
    model_config = ConfigDict(extra="forbid")

    basket: list[str] = Field(
        ...,
        description="List of product descriptions currently in the basket",
    )
    top_k: int = Field(default=5, ge=1, le=20)
    customer_id: str | None = None

class FiringRule(BaseModel):
    antecedent: list[str]
    consequent: list[str]
    confidence: float
    lift: float
    support: float

class RecommendationResponse(BaseModel):
    recommendations: list[str]
    because: list[FiringRule] = Field(
        default_factory=list,
        description="The association rules that produced each recommendation",
    )
    strategy: str = Field(
        default="rule_based",
        description="'rule_based' | 'popularity_fallback' | 'empty_basket'",
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Head C — Assistant
# ---------------------------------------------------------------------------
class Provider(str, Enum):
    REGEX = "regex"
    OPENAI = "openai"
    OLLAMA = "ollama"

class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(..., min_length=1, max_length=2000)
    conversation_id: str | None = None
    customer_id: str | None = None
    provider: Provider = Provider.REGEX
    language: str = Field(default="en", pattern="^(en|fr)$")

class Citation(BaseModel):
    source: str
    chunk: str
    score: float

class ToolCall(BaseModel):
    tool_name: str
    arguments: dict[str, Any]
    result: Any

class ChatResponse(BaseModel):
    reply: str
    citations: list[Citation] = []
    tool_calls: list[ToolCall] = []
    provider_used: str
    refused: bool = False
    refusal_reason: str | None = None
    language: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------
class DriftMetric(BaseModel):
    feature: str
    statistic: str  # "psi" or "ks"
    value: float
    threshold: float
    alert: bool

class MonitoringResponse(BaseModel):
    request_count: int
    error_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    drift_metrics: list[DriftMetric] = []
    status: str  # "green", "yellow", "red"
