"""
Pydantic v2 schemas — the contract between every component.

Every request and response passes through these models.
Extra fields are FORBIDDEN so a misspelled key fails loudly.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.contracts import FEATURE_NAMES


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = "ok"
    message: str = "Boréal Marché API is running."


class ReadyResponse(BaseModel):
    ready: bool
    models_loaded: list[str] = []
    errors: dict[str, str] = Field(default_factory=dict)


class VersionResponse(BaseModel):
    version: str
    git_sha: str | None = None
    model_version: str | None = None
    trained_at: str | None = None
    model_family: str | None = None
    calibration_method: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    decision_threshold: float | None = None
    # Where the served model came from: the registry alias and version, or the local
    # fallback with its reason.
    model_source: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Head A — Retention prediction
# ---------------------------------------------------------------------------
class PredictionRequest(StrictModel):
    """Single-customer retention prediction."""

    customer_id: str = Field(
        ..., min_length=1, max_length=100, description="Demo customer identifier"
    )
    features: dict[str, float] = Field(
        ...,
        description="Pre-computed feature vector keyed by feature name",
    )

    @field_validator("features", mode="before")
    @classmethod
    def canonical_features(cls, value: Any) -> Any:
        if not isinstance(value, dict) or set(value) != set(FEATURE_NAMES):
            raise ValueError(
                f"features must contain exactly the canonical {len(FEATURE_NAMES)} names; "
                "see /feature-contract"
            )
        for name, number in value.items():
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(number)
            ):
                raise ValueError(f"{name} must be a finite number")
            if number < (-1 if name == "country_code" else 0):
                raise ValueError(f"{name} is outside its valid range")
            if (
                name.endswith("_ratio") or name.endswith("_rate") or name == "top_category_share"
            ) and number > 1:
                raise ValueError(f"{name} must be between zero and one")
            if name in (
                "is_one_time_buyer",
                "has_ever_cancelled",
                "is_domestic",
            ) and number not in (0, 1):
                raise ValueError(f"{name} must be 0 or 1")
        return value


class FeatureEffect(BaseModel):
    """How far one feature of this customer moves their probability, against a typical value."""

    feature: str
    value: float | None
    typical_value: float = Field(..., description="The training median of this feature.")
    effect: float = Field(
        ...,
        description=(
            "Change in the probability of returning when the feature has this customer's "
            "value instead of the typical one. Positive raises it."
        ),
    )


class PredictionResponse(BaseModel):
    customer_id: str
    probability: float = Field(..., ge=0.0, le=1.0)
    will_return: bool
    threshold: float
    risk_label: str  # "high", "medium", "low"
    top_factors: list[dict[str, Any]] = []  # global importances, the same for every customer
    explanation: list[FeatureEffect] = []  # this customer's own largest feature effects
    model_version: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DecisionRequest(StrictModel):
    """A person's decision on the action the model proposed for one customer."""

    customer_id: str = Field(..., min_length=1, max_length=100)
    probability: float = Field(..., ge=0.0, le=1.0)
    threshold: float = Field(..., ge=0.0, le=1.0)
    model_version: str | None = Field(default=None, max_length=64)
    proposed_action: Literal["offer_coupon", "no_action"]
    decision: Literal["approved", "rejected"]
    reviewer: str = Field(..., min_length=1, max_length=100, pattern=r"\S")
    reason: str = Field(..., min_length=1, max_length=500, pattern=r"\S")


class DecisionRecord(BaseModel):
    id: int
    decided_at: datetime
    customer_id: str
    probability: float
    threshold: float
    model_version: str | None = None
    proposed_action: str
    decision: str
    reviewer: str
    reason: str


class BatchPredictionRequest(StrictModel):
    model_config = ConfigDict(extra="forbid")
    customers: list[PredictionRequest] = Field(..., min_length=1, max_length=500)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int


# ---------------------------------------------------------------------------
# Head B — Recommendation
# ---------------------------------------------------------------------------
class RecommendationRequest(StrictModel):
    """Given a basket of item descriptions, recommend next items."""

    model_config = ConfigDict(extra="forbid")

    basket: list[str] = Field(
        ...,
        description="List of product descriptions currently in the basket",
        max_length=100,
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
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RecommendationDetailsResponse(BaseModel):
    items: list[str]
    rules: list[FiringRule]
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Head C — Assistant
# ---------------------------------------------------------------------------
class Provider(StrEnum):
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
    # For policy answers: "uncertain" when the best passage scored between the similarity
    # floor and the confident level, so it may not answer the question. None otherwise.
    passage_confidence: Literal["confident", "uncertain"] | None = None
    language: str
    provider_latency_ms: float | None = None
    token_usage: dict[str, int] | None = None
    cost_usd: float | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------
class DriftMetric(BaseModel):
    feature: str
    statistic: str  # "psi" or "ks"
    value: float
    threshold: float
    alert: bool


class Check(BaseModel):
    """One traffic-light rule: the measured value next to the rule it is judged by."""

    check: str
    value: float | None = None
    rule: str  # e.g. "yellow above 0.05, red above 0.10"
    level: str  # "green", "yellow", "red" or "not_evaluated"
    detail: str = ""


class MonitoringResponse(BaseModel):
    request_count: int
    error_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    drift_metrics: list[DriftMetric] = []  # input drift, per feature
    status: str  # "green", "yellow", "red": the worst level among the checks
    drift_status: str = "not_evaluated"
    window_description: str = "bounded local request log"
    log_write_failures: int = 0  # rows that could not be written to the prediction_log table
    requests_by_route: dict[str, int] = {}
    latency_p95_by_route_ms: dict[str, float] = {}
    prediction_drift: list[DriftMetric] = []  # distribution of predicted probabilities
    data_quality: dict[str, Any] = {}
    recommendation: dict[str, Any] = {}
    checks: list[Check] = []


class AlertEvent(BaseModel):
    at: datetime
    check: str
    state: str  # "raised" or "resolved"
    level: str
    value: float | None = None
    rule: str
    detail: str = ""
    delivered_to: list[str] = []


class AlertsResponse(BaseModel):
    active: list[str]  # checks currently breached
    alerts: list[AlertEvent]  # newest first
    channels: list[str]
    webhook_configured: bool
    delivery_failures: int
