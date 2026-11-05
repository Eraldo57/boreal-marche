"""
Shared fixtures for the Boréal Marché test suite.

All fixtures use mocks so the test suite runs without trained model files.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from api.lifespan import PredictionLog, RuleEngine

# ---------------------------------------------------------------------------
# Feature fixtures
# ---------------------------------------------------------------------------
from src.contracts import FEATURE_NAMES as FEATURE_ORDER


@pytest.fixture(autouse=True)
def isolated_prediction_log(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """Every test that starts the app logs to a throwaway table, never to data/logs."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'predictions.db'}")


@pytest.fixture(autouse=True)
def no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never read the developer's .env, so a real OPENAI_API_KEY cannot reach them."""
    monkeypatch.setenv("BOREAL_NO_DOTENV", "1")


@pytest.fixture(autouse=True)
def local_model_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never read the developer's MLflow registry; registry tests opt back in.

    mlflow.set_tracking_uri/set_registry_uri also write these environment variables, so
    registering them here makes pytest undo whatever a test (or MLflow) sets.
    """
    monkeypatch.setenv("MODEL_SOURCE", "local")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_REGISTRY_URI", raising=False)


@pytest.fixture
def sample_features() -> dict[str, float]:
    """A plausible feature vector for one customer."""
    all_features = {
        "recency_days": 15.0,
        "frequency": 8.0,
        "monetary_total": 1250.0,
        "monetary_mean": 156.25,
        "monetary_std": 45.0,
        "days_since_first_purchase": 400.0,
        "avg_days_between_orders": 50.0,
        "std_days_between_orders": 12.0,
        "is_one_time_buyer": 0.0,
        "avg_basket_size": 5.0,
        "avg_basket_value": 62.50,
        "max_basket_value": 150.0,
        "avg_unique_items_per_basket": 4.0,
        "single_item_basket_ratio": 0.1,
        "top_category_share": 0.4,
        "n_unique_categories": 6.0,
        "category_entropy": 1.5,
        "has_ever_cancelled": 0.0,
        "cancellation_rate": 0.0,
        "return_rate": 0.02,
        "weekend_purchase_ratio": 0.3,
        "morning_purchase_ratio": 0.5,
        "country_code": 1.0,
        "is_domestic": 1.0,
        "n_countries": 1.0,
    }
    return {name: all_features[name] for name in FEATURE_ORDER}


@pytest.fixture
def sample_basket() -> list[str]:
    """A small product basket."""
    return ["organic maple syrup", "sourdough bread", "free-range eggs"]


@pytest.fixture
def empty_basket() -> list[str]:
    return []


# ---------------------------------------------------------------------------
# Mock model
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_model() -> MagicMock:
    """A fake sklearn pipeline that returns deterministic probabilities."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.35, 0.65]])
    model.predict.return_value = np.array([1])

    # Simulate a pipeline with feature_importances_
    inner_estimator = MagicMock()
    inner_estimator.feature_importances_ = np.random.dirichlet(np.ones(len(FEATURE_ORDER)))
    model.named_steps = {"scaler": MagicMock(), "classifier": inner_estimator}

    return model


# ---------------------------------------------------------------------------
# Mock rule engine
# ---------------------------------------------------------------------------
SAMPLE_RULES = [
    {
        "antecedent": ["organic maple syrup"],
        "consequent": ["pancake mix"],
        "confidence": 0.75,
        "lift": 3.2,
        "support": 0.02,
    },
    {
        "antecedent": ["sourdough bread"],
        "consequent": ["butter"],
        "confidence": 0.60,
        "lift": 2.8,
        "support": 0.03,
    },
    {
        "antecedent": ["sourdough bread", "butter"],
        "consequent": ["jam"],
        "confidence": 0.55,
        "lift": 2.5,
        "support": 0.015,
    },
    {
        "antecedent": ["free-range eggs"],
        "consequent": ["bacon"],
        "confidence": 0.50,
        "lift": 2.1,
        "support": 0.025,
    },
    {
        "antecedent": ["free-range eggs", "bacon"],
        "consequent": ["orange juice"],
        "confidence": 0.45,
        "lift": 1.9,
        "support": 0.01,
    },
]


@pytest.fixture
def mock_rule_engine() -> RuleEngine:
    engine = RuleEngine(
        rules=SAMPLE_RULES,
        popularity_fallback=["coffee", "milk", "bananas", "rice", "olive oil"],
    )
    engine.build_index()
    return engine


# ---------------------------------------------------------------------------
# Mock RAG index
# ---------------------------------------------------------------------------
class MockRagIndex:
    """Returns canned retrieval results for testing."""

    def query(self, text: str, top_k: int = 5) -> list[dict[str, Any]]:
        if "delivery" in text.lower() or "livraison" in text.lower():
            return [
                {
                    "source": "delivery_policy.md",
                    "text": "Boréal Marché delivers within 2-5 business days across Québec.",
                    "score": 0.85,
                },
            ]
        if "return" in text.lower() or "retour" in text.lower():
            return [
                {
                    "source": "return_policy.md",
                    "text": "Returns are accepted within 30 days for non-perishable items.",
                    "score": 0.80,
                },
            ]
        if "allergen" in text.lower() or "allergène" in text.lower():
            return [
                {
                    "source": "allergen_policy.md",
                    "text": "All products list allergens on their packaging "
                    "per Health Canada requirements.",
                    "score": 0.78,
                },
            ]
        # Below similarity floor
        return [{"source": "general.md", "text": "General info", "score": 0.10}]


@pytest.fixture
def mock_rag_index() -> MockRagIndex:
    return MockRagIndex()


# ---------------------------------------------------------------------------
# Prediction log fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def prediction_log() -> PredictionLog:
    return PredictionLog()


# ---------------------------------------------------------------------------
# FastAPI test client — with all mocks wired in
# ---------------------------------------------------------------------------
@pytest.fixture
async def client(
    mock_model: MagicMock,
    mock_rule_engine: RuleEngine,
    mock_rag_index: MockRagIndex,
    prediction_log: PredictionLog,
) -> AsyncClient:
    """An httpx AsyncClient pointing at the FastAPI app with mocked state."""
    from api.main import app

    # Inject mocks
    app.state.model = mock_model
    app.state.metadata = {
        "model_version": "test-v0.1",
        "trained_at": "2026-08-30T12:00:00",
        "decision_threshold": 0.45,
        "feature_names": FEATURE_ORDER,
    }
    app.state.threshold = 0.45
    app.state.feature_order = FEATURE_ORDER
    app.state.rule_engine = mock_rule_engine
    app.state.rag_index = mock_rag_index
    app.state.prediction_log = prediction_log
    app.state.startup_errors = {}
    app.state.customer_features = None
    app.state.drift_reference = None

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
