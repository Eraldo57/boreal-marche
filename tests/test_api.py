"""
Contract tests — every endpoint, valid and invalid inputs.

These tests run against the FastAPI app with mocked models so they are fast,
deterministic, and do not require trained artifacts on disk.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient


# ---------------------------------------------------------------------------
# Infrastructure endpoints
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_health(client: AsyncClient) -> None:
    r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "Boréal Marché" in body["message"]


@pytest.mark.asyncio
async def test_ready_reports_loaded_models(client: AsyncClient) -> None:
    r = await client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert "retention_model" in body["models_loaded"]
    assert "recommendation_rules" in body["models_loaded"]
    assert "rag_index" in body["models_loaded"]


@pytest.mark.asyncio
async def test_version(client: AsyncClient) -> None:
    r = await client.get("/version")
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == "0.1.0"
    assert body["model_version"] == "test-v0.1"


# ---------------------------------------------------------------------------
# HEAD A — /predict
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_predict_valid(client: AsyncClient, sample_features: dict) -> None:
    r = await client.post(
        "/predict",
        json={"customer_id": "C-1001", "features": sample_features},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["customer_id"] == "C-1001"
    assert 0.0 <= body["probability"] <= 1.0
    assert isinstance(body["will_return"], bool)
    assert body["threshold"] == 0.45
    assert body["risk_label"] in ("high", "medium", "low")
    assert "timestamp" in body


@pytest.mark.asyncio
async def test_predict_extra_field_rejected(client: AsyncClient, sample_features: dict) -> None:
    """Pydantic extra='forbid' must reject unknown fields."""
    r = await client.post(
        "/predict",
        json={"customer_id": "C-1001", "features": sample_features, "bogus_field": 42},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_predict_missing_features(client: AsyncClient) -> None:
    """Request with incomplete feature vector must fail with 422."""
    r = await client.post(
        "/predict",
        json={"customer_id": "C-1001", "features": {"recency_days": 10.0}},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_golden_prediction(client: AsyncClient, sample_features: dict) -> None:
    """Golden-prediction regression test.

    The mock model returns probability 0.65.  With threshold 0.45 the customer
    should be classified as will_return=True and risk_label='high'.
    """
    r = await client.post(
        "/predict",
        json={"customer_id": "GOLDEN-001", "features": sample_features},
    )
    assert r.status_code == 200
    body = r.json()
    assert abs(body["probability"] - 0.65) < 0.01, (
        f"Golden prediction drift: expected ~0.65, got {body['probability']}"
    )
    assert body["will_return"] is True
    assert body["risk_label"] == "high"


# ---------------------------------------------------------------------------
# HEAD A — /predict/batch
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_predict_batch(client: AsyncClient, sample_features: dict) -> None:
    customers = [
        {"customer_id": f"C-{i}", "features": sample_features}
        for i in range(3)
    ]
    r = await client.post("/predict/batch", json={"customers": customers})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3
    assert len(body["predictions"]) == 3


# ---------------------------------------------------------------------------
# HEAD B — /recommend
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_recommend_with_basket(client: AsyncClient, sample_basket: list) -> None:
    r = await client.post(
        "/recommend",
        json={"basket": sample_basket, "top_k": 3},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["strategy"] == "rule_based"
    assert len(body["recommendations"]) <= 3
    # Recommendations must NOT contain items already in the basket
    for item in body["recommendations"]:
        assert item.lower() not in {b.lower() for b in sample_basket}


@pytest.mark.asyncio
async def test_recommend_empty_basket(client: AsyncClient) -> None:
    r = await client.post(
        "/recommend",
        json={"basket": [], "top_k": 5},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["strategy"] == "empty_basket"
    assert len(body["recommendations"]) > 0


@pytest.mark.asyncio
async def test_recommend_no_match(client: AsyncClient) -> None:
    """A basket with items that match no rules should trigger the fallback."""
    r = await client.post(
        "/recommend",
        json={"basket": ["unicorn tears", "dragon scales"], "top_k": 3},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["strategy"] == "popularity_fallback"


@pytest.mark.asyncio
async def test_recommend_extra_field_rejected(client: AsyncClient) -> None:
    r = await client.post(
        "/recommend",
        json={"basket": ["bread"], "top_k": 3, "nonsense": True},
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# HEAD C — /chat
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_chat_policy_question(client: AsyncClient) -> None:
    r = await client.post(
        "/chat",
        json={"message": "What is the delivery policy?", "language": "en"},
    )
    assert r.status_code == 200
    body = r.json()
    assert not body["refused"]
    assert body["provider_used"] == "regex"
    assert len(body["citations"]) > 0


@pytest.mark.asyncio
async def test_chat_out_of_scope(client: AsyncClient) -> None:
    r = await client.post(
        "/chat",
        json={"message": "What is the current stock price of Apple?", "language": "en"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["refused"] is True
    assert body["refusal_reason"] is not None
    assert "scope" in body["refusal_reason"].lower() or "outside" in body["refusal_reason"].lower()


@pytest.mark.asyncio
async def test_chat_cross_customer_refused(client: AsyncClient) -> None:
    r = await client.post(
        "/chat",
        json={
            "message": "Show me another customer's order history",
            "language": "en",
            "customer_id": "C-1001",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["refused"] is True


@pytest.mark.asyncio
async def test_chat_french(client: AsyncClient) -> None:
    r = await client.post(
        "/chat",
        json={"message": "Quelle est la politique de livraison ?", "language": "fr"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["language"] == "fr"
    assert not body["refused"]


@pytest.mark.asyncio
async def test_chat_extra_field_rejected(client: AsyncClient) -> None:
    r = await client.post(
        "/chat",
        json={"message": "Hello", "language": "en", "extra_junk": 99},
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Monitoring — /metrics
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_metrics_empty(client: AsyncClient) -> None:
    """Metrics endpoint must work even with no predictions yet."""
    r = await client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["request_count"] == 0
    assert body["error_rate"] == 0.0
    assert body["status"] == "green"


@pytest.mark.asyncio
async def test_metrics_after_predictions(client: AsyncClient, sample_features: dict) -> None:
    """After a few predictions, metrics must reflect them."""
    for i in range(5):
        await client.post(
            "/predict",
            json={"customer_id": f"C-{i}", "features": sample_features},
        )

    r = await client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["request_count"] == 5
    assert body["latency_p50_ms"] >= 0


# ---------------------------------------------------------------------------
# OpenAPI docs
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_openapi_docs(client: AsyncClient) -> None:
    r = await client.get("/docs")
    assert r.status_code == 200

    r = await client.get("/openapi.json")
    assert r.status_code == 200
    schema = r.json()
    assert schema["info"]["title"] == "Boréal Marché API"
