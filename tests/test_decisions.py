"""Human approval step: decisions are validated, recorded and auditable. Synthetic values."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select

from api.storage import DECISION_LOG_TABLE
from tests.test_prediction_log_db import _offline_app

DECISION = {
    "customer_id": "12347",
    "probability": 0.81,
    "threshold": 0.22,
    "model_version": "test-v0.1",
    "proposed_action": "offer_coupon",
    "decision": "rejected",
    "reviewer": "Test Reviewer",
    "reason": "Already a frequent customer; a coupon is unlikely to change anything.",
}


@pytest.mark.asyncio
async def test_decision_is_recorded_and_listed(client):
    created = await client.post("/decisions", json=DECISION)
    assert created.status_code == 201
    body = created.json()
    assert body["id"] >= 1 and body["decision"] == "rejected" and body["decided_at"]
    listed = (await client.get("/decisions")).json()
    assert listed[0]["reviewer"] == "Test Reviewer" and listed[0]["id"] == body["id"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        {"reviewer": "   "},  # a name is required, not just spaces
        {"reason": ""},
        {"decision": "maybe"},
        {"proposed_action": "send_coupon_now"},
        {"probability": 1.5},
        {"auto_send": True},  # extra fields are forbidden
    ],
)
async def test_invalid_decisions_are_rejected(client, change):
    response = await client.post("/decisions", json={**DECISION, **change})
    assert response.status_code == 422


def test_decisions_are_stored_in_the_log_database(monkeypatch, tmp_path):
    url = f"sqlite:///{tmp_path / 'logs.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    with TestClient(_offline_app(monkeypatch, tmp_path)) as client:
        assert client.post("/decisions", json=DECISION).status_code == 201
    with TestClient(_offline_app(monkeypatch, tmp_path)) as client:  # survives a restart
        assert client.get("/decisions").json()[0]["reason"] == DECISION["reason"]
    with create_engine(url).connect() as conn:
        (row,) = conn.execute(select(DECISION_LOG_TABLE)).mappings()
    assert (row["customer_id"], row["proposed_action"]) == ("12347", "offer_coupon")
    assert not client.post("/decisions", json={**DECISION, "decision": "maybe"}).is_success
