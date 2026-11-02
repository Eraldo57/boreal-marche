"""
Monitoring checks and automated alerts on SYNTHETIC request logs: hand-built log entries
and reference columns, not workbook rows or real traffic.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import httpx
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from api import monitoring
from api.lifespan import PredictionLog
from api.schemas import Check
from api.storage import ALERT_LOG_TABLE, LOG_METADATA, log_engine
from tests.test_prediction_log_db import _offline_app


def _prediction(x: float, probability: float = 0.5, status: int = 200) -> dict:
    predictions = [] if status != 200 else [{"features": {"x": x}, "probability": probability}]
    return {
        "path": "/predict",
        "status_code": status,
        "latency_ms": 5.0,
        "predictions": predictions,
    }


def _recommend(strategy: str) -> dict:
    return {
        "path": "/recommend",
        "status_code": 200,
        "latency_ms": 3.0,
        "predictions": [],
        "recommendation": {"strategy": strategy, "basket_size": 1, "rules_fired": 1},
    }


def test_prediction_drift_flags_shifted_scores():
    entries = [_prediction(0.5, probability=0.95) for _ in range(120)]
    metrics = monitoring.prediction_drift(entries, np.linspace(0, 1, 1000), 0.2, 0.05, 100)
    psi = next(m for m in metrics if m.statistic == "psi")
    assert psi.feature == "predicted_probability" and psi.alert
    assert monitoring.prediction_drift(entries[:10], np.linspace(0, 1, 50), 0.2, 0.05, 100) == []


def test_input_drift_skips_requests_without_features():
    reference = pd.DataFrame({"x": np.linspace(0, 1, 1000)})
    entries = [_recommend("rule_based"), *[_prediction(5.0) for _ in range(110)]]
    assert monitoring.input_drift(entries, reference)  # measured despite the first entry


def test_data_quality_counts_rejections_and_out_of_range_values():
    reference = pd.DataFrame({"x": np.linspace(0, 1, 200)})
    entries = [_prediction(0.5) for _ in range(110)] + [_prediction(2.0) for _ in range(10)]
    entries += [_prediction(0.5, status=422) for _ in range(10)]
    quality = monitoring.data_quality(entries, reference, min_requests=20, min_predictions=100)
    assert quality["rejected_by_validation"] == 10
    assert quality["invalid_request_rate"] == 10 / 130
    assert quality["out_of_range_share"]["x"] == 10 / 120


def test_rule_fire_rate_ignores_empty_baskets():
    entries = [_recommend("rule_based")] * 6 + [_recommend("popularity_fallback")] * 2
    entries += [_recommend("empty_basket")] * 5
    stats = monitoring.recommendation_stats(entries)
    assert (stats["non_empty_baskets"], stats["rule_fire_rate"]) == (8, 0.75)


def test_snapshot_status_is_the_worst_check():
    log = PredictionLog()
    for i in range(30):
        log.append(_prediction(0.5, status=500 if i < 4 else 200))
    result = monitoring.snapshot(SimpleNamespace(prediction_log=log))
    levels = {c.check: c.level for c in result.checks}
    assert levels["error_rate"] == "red" and result.status == "red"
    assert levels["input_drift"] == "not_evaluated"  # no reference: not computed, not "green"
    assert all(c.rule for c in result.checks), "every check states its rule"
    assert result.requests_by_route == {"/predict": 30}
    assert levels["predict_latency_p95_ms"] == "green"  # single /predict requests at 5 ms
    assert result.latency_p95_by_route_ms == {"/predict": 5.0}


def test_latency_check_ignores_slower_batch_requests():
    log = PredictionLog()
    for _ in range(30):
        log.append(_prediction(0.5))
    for _ in range(10):
        log.append({**_prediction(0.5), "path": "/predict/batch", "latency_ms": 2000.0})
    result = monitoring.snapshot(SimpleNamespace(prediction_log=log))
    check = next(c for c in result.checks if c.check == "predict_latency_p95_ms")
    assert check.level == "green" and check.value == 5.0
    assert result.latency_p95_by_route_ms["/predict/batch"] == 2000.0


def _red_error_rate(level: str = "red") -> list[Check]:
    return [Check(check="error_rate", value=0.2, rule="red above 0.10", level=level)]


def test_alerts_raise_once_resolve_and_persist(tmp_path):
    engine = log_engine(f"sqlite:///{tmp_path / 'logs.db'}")
    LOG_METADATA.create_all(engine)
    sent = []
    manager = monitoring.AlertManager(
        engine,
        ["log", "database", "webhook"],
        webhook_url="https://example.invalid/hook",
        send=lambda url, payload: sent.append(payload),
    )
    (raised,) = manager.process(_red_error_rate())
    assert raised.state == "raised" and raised.delivered_to == ["log", "database", "webhook"]
    assert manager.process(_red_error_rate()) == [], "a check that stays breached alerts once"
    (resolved,) = manager.process(_red_error_rate(level="green"))
    assert resolved.state == "resolved" and manager.active == {}
    assert [p["state"] for p in sent] == ["raised", "resolved"] and "error_rate" in sent[0]["text"]
    with engine.connect() as conn:
        assert conn.scalar(select(func.count()).select_from(ALERT_LOG_TABLE)) == 2
    restarted = monitoring.AlertManager(engine, ["database"])
    assert [e.state for e in restarted.recent] == ["resolved", "raised"]


def test_webhook_failure_is_counted_not_raised():
    def broken(url, payload):
        raise httpx.ConnectError("no route")

    manager = monitoring.AlertManager(None, ["webhook"], "https://example.invalid", send=broken)
    (event,) = manager.process(_red_error_rate())
    assert manager.delivery_failures == 1 and event.delivered_to == []


def test_background_monitor_alerts_without_anyone_watching(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'logs.db'}")
    monkeypatch.setenv("ALERT_INTERVAL_SECONDS", "0.05")
    with TestClient(_offline_app(monkeypatch, tmp_path)) as client:
        for _ in range(3):
            client.post("/recommend", json={"basket": []})  # no rules loaded: 503 errors
        deadline = time.monotonic() + 5
        alerts = client.get("/alerts").json()
        while not alerts["alerts"] and time.monotonic() < deadline:
            time.sleep(0.05)
            alerts = client.get("/alerts").json()
        metrics = client.get("/metrics").json()
    assert "error_rate" in alerts["active"]
    assert alerts["alerts"][0]["check"] == "error_rate" and alerts["alerts"][0]["state"] == "raised"
    assert metrics["status"] == "red"
    assert {c["check"] for c in metrics["checks"]} >= {"error_rate", "prediction_drift"}
