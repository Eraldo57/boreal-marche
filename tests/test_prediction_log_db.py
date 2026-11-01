"""
Prediction-log table tests on a temporary SQLite file.

The Compose stack uses PostgreSQL through the same SQLAlchemy code; SQLite keeps these
tests self-contained so they also run on a clean CI checkout. Records are synthetic.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select

from api.lifespan import PREDICTION_LOG_TABLE, PredictionLog

OK_RECORD = {
    "timestamp": "2011-09-01T10:00:00+00:00",
    "path": "/predict",
    "status_code": 200,
    "latency_ms": 12.5,
    "predictions": [{"probability": 0.4, "will_return": True, "features": {"frequency": 3.0}}],
}
ERROR_RECORD = {
    "timestamp": "2011-09-01T10:00:01+00:00",
    "path": "/chat",
    "status_code": 503,
    "latency_ms": 2.0,
    "predictions": [],
}


def _url(tmp_path):
    return f"sqlite:///{tmp_path / 'logs' / 'predictions.db'}"


def _rows(url):
    with create_engine(url).connect() as conn:
        return conn.execute(select(PREDICTION_LOG_TABLE)).mappings().all()


def test_each_record_becomes_one_row_with_its_json(tmp_path):
    url = _url(tmp_path)
    log = PredictionLog(url)
    log.append(OK_RECORD)
    log.append(ERROR_RECORD)
    rows = _rows(url)
    assert [r["path"] for r in rows] == ["/predict", "/chat"]
    assert [r["status_code"] for r in rows] == [200, 503]
    assert rows[0]["latency_ms"] == 12.5
    assert rows[0]["record"] == OK_RECORD


def test_restart_replays_counts_and_window_from_the_table(tmp_path):
    url = _url(tmp_path)
    first = PredictionLog(url)
    first.append(OK_RECORD)
    first.append(ERROR_RECORD)
    restarted = PredictionLog(url)
    assert (restarted.request_count, restarted.error_count) == (2, 1)
    assert restarted.entries == [OK_RECORD, ERROR_RECORD]
    assert restarted.latencies_ms == [12.5, 2.0]


def test_record_without_timestamp_or_path_gets_defaults(tmp_path):
    url = _url(tmp_path)
    PredictionLog(url).append({"status_code": 500})
    (row,) = _rows(url)
    assert (row["path"], row["status_code"]) == ("unknown", 500)
    assert row["logged_at"] is not None


def test_write_failure_is_counted_and_does_not_raise(tmp_path):
    url = _url(tmp_path)
    log = PredictionLog(url)
    PREDICTION_LOG_TABLE.drop(create_engine(url))
    log.append(OK_RECORD)
    assert log.write_failures == 1
    assert log.request_count == 1 and log.entries == [OK_RECORD]


def test_memory_only_log_writes_nothing():
    log = PredictionLog()
    log.append(OK_RECORD)
    assert log.entries == [OK_RECORD] and log.write_failures == 0


def _offline_app(monkeypatch, tmp_path):
    """The real app with no artifacts, so startup is fast and deterministic."""
    import api.lifespan as life
    import src.rag.index as rag
    from api.main import app

    monkeypatch.setattr(life, "_META_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(life, "_RULES_PATH", tmp_path / "absent_rules.json")

    def missing():
        raise FileNotFoundError("Build the index")

    monkeypatch.setattr(rag, "RagIndex", missing)
    return app


def test_middleware_writes_business_requests_only(monkeypatch, tmp_path):
    url = _url(tmp_path)
    monkeypatch.setenv("DATABASE_URL", url)
    with TestClient(_offline_app(monkeypatch, tmp_path)) as client:
        client.get("/health")
        client.post("/recommend", json={"basket": []})
        metrics = client.get("/metrics").json()
    rows = _rows(url)
    assert [r["path"] for r in rows] == ["/recommend"]
    assert rows[0]["record"]["status_code"] == rows[0]["status_code"]
    assert metrics["request_count"] == 1 and metrics["log_write_failures"] == 0


def test_crashing_endpoint_is_logged_as_500(monkeypatch, tmp_path, sample_features):
    url = _url(tmp_path)
    monkeypatch.setenv("DATABASE_URL", url)
    app = _offline_app(monkeypatch, tmp_path)
    crashing_model = MagicMock()
    crashing_model.predict_proba.side_effect = ValueError("synthetic model failure")
    with TestClient(app, raise_server_exceptions=False) as client:
        monkeypatch.setattr(app.state, "model", crashing_model)  # after startup, which resets it
        batch = {"customers": [{"customer_id": "synthetic-1", "features": sample_features}]}
        response = client.post("/predict/batch", json=batch)
        metrics = client.get("/metrics").json()
    assert response.status_code == 500
    assert [(r["path"], r["status_code"]) for r in _rows(url)] == [("/predict/batch", 500)]
    assert metrics["request_count"] == 1 and metrics["error_rate"] == 1.0


def test_unusable_database_keeps_api_live_but_not_ready(monkeypatch, tmp_path):
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("a file where the log directory should be")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{blocker / 'predictions.db'}")
    with TestClient(_offline_app(monkeypatch, tmp_path)) as client:
        assert client.get("/health").status_code == 200
        ready = client.get("/ready")
    assert ready.status_code == 503
    assert "prediction_log" in ready.json()["errors"]
