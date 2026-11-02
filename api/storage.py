"""Durable tables of the API, in one database (SQLite locally, PostgreSQL in Compose).

- prediction_log: one row per business request, the source of every monitoring metric.
- alert_log: every alert raised or resolved by the monitor.
- decision_log: every human approval or rejection of a proposed action.

Only decision_log holds a customer identifier: a reviewed decision is about one customer
and has to be traceable for that person's rights. No table stores chat text.
"""

from __future__ import annotations

from pathlib import Path

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
    make_url,
)
from sqlalchemy.engine import Engine

LOG_METADATA = MetaData()

PREDICTION_LOG_TABLE = Table(
    "prediction_log",
    LOG_METADATA,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("logged_at", DateTime(timezone=True), nullable=False),
    Column("path", String(64), nullable=False),
    Column("status_code", Integer, nullable=False),
    Column("latency_ms", Float),
    Column("record", JSON, nullable=False),  # the full structured JSON record
)

ALERT_LOG_TABLE = Table(
    "alert_log",
    LOG_METADATA,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("at", DateTime(timezone=True), nullable=False),
    Column("check", String(64), nullable=False),
    Column("state", String(16), nullable=False),  # raised or resolved
    Column("level", String(16), nullable=False),
    Column("value", Float),
    Column("rule", String(200), nullable=False),
    Column("detail", String(500), nullable=False),
    Column("delivered_to", JSON, nullable=False),
)

DECISION_LOG_TABLE = Table(
    "decision_log",
    LOG_METADATA,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("decided_at", DateTime(timezone=True), nullable=False),
    Column("customer_id", String(100), nullable=False),
    Column("probability", Float, nullable=False),  # what the reviewer saw
    Column("threshold", Float, nullable=False),
    Column("model_version", String(64)),
    Column("proposed_action", String(32), nullable=False),
    Column("decision", String(16), nullable=False),  # approved or rejected
    Column("reviewer", String(100), nullable=False),
    Column("reason", String(500), nullable=False),
)


def log_engine(database_url: str) -> Engine:
    """Engine for the log database; creates a local SQLite file's folder if needed."""
    url = make_url(database_url)
    if url.get_backend_name() == "sqlite" and url.database not in (None, "", ":memory:"):
        Path(url.database).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(database_url, pool_pre_ping=True)
