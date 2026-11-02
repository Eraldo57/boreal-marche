"""
Human approval of proposed actions (brief section 3.7).

The retention model only proposes an action for a customer; a named person approves or
rejects it, with a reason. Why this step exists: under Québec's Law 25, a decision based
exclusively on automated processing carries disclosure and human-review obligations, so
no coupon decision here is left to the model alone. Nothing is ever sent automatically:
an approval is recorded, not executed.

POST /decisions  record one decision
GET  /decisions  the most recent decisions, newest first (the audit trail)

No authentication exists in this demo, so the reviewer name is declared, not verified.
"""

from __future__ import annotations

from collections import deque
from datetime import UTC, datetime
from threading import Lock
from typing import Any

import structlog
from fastapi import APIRouter, Query, Request
from sqlalchemy import insert, select
from sqlalchemy.engine import Engine

from api.schemas import DecisionRecord, DecisionRequest
from api.storage import DECISION_LOG_TABLE

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["human review"])


class DecisionLog:
    """Decisions in the log database, or in memory when the API runs without one."""

    def __init__(self, engine: Engine | None) -> None:
        self.engine = engine
        self._lock = Lock()
        self._memory: deque[dict[str, Any]] = deque(maxlen=1000)

    def record(self, decision: DecisionRequest) -> DecisionRecord:
        row = {"decided_at": datetime.now(UTC), **decision.model_dump()}
        with self._lock:
            if self.engine is None:
                row["id"] = len(self._memory) + 1
                self._memory.appendleft(row)
            else:
                with self.engine.begin() as conn:
                    key = conn.execute(
                        insert(DECISION_LOG_TABLE).values(**row)
                    ).inserted_primary_key
                if key is None:
                    raise RuntimeError("The database did not return the new decision's id")
                row["id"] = key[0]
        logger.info("decision_recorded", id=row["id"], decision=row["decision"])
        return DecisionRecord(**row)

    def recent(self, limit: int) -> list[DecisionRecord]:
        if self.engine is None:
            return [DecisionRecord(**row) for row in list(self._memory)[:limit]]
        query = select(DECISION_LOG_TABLE).order_by(DECISION_LOG_TABLE.c.id.desc()).limit(limit)
        with self.engine.connect() as conn:
            return [DecisionRecord(**row) for row in conn.execute(query).mappings()]


def _log(request: Request) -> DecisionLog:
    log = getattr(request.app.state, "decision_log", None)
    if log is None:  # app started without the lifespan (unit tests): keep decisions in memory
        log = request.app.state.decision_log = DecisionLog(None)
    return log


@router.post("/decisions", response_model=DecisionRecord, status_code=201)
async def record_decision(decision: DecisionRequest, request: Request) -> DecisionRecord:
    """Record a reviewer's approval or rejection; nothing is sent to the customer."""
    return _log(request).record(decision)


@router.get("/decisions", response_model=list[DecisionRecord])
async def recent_decisions(
    request: Request, limit: int = Query(default=20, ge=1, le=200)
) -> list[DecisionRecord]:
    """The audit trail: who decided what, when and why."""
    return _log(request).recent(limit)
