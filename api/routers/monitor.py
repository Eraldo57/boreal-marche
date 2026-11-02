"""
Monitoring endpoints.

GET /metrics  Live metrics, explicit thresholds and the traffic light (api/monitoring.py).
GET /alerts   Alerts raised and resolved by the background monitor, newest first.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from api.monitoring import snapshot
from api.schemas import AlertsResponse, MonitoringResponse

router = APIRouter(tags=["monitoring"])


@router.get("/metrics", response_model=MonitoringResponse)
async def metrics(request: Request) -> MonitoringResponse:
    """Return live operational metrics with each check's rule and level."""
    return snapshot(request.app.state)


@router.get("/alerts", response_model=AlertsResponse)
async def alerts(request: Request) -> AlertsResponse:
    """Alerts delivered by the background monitor; empty before the first evaluation."""
    manager = getattr(request.app.state, "alerts", None)
    if manager is None:
        return AlertsResponse(
            active=[], alerts=[], channels=[], webhook_configured=False, delivery_failures=0
        )
    return AlertsResponse(
        active=sorted(manager.active),
        alerts=list(manager.recent),
        channels=manager.channels,
        webhook_configured=manager.webhook_url is not None,
        delivery_failures=manager.delivery_failures,
    )
