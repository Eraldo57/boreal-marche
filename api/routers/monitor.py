"""
Monitoring endpoint.

GET /metrics  Live operational metrics computed from the in-memory prediction log.
"""

from __future__ import annotations

import numpy as np
import structlog
from fastapi import APIRouter, Request

from api.schemas import DriftMetric, MonitoringResponse

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["monitoring"])


def _percentile(values: list[float], q: float) -> float:
    """Return the q-th percentile, or 0.0 if the list is empty."""
    if not values:
        return 0.0
    return float(np.percentile(values, q))


def _compute_drift(
    entries: list[dict],
    psi_threshold: float = 0.2,
    ks_threshold: float = 0.05,
) -> list[DriftMetric]:
    """Placeholder drift computation.

    Real implementation would compare the feature distributions in ``entries``
    against a stored training-time reference.  Here we return empty until the
    reference window is populated.
    """
    if len(entries) < 100:
        return []

    # Collect feature names from the first entry
    sample = entries[0].get("features", {})
    if not sample:
        return []

    metrics: list[DriftMetric] = []
    feature_names = list(sample.keys())[:10]  # limit to first 10 for performance

    for fname in feature_names:
        values = [e["features"][fname] for e in entries if fname in e.get("features", {})]
        if len(values) < 50:
            continue

        # PSI placeholder — a real implementation would bucket against the
        # reference distribution.  For now, report 0 (no drift).
        metrics.append(
            DriftMetric(
                feature=fname,
                statistic="psi",
                value=0.0,
                threshold=psi_threshold,
                alert=False,
            )
        )

    return metrics


def _traffic_light(error_rate: float, drift_metrics: list[DriftMetric]) -> str:
    """Compute the overall status traffic light."""
    if error_rate > 0.10:
        return "red"
    if any(m.alert for m in drift_metrics):
        return "red"
    if error_rate > 0.05:
        return "yellow"
    return "green"


@router.get("/metrics", response_model=MonitoringResponse)
async def metrics(request: Request) -> MonitoringResponse:
    """Return live operational metrics."""
    prediction_log = request.app.state.prediction_log

    total = prediction_log.request_count
    errors = prediction_log.error_count
    error_rate = errors / total if total > 0 else 0.0

    latencies = prediction_log.latencies_ms

    entries = prediction_log.entries
    drift_metrics = _compute_drift(entries)

    status = _traffic_light(error_rate, drift_metrics)

    return MonitoringResponse(
        request_count=total,
        error_rate=round(error_rate, 4),
        latency_p50_ms=round(_percentile(latencies, 50), 2),
        latency_p95_ms=round(_percentile(latencies, 95), 2),
        latency_p99_ms=round(_percentile(latencies, 99), 2),
        drift_metrics=drift_metrics,
        status=status,
    )
