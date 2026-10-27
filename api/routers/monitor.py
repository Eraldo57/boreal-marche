"""
Monitoring endpoint.

GET /metrics  Live operational metrics computed from the in-memory prediction log.
"""

from __future__ import annotations

import numpy as np
import structlog
from fastapi import APIRouter, Request
from scipy.stats import ks_2samp

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
    reference=None,
    psi_threshold: float = 0.2,
    ks_threshold: float = 0.05,
) -> list[DriftMetric]:
    """Measured PSI and two-sided KS p-values; empty means unavailable, not zero."""
    entries = [p for entry in entries for p in entry.get("predictions", [])]
    if reference is None or len(entries) < 100:
        return []

    # Collect feature names from the first entry
    sample = entries[0].get("features", {})
    if not sample:
        return []

    metrics: list[DriftMetric] = []
    feature_names = list(reference.columns)

    for fname in feature_names:
        values = [e["features"][fname] for e in entries if fname in e.get("features", {})]
        if len(values) < 50:
            continue

        expected = reference[fname].dropna().to_numpy()
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, 11)))
        if len(edges) < 3:
            edges = np.unique(np.r_[-np.inf, np.unique(expected) + 0.5, np.inf])
        else:
            edges[0], edges[-1] = -np.inf, np.inf
        ref_hist = np.histogram(expected, bins=edges)[0].astype(float) + 0.5
        now_hist = np.histogram(values, bins=edges)[0].astype(float) + 0.5
        ref_hist /= ref_hist.sum()
        now_hist /= now_hist.sum()
        psi = float(np.sum((now_hist - ref_hist) * np.log(now_hist / ref_hist)))
        ks_p = float(ks_2samp(expected, values).pvalue)
        metrics.append(
            DriftMetric(
                feature=fname,
                statistic="psi",
                value=psi,
                threshold=psi_threshold,
                alert=psi > psi_threshold,
            )
        )
        metrics.append(
            DriftMetric(
                feature=fname,
                statistic="ks_pvalue",
                value=ks_p,
                threshold=ks_threshold,
                alert=ks_p < ks_threshold,
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
    drift_metrics = _compute_drift(entries, getattr(request.app.state, "drift_reference", None))

    status = _traffic_light(error_rate, drift_metrics)

    return MonitoringResponse(
        request_count=total,
        error_rate=round(error_rate, 4),
        latency_p50_ms=round(_percentile(latencies, 50), 2),
        latency_p95_ms=round(_percentile(latencies, 95), 2),
        latency_p99_ms=round(_percentile(latencies, 99), 2),
        drift_metrics=drift_metrics,
        status=status,
        drift_status="measured" if drift_metrics else "insufficient_data_or_reference",
    )
