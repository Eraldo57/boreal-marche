"""Live monitoring: metrics, explicit thresholds, a traffic light and automated alerts.

All metrics come from the prediction log's window (its newest 50,000 requests) and the
training reference saved with the model (artifacts/drift_reference.parquet). Thresholds
come from configs/pipeline.yaml (monitoring), and every check carries its rule, so the
dashboard shows each value next to the line it is judged against.

A background task re-evaluates the checks every interval and raises an alert as soon as a
check turns yellow or red, and a "resolved" event when it recovers, so nobody has to be
watching the dashboard. Alerts go to the structured log, the alert_log table and, when
ALERT_WEBHOOK_URL is set, a webhook (for example a Slack or Teams incoming webhook).
Design notes and the delayed-label backfill design: docs/MONITORING.md.
"""

from __future__ import annotations

import asyncio
from collections import Counter, deque
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx
import numpy as np
import pandas as pd
import structlog
from scipy.stats import ks_2samp
from sqlalchemy import insert, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from api.schemas import AlertEvent, Check, DriftMetric, MonitoringResponse
from api.storage import ALERT_LOG_TABLE
from src.ingest import load_config

logger = structlog.get_logger(__name__)
PREDICTION_ROUTES = {"/predict", "/predict/batch"}
SEVERITY = {"not_evaluated": 0, "green": 1, "yellow": 2, "red": 3}


def settings() -> dict[str, Any]:
    return dict(load_config()["monitoring"])


def _percentile(values: list[float], q: float) -> float:
    """Return the q-th percentile, or 0.0 if the list is empty."""
    return float(np.percentile(values, q)) if values else 0.0


def _psi_ks(expected: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """PSI over the reference's deciles (open-ended outer bins) and the KS p-value."""
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
    return psi, float(ks_2samp(expected, values).pvalue)


def _drift_pair(
    name: str, expected: np.ndarray, values: np.ndarray, psi_limit: float, ks_limit: float
) -> list[DriftMetric]:
    psi, ks_p = _psi_ks(expected, values)
    return [
        DriftMetric(
            feature=name, statistic="psi", value=psi, threshold=psi_limit, alert=psi > psi_limit
        ),
        DriftMetric(
            feature=name,
            statistic="ks_pvalue",
            value=ks_p,
            threshold=ks_limit,
            alert=ks_p < ks_limit,
        ),
    ]


def prediction_records(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every logged retention prediction that carries its input features."""
    return [p for entry in entries for p in entry.get("predictions", []) if p.get("features")]


def input_drift(
    entries: list[dict[str, Any]],
    reference: pd.DataFrame | None,
    psi_threshold: float = 0.2,
    ks_threshold: float = 0.05,
    min_predictions: int = 100,
) -> list[DriftMetric]:
    """Measured PSI and two-sided KS p-value per feature; empty means unavailable, not zero."""
    records = prediction_records(entries)
    if reference is None or len(records) < min_predictions:
        return []
    metrics: list[DriftMetric] = []
    for name in reference.columns:
        values = np.array([r["features"][name] for r in records if name in r["features"]])
        if len(values) < min_predictions // 2:
            continue
        expected = reference[name].dropna().to_numpy()
        metrics += _drift_pair(name, expected, values.astype(float), psi_threshold, ks_threshold)
    return metrics


def prediction_drift(
    entries: list[dict[str, Any]],
    reference_scores: np.ndarray | None,
    psi_threshold: float,
    ks_threshold: float,
    min_predictions: int,
) -> list[DriftMetric]:
    """Shift of the served probabilities against the model's scores on the reference."""
    scores = [r["probability"] for r in prediction_records(entries) if "probability" in r]
    if reference_scores is None or len(scores) < min_predictions:
        return []
    values = np.asarray(scores, dtype=float)
    return _drift_pair(
        "predicted_probability", reference_scores, values, psi_threshold, ks_threshold
    )


def data_quality(
    entries: list[dict[str, Any]],
    reference: pd.DataFrame | None,
    min_requests: int,
    min_predictions: int,
) -> dict[str, Any]:
    """Prediction requests rejected by validation, and values outside the training range."""
    requests = [e for e in entries if e.get("path") in PREDICTION_ROUTES]
    rejected = sum(e.get("status_code") == 422 for e in requests)
    result: dict[str, Any] = {
        "prediction_requests": len(requests),
        "rejected_by_validation": rejected,
        "invalid_request_rate": rejected / len(requests) if len(requests) >= min_requests else None,
        "out_of_range_share": {},
    }
    records = prediction_records(entries)
    if reference is not None and len(records) >= min_predictions:
        frame = pd.DataFrame([r["features"] for r in records])
        low, high = reference.min(), reference.max()
        for name in reference.columns.intersection(frame.columns):
            outside = (frame[name] < low[name]) | (frame[name] > high[name])
            result["out_of_range_share"][name] = float(outside.mean())
    return result


def recommendation_stats(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """How often rules fire for non-empty baskets; empty baskets use popularity by design."""
    served = [
        e["recommendation"]
        for e in entries
        if e.get("path") == "/recommend" and e.get("status_code") == 200 and e.get("recommendation")
    ]
    strategies = Counter(r["strategy"] for r in served)
    non_empty = strategies["rule_based"] + strategies["popularity_fallback"]
    return {
        "requests": len(served),
        "by_strategy": dict(strategies),
        "non_empty_baskets": non_empty,
        "rule_fire_rate": strategies["rule_based"] / non_empty if non_empty else None,
    }


def _above(value: float | None, yellow: float, red: float | None = None) -> str:
    if value is None:
        return "not_evaluated"
    if red is not None and value > red:
        return "red"
    return "yellow" if value > yellow else "green"


def _drift_check(name: str, metrics: list[DriftMetric], psi: float, ks: float) -> Check:
    alerting = sorted({m.feature for m in metrics if m.alert})
    worst = max((m.value for m in metrics if m.statistic == "psi"), default=None)
    return Check(
        check=name,
        value=worst,
        rule=f"red when any PSI > {psi} or KS p-value < {ks}; value is the largest PSI",
        level="not_evaluated" if not metrics else "red" if alerting else "green",
        detail=f"alerting: {', '.join(alerting)}" if alerting else "",
    )


def snapshot(state: Any) -> MonitoringResponse:
    """All metrics and checks for the current window."""
    cfg = settings()
    drift, limits = cfg["drift"], cfg["thresholds"]
    log = state.prediction_log
    entries = log.entries
    total = log.request_count
    error_rate = log.error_count / total if total else 0.0
    latencies = log.latencies_ms
    reference = getattr(state, "drift_reference", None)
    psi, ks, min_predictions = (
        drift["psi_threshold"],
        drift["ks_threshold"],
        drift["min_predictions"],
    )
    inputs = input_drift(entries, reference, psi, ks, min_predictions)
    outputs = prediction_drift(
        entries, getattr(state, "prediction_reference", None), psi, ks, min_predictions
    )
    quality = data_quality(entries, reference, limits["min_requests"], min_predictions)
    recommendation = recommendation_stats(entries)
    out_of_range = quality["out_of_range_share"]
    too_often = sorted(
        f for f, share in out_of_range.items() if share > limits["out_of_range_share_yellow"]
    )
    fire_rate = (
        recommendation["rule_fire_rate"]
        if recommendation["non_empty_baskets"] >= limits["min_requests"]
        else None
    )
    p95 = _percentile(latencies, 95)
    by_route: dict[str, list[float]] = {}
    for entry in entries:
        if "latency_ms" in entry:
            by_route.setdefault(entry.get("path", "unknown"), []).append(entry["latency_ms"])
    # The brief's latency target is for single predictions (Appendix B: /predict p95); a
    # 500-row batch is legitimately slower and would hide or fake a problem.
    single = by_route.get("/predict", [])
    predict_p95 = _percentile(single, 95) if single else None
    checks = [
        Check(
            check="error_rate",
            value=error_rate if total else None,
            rule=f"yellow above {limits['error_rate_yellow']}, "
            f"red above {limits['error_rate_red']}",
            level=_above(
                error_rate if total else None, limits["error_rate_yellow"], limits["error_rate_red"]
            ),
        ),
        Check(
            check="predict_latency_p95_ms",
            value=predict_p95,
            rule=f"yellow above {limits['latency_p95_ms_yellow']} ms, single /predict requests",
            level=_above(predict_p95, limits["latency_p95_ms_yellow"]),
        ),
        _drift_check("input_drift", inputs, psi, ks),
        _drift_check("prediction_drift", outputs, psi, ks),
        Check(
            check="invalid_request_rate",
            value=quality["invalid_request_rate"],
            rule=f"yellow above {limits['invalid_request_rate_yellow']} "
            f"(from {limits['min_requests']} prediction requests)",
            level=_above(quality["invalid_request_rate"], limits["invalid_request_rate_yellow"]),
        ),
        Check(
            check="out_of_range_share",
            value=max(out_of_range.values()) if out_of_range else None,
            rule=f"yellow when any feature has more than {limits['out_of_range_share_yellow']} "
            "of its values outside the training range",
            level=_above(
                max(out_of_range.values()) if out_of_range else None,
                limits["out_of_range_share_yellow"],
            ),
            detail=f"features: {', '.join(too_often)}" if too_often else "",
        ),
        Check(
            check="rule_fire_rate",
            value=fire_rate,
            rule=f"yellow below {limits['rule_fire_rate_yellow']} "
            f"(from {limits['min_requests']} non-empty baskets)",
            level="not_evaluated"
            if fire_rate is None
            else "yellow"
            if fire_rate < limits["rule_fire_rate_yellow"]
            else "green",
        ),
    ]
    worst = max((c.level for c in checks), key=SEVERITY.__getitem__)
    return MonitoringResponse(
        request_count=total,
        error_rate=round(error_rate, 4),
        latency_p50_ms=round(_percentile(latencies, 50), 2),
        latency_p95_ms=round(p95, 2),
        latency_p99_ms=round(_percentile(latencies, 99), 2),
        drift_metrics=inputs,
        status="green" if worst == "not_evaluated" else worst,
        drift_status="measured" if inputs else "insufficient_data_or_reference",
        log_write_failures=log.write_failures,
        requests_by_route=dict(Counter(e.get("path", "unknown") for e in entries)),
        latency_p95_by_route_ms={
            route: round(_percentile(v, 95), 2) for route, v in by_route.items()
        },
        prediction_drift=outputs,
        data_quality=quality,
        recommendation=recommendation,
        checks=checks,
    )


def _post_json(url: str, payload: dict[str, Any]) -> None:
    httpx.post(url, json=payload, timeout=5.0).raise_for_status()


class AlertManager:
    """Raise an alert when a check turns yellow or red; resolve it when it recovers.

    A check that stays breached alerts once (again only if its level changes). Delivery
    failures are counted and logged, never raised, so monitoring cannot stop the API.
    After a restart the active set starts empty, so a still-breached check alerts again.
    """

    def __init__(
        self,
        engine: Engine | None,
        channels: list[str],
        webhook_url: str | None = None,
        send: Callable[[str, dict[str, Any]], None] = _post_json,
        history: int = 200,
    ) -> None:
        self.engine, self.channels, self.webhook_url, self._send = (
            engine,
            channels,
            webhook_url,
            send,
        )
        self.active: dict[str, str] = {}
        self.recent: deque[AlertEvent] = deque(maxlen=history)
        self.delivery_failures = 0
        if engine is not None and "database" in channels:
            with engine.connect() as conn:
                rows = conn.execute(
                    select(ALERT_LOG_TABLE).order_by(ALERT_LOG_TABLE.c.id.desc()).limit(history)
                ).mappings()
                self.recent.extend(
                    AlertEvent(**{k: v for k, v in r.items() if k != "id"}) for r in rows
                )

    def process(self, checks: list[Check]) -> list[AlertEvent]:
        breached = {c.check: c for c in checks if c.level in ("yellow", "red")}
        events = [
            self._event(c, "raised")
            for name, c in breached.items()
            if self.active.get(name) != c.level
        ]
        by_name = {c.check: c for c in checks}
        for name in sorted(set(self.active) - set(breached)):
            events.append(self._event(by_name[name], "resolved", previous=self.active[name]))
        self.active = {name: c.level for name, c in breached.items()}
        for event in events:
            self._deliver(event)
        return events

    @staticmethod
    def _event(check: Check, state: str, previous: str | None = None) -> AlertEvent:
        return AlertEvent(
            at=datetime.now(UTC),
            check=check.check,
            state=state,
            level=previous if state == "resolved" and previous else check.level,
            value=check.value,
            rule=check.rule,
            detail=check.detail if state == "raised" else f"now {check.level}",
        )

    def _deliver(self, event: AlertEvent) -> None:
        if "log" in self.channels:
            log = logger.warning if event.state == "raised" else logger.info
            log("monitoring_alert", **event.model_dump(mode="json", exclude={"delivered_to"}))
            event.delivered_to.append("log")
        if "database" in self.channels and self.engine is not None:
            row = event.model_dump(exclude={"delivered_to"})
            row.update(rule=event.rule[:200], detail=event.detail[:500])
            try:
                with self.engine.begin() as conn:
                    delivered = [*event.delivered_to, "database"]
                    conn.execute(insert(ALERT_LOG_TABLE).values(**row, delivered_to=delivered))
                event.delivered_to.append("database")
            except SQLAlchemyError as exc:
                self._failed("database", exc)
        if "webhook" in self.channels and self.webhook_url:
            value = "n/a" if event.value is None else f"{event.value:.4g}"
            text = f"[{event.level}] {event.check} {event.state}: {value} ({event.rule})"
            try:
                self._send(
                    self.webhook_url,
                    {"text": f"{text} {event.detail}".strip(), **event.model_dump(mode="json")},
                )
                event.delivered_to.append("webhook")
            except Exception as exc:  # any delivery problem: count it, keep serving
                self._failed("webhook", exc)
        self.recent.appendleft(event)

    def _failed(self, channel: str, exc: Exception) -> None:
        self.delivery_failures += 1
        logger.error("alert_delivery_failed", channel=channel, error=type(exc).__name__)


async def run_alert_loop(app: Any, interval_seconds: float) -> None:
    """Evaluate the checks every interval and alert on changes, until cancelled."""
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            await asyncio.to_thread(lambda: app.state.alerts.process(snapshot(app.state).checks))
        except Exception as exc:  # monitoring must never stop the service
            logger.error("alert_evaluation_failed", error=type(exc).__name__)
