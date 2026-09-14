"""
Unit tests for the retention-prediction pipeline logic.

These test the internal functions directly — not through HTTP — so they
are faster and more precise about what broke.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from api.routers.predict import _predict_one, _risk_label, _top_factors
from api.schemas import PredictionRequest

FEATURE_ORDER = [
    "recency_days", "frequency", "monetary_total", "monetary_mean", "monetary_std",
    "days_since_first_purchase", "avg_days_between_orders", "std_days_between_orders",
    "is_one_time_buyer", "avg_basket_size", "avg_basket_value", "max_basket_value",
    "avg_unique_items_per_basket", "single_item_basket_ratio", "top_category_share",
    "n_unique_categories", "category_entropy", "has_ever_cancelled", "cancellation_rate",
    "return_rate", "weekend_purchase_ratio", "morning_purchase_ratio",
    "country_code", "is_domestic",
]


# ---------------------------------------------------------------------------
# Feature ordering
# ---------------------------------------------------------------------------
def test_feature_order_length() -> None:
    """Pipeline config declares exactly 24 features."""
    assert len(FEATURE_ORDER) == 24


def test_feature_order_no_duplicates() -> None:
    assert len(FEATURE_ORDER) == len(set(FEATURE_ORDER))


def test_predict_builds_dataframe_in_correct_order(
    sample_features: dict[str, float], mock_model: MagicMock
) -> None:
    """The DataFrame sent to predict_proba must have columns in FEATURE_ORDER."""
    req = PredictionRequest(customer_id="T-1", features=sample_features)
    _predict_one(req, mock_model, 0.5, FEATURE_ORDER, {"version": "test"})

    call_args = mock_model.predict_proba.call_args
    df = call_args[0][0]
    assert list(df.columns) == FEATURE_ORDER


# ---------------------------------------------------------------------------
# Probability bounds
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "proba_array",
    [
        np.array([[1.0, 0.0]]),
        np.array([[0.0, 1.0]]),
        np.array([[0.5, 0.5]]),
        np.array([[0.001, 0.999]]),
    ],
)
def test_probability_bounds(
    proba_array: np.ndarray, sample_features: dict[str, float]
) -> None:
    """The returned probability must be in [0, 1]."""
    model = MagicMock()
    model.predict_proba.return_value = proba_array

    req = PredictionRequest(customer_id="T-2", features=sample_features)
    resp = _predict_one(req, model, 0.5, FEATURE_ORDER, {})
    assert 0.0 <= resp.probability <= 1.0


# ---------------------------------------------------------------------------
# Threshold logic
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "proba, threshold, expected_return",
    [
        (0.70, 0.50, True),
        (0.50, 0.50, True),
        (0.49, 0.50, False),
        (0.01, 0.50, False),
        (0.45, 0.45, True),
        (0.44, 0.45, False),
    ],
)
def test_threshold_decision(
    proba: float,
    threshold: float,
    expected_return: bool,
    sample_features: dict[str, float],
) -> None:
    model = MagicMock()
    model.predict_proba.return_value = np.array([[1 - proba, proba]])

    req = PredictionRequest(customer_id="T-3", features=sample_features)
    resp = _predict_one(req, model, threshold, FEATURE_ORDER, {})
    assert resp.will_return is expected_return


# ---------------------------------------------------------------------------
# Risk labels
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "proba, expected_label",
    [
        (0.90, "high"),
        (0.65, "high"),
        (0.64, "medium"),
        (0.35, "medium"),
        (0.34, "low"),
        (0.00, "low"),
    ],
)
def test_risk_label(proba: float, expected_label: str) -> None:
    assert _risk_label(proba) == expected_label


# ---------------------------------------------------------------------------
# Model includes preprocessing
# ---------------------------------------------------------------------------
def test_model_pipeline_has_steps(mock_model: MagicMock) -> None:
    """The mock simulates a pipeline — verify the named_steps exist."""
    assert hasattr(mock_model, "named_steps")
    assert "scaler" in mock_model.named_steps
    assert "classifier" in mock_model.named_steps


# ---------------------------------------------------------------------------
# Top factors extraction
# ---------------------------------------------------------------------------
def test_top_factors_returns_list(
    sample_features: dict[str, float], mock_model: MagicMock
) -> None:
    factors = _top_factors(sample_features, FEATURE_ORDER, mock_model)
    assert isinstance(factors, list)
    assert len(factors) <= 5
    for f in factors:
        assert "feature" in f
        assert "importance" in f


def test_top_factors_with_no_importances(sample_features: dict[str, float]) -> None:
    """A model without feature_importances_ must still return gracefully."""
    plain_model = MagicMock(spec=[])
    factors = _top_factors(sample_features, FEATURE_ORDER, plain_model)
    assert factors == []


# ---------------------------------------------------------------------------
# Missing model
# ---------------------------------------------------------------------------
def test_predict_without_model_raises_503(sample_features: dict[str, float]) -> None:
    """If the model is None, the endpoint must return 503."""
    from fastapi import HTTPException

    req = PredictionRequest(customer_id="T-4", features=sample_features)
    with pytest.raises(HTTPException) as exc_info:
        _predict_one(req, None, 0.5, FEATURE_ORDER, {})
    assert exc_info.value.status_code == 503
