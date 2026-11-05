"""Shared feature order and artifact paths for offline and online components."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "artifacts"
PROCESSED = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = ARTIFACTS / "retention_model.joblib"
METADATA_PATH = ARTIFACTS / "retention_model_metadata.json"
CONTRACT_VERSION = "2.0"
ENGINEERED_FEATURE_NAMES = [
    "recency_days",
    "frequency",
    "monetary_total",
    "monetary_mean",
    "monetary_std",
    "days_since_first_purchase",
    "avg_days_between_orders",
    "std_days_between_orders",
    "is_one_time_buyer",
    "avg_basket_size",
    "avg_basket_value",
    "max_basket_value",
    "avg_unique_items_per_basket",
    "single_item_basket_ratio",
    "top_category_share",
    "n_unique_categories",
    "category_entropy",
    "has_ever_cancelled",
    "cancellation_rate",
    "return_rate",
    "weekend_purchase_ratio",
    "morning_purchase_ratio",
    "country_code",
    "is_domestic",
    "n_countries",
]

# Applied from the earlier June-period minimisation audit, before inspecting the new test result.
# All 25 remain documented/engineered offline; only these 12 enter model training and serving.
# Implementation developed with AI assistance; see docs/AI_USE.md.
FEATURE_NAMES = [
    name
    for name in ENGINEERED_FEATURE_NAMES
    if name
    in {
        "recency_days",
        "frequency",
        "monetary_mean",
        "monetary_std",
        "avg_days_between_orders",
        "std_days_between_orders",
        "is_one_time_buyer",
        "avg_basket_size",
        "single_item_basket_ratio",
        "top_category_share",
        "n_unique_categories",
        "morning_purchase_ratio",
    }
]


def validate_metadata(metadata: dict) -> None:
    """Fail closed when an artifact does not match the serving contract."""
    if metadata.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Model feature names/order differ from src.contracts.FEATURE_NAMES")
    if metadata.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("Model contract version mismatch; rerun features and training")
    threshold = metadata.get("decision_threshold")
    if not isinstance(threshold, (float, int)) or not 0 <= threshold <= 1:
        raise ValueError("Model metadata is missing a valid decision_threshold")
    if metadata.get("target") != "will_return":
        raise ValueError("Expected positive target will_return")
