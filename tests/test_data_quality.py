"""Data-quality assertions on the real processed data.

CI rebuilds data/processed from the public UCI workbook, verified against its pinned
SHA-256, and runs this module on every pull request (brief section 3.6). Each check follows
a cleaning rule in configs/pipeline.yaml as src/ingest.py applies it, or a feature
definition in src/features.py. Elsewhere it skips until `python -m src.ingest` and
`python -m src.features` have run.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.contracts import FEATURE_NAMES, PROCESSED
from src.ingest import SERVICE_SKUS, _sha256, load_config

FILES = ("manifest.json", "transactions.parquet", "X.parquet", "y.parquet", "snapshots.parquet")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not all((PROCESSED / name).exists() for name in FILES),
        reason="Run python -m src.ingest and python -m src.features to build the processed data",
    ),
]

BINARY = ("is_one_time_buyer", "has_ever_cancelled", "is_domestic")
SHARES = (  # bounded 0-1 by their definitions in src/features.py
    "single_item_basket_ratio",
    "top_category_share",
    "cancellation_rate",
    "return_rate",
    "weekend_purchase_ratio",
    "morning_purchase_ratio",
)


def test_processed_data_quality():
    config = load_config()
    manifest = json.loads((PROCESSED / "manifest.json").read_text())
    t = pd.read_parquet(PROCESSED / "transactions.parquet")

    # The manifest describes this file, and the file comes from the pinned workbook.
    assert manifest["source_sha256"] == config["data"]["source_sha256"]
    assert _sha256(PROCESSED / "transactions.parquet") == manifest["processed_sha256"]
    assert len(t) == manifest["row_count"]
    assert t["IsPurchase"].sum() == manifest["purchase_rows"]
    assert t["IsCancellation"].sum() == manifest["cancellation_rows"]
    assert t["IsReturn"].sum() == manifest["return_rows"]
    assert str(t["InvoiceDate"].min()) == manifest["date_min"]
    assert str(t["InvoiceDate"].max()) == manifest["date_max"]

    # Cleaning rules, as src/ingest.py applies them.
    required = ["Invoice", "StockCode", "InvoiceDate", "Customer ID"]
    assert t[required].notna().all().all(), "remove_invalid_schema / remove_null_customer"
    assert (t["Quantity"] != 0).all(), "remove_zero_quantity"
    assert (t["Price"] > 0).all(), "remove_non_positive_price"
    assert not t["StockCode"].str.upper().isin(SERVICE_SKUS).any(), "remove_service_skus"
    canonical = t["Description"].str.lower().str.strip().str.replace(r"\s+", " ", regex=True)
    assert (t["Description"] == canonical).all(), "canonicalise_descriptions"
    assert (t["Description"] != "").all(), "canonicalise_descriptions"
    caps = next(r for r in config["cleaning"]["rules"] if r["name"] == "cap_extreme_values")
    assert (t["Quantity"].abs() <= caps["max_abs_quantity"]).all(), "cap_extreme_values"
    assert (t["Price"] <= caps["max_unit_price"]).all(), "cap_extreme_values"
    assert np.allclose(t["LineTotal"], t["Quantity"] * t["Price"])

    # Event flags (flag_events): returns are kept, and never counted as purchases.
    assert (t["IsCancellation"] == t["Invoice"].str.upper().str.startswith("C")).all()
    assert (t["IsReturn"] == (t["Quantity"] < 0)).all()
    assert (t["IsPurchase"] == (~t["IsCancellation"] & (t["Quantity"] > 0))).all()
    assert t["IsReturn"].any(), "returns are preserved, not dropped"
    assert not (t["IsPurchase"] & t["IsReturn"]).any()

    # The feature table, by the definitions in src/features.py.
    features = pd.read_parquet(PROCESSED / "X.parquet")
    y = pd.read_parquet(PROCESSED / "y.parquet")["will_return"]
    snapshots = pd.read_parquet(PROCESSED / "snapshots.parquet")
    assert list(features.columns) == FEATURE_NAMES
    assert len(features) == len(y) == len(snapshots)
    values = features.to_numpy(dtype=float)
    assert np.isfinite(values).all(), "no missing or infinite feature values"
    assert y.notna().all() and set(pd.unique(y)) <= {0, 1}, "binary target"
    assert (features["recency_days"] >= 0).all()
    assert (features["frequency"] >= 1).all()
    assert (features["days_since_first_purchase"] >= features["recency_days"]).all()
    for column in BINARY:
        assert features[column].isin([0, 1]).all(), column
    for column in SHARES:
        assert features[column].between(0, 1).all(), column
