"""Export the product catalogue from cleaned UCI purchases: one row per StockCode.

Only fields present in the source are exported. The data has no categories,
allergens, dietary tags or stock levels, so the catalogue has none either.

Usage::

    python -m src.catalogue
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd

from src.contracts import ARTIFACTS, PROJECT_ROOT
from src.features import purchases
from src.ingest import _sha256, load_config

CATALOGUE_PATH = ARTIFACTS / "product_catalogue.csv"
CATALOGUE_META_PATH = ARTIFACTS / "product_catalogue_meta.json"
COLUMNS = [
    "stock_code",
    "description",
    "other_descriptions",
    "median_unit_price_gbp",
    "invoice_count",
    "first_seen",
    "last_seen",
]


def build_catalogue(df: pd.DataFrame) -> pd.DataFrame:
    """Name each StockCode by its most frequent description; keep variants for search."""
    bought = purchases(df)
    counts = (
        bought.groupby(["StockCode", "Description"])
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["StockCode", "n", "Description"], ascending=[True, False, True])
    )
    names = counts.groupby("StockCode")["Description"].agg(list)
    stats = bought.groupby("StockCode").agg(
        median_unit_price_gbp=("Price", "median"),
        invoice_count=("Invoice", "nunique"),
        first_seen=("InvoiceDate", "min"),
        last_seen=("InvoiceDate", "max"),
    )
    catalogue = stats.assign(
        description=names.str[0],
        other_descriptions=names.str[1:].str.join("; "),
    )
    catalogue = catalogue.reset_index().rename(columns={"StockCode": "stock_code"})
    catalogue["median_unit_price_gbp"] = catalogue["median_unit_price_gbp"].round(2)
    for col in ["first_seen", "last_seen"]:
        catalogue[col] = catalogue[col].dt.date.astype(str)
    catalogue = catalogue.sort_values(["invoice_count", "stock_code"], ascending=[False, True])
    return catalogue[COLUMNS].reset_index(drop=True)


def run() -> pd.DataFrame:
    cfg = load_config()
    source = PROJECT_ROOT / cfg["data"]["parquet_path"]
    catalogue = build_catalogue(pd.read_parquet(source))
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    catalogue.to_csv(CATALOGUE_PATH, index=False)
    meta = {
        "exported_at": datetime.now(UTC).isoformat(),
        "source": "data/processed/transactions.parquet (cleaned UCI Online Retail II)",
        "source_data_sha256": _sha256(source),
        "catalogue_sha256": _sha256(CATALOGUE_PATH),
        "row_count": len(catalogue),
        "columns": COLUMNS,
        "currency": "GBP (source currency; no conversion)",
        "method": (
            "positive purchases only; one row per StockCode; description is the most "
            "frequent canonical description, other variants in other_descriptions; "
            "price is the median observed unit price"
        ),
        "not_in_source": "categories, allergens, dietary tags and stock levels",
    }
    CATALOGUE_META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"[catalogue] {len(catalogue):,} products exported to {CATALOGUE_PATH}")
    return catalogue


if __name__ == "__main__":
    run()
