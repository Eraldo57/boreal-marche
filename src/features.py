"""
Boréal Marché — Feature Engineering
=====================================

Builds **25+ documented features** from the cleaned transaction data and
creates the binary retention label (``will_return``).

Feature groups
--------------
1. **RFM** (5) — recency, frequency, monetary total / mean / std
2. **Cadence** (4) — tenure, average / std inter-order gap, one-timer flag
3. **Basket shape** (5) — avg size / value, max value, unique items, single-item ratio
4. **Category mix** (3) — top-category share, distinct categories, Shannon entropy
5. **Behavioural** (5) — cancellation & return flags/rates, weekend & morning ratios
6. **Geography** (3) — country code (label-encoded), domestic flag, n_countries

Total: **25 features** (all documented below).

Missingness policy
------------------
**Explicit**.  Features that cannot be computed for a customer (e.g.,
``std_days_between_orders`` for a one-time buyer) are set to a
documented sentinel (``-1`` for undefined ratios, ``0`` for std of a
single observation) and the policy is stated in the feature docstring.

Usage::

    python -m src.features
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline.yaml"


def load_config() -> dict:
    """Return the full pipeline config."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_transactions(cfg: dict) -> pd.DataFrame:
    """Load the cleaned Parquet file produced by ``src.ingest``."""
    path = PROJECT_ROOT / cfg["data"]["parquet_path"]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.ingest` first."
        )
    df = pd.read_parquet(path)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


# ---------------------------------------------------------------------------
# Label creation
# ---------------------------------------------------------------------------
def create_labels(
    df: pd.DataFrame,
    cutoff: str,
    window_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Split transactions into observation / label windows and compute target.

    Parameters
    ----------
    df : DataFrame
        All cleaned transactions.
    cutoff : str
        ISO date string for the observation window boundary.
    window_days : int
        Number of days after *cutoff* that define the label window.

    Returns
    -------
    obs_df : DataFrame
        Transactions in the observation window (feature inputs).
    label_df : DataFrame
        Transactions in the label window (used only to derive the target).
    y : Series
        Boolean series indexed by ``Customer ID``: True if the customer
        placed at least one order in the label window.
    """
    cutoff_dt = pd.Timestamp(cutoff)
    label_end = cutoff_dt + timedelta(days=window_days)

    obs_df = df[df["InvoiceDate"] < cutoff_dt].copy()
    label_df = df[
        (df["InvoiceDate"] >= cutoff_dt) & (df["InvoiceDate"] < label_end)
    ].copy()

    # Customers who appear in the observation window
    obs_customers = set(obs_df["Customer ID"].unique())

    # Which of those ordered in the label window?
    label_customers = set(label_df["Customer ID"].unique()) & obs_customers

    y = pd.Series(
        {cid: cid in label_customers for cid in obs_customers},
        name="will_return",
    )

    print(f"[features] Observation window: < {cutoff}")
    print(f"[features] Label window:       {cutoff} … {label_end.date()}")
    print(f"[features] Customers in obs:   {len(obs_customers):,}")
    print(f"[features] Positive labels:    {y.sum():,}  ({y.mean():.1%})")

    return obs_df, label_df, y


# ---------------------------------------------------------------------------
# Feature builders — each returns a DataFrame indexed by Customer ID
# ---------------------------------------------------------------------------

def _rfm_features(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Recency / Frequency / Monetary features (5 features).

    - ``recency_days``: days between the customer's last order and the cutoff.
    - ``frequency``: number of distinct invoices.
    - ``monetary_total``: sum of all line totals.
    - ``monetary_mean``: average invoice total.
    - ``monetary_std``: std of invoice totals.  *Missingness:* 0 for one-order customers.
    """
    inv = (
        df.groupby(["Customer ID", "Invoice"])
        .agg(
            invoice_date=("InvoiceDate", "min"),
            invoice_total=("LineTotal", "sum"),
        )
        .reset_index()
    )
    agg = inv.groupby("Customer ID").agg(
        last_date=("invoice_date", "max"),
        frequency=("Invoice", "nunique"),
        monetary_total=("invoice_total", "sum"),
        monetary_mean=("invoice_total", "mean"),
        monetary_std=("invoice_total", "std"),
    )
    agg["recency_days"] = (cutoff - agg["last_date"]).dt.days
    agg["monetary_std"] = agg["monetary_std"].fillna(0.0)  # single-order → 0
    return agg[["recency_days", "frequency", "monetary_total",
                "monetary_mean", "monetary_std"]]


def _cadence_features(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Purchase-cadence features (4 features).

    - ``days_since_first_purchase``: tenure in days.
    - ``avg_days_between_orders``: mean inter-order gap.
    - ``std_days_between_orders``: std of inter-order gaps.
      *Missingness:* 0 for ≤ 2 orders (undefined std).
    - ``is_one_time_buyer``: 1 if only one invoice, else 0.
    """
    inv_dates = (
        df.groupby(["Customer ID", "Invoice"])["InvoiceDate"]
        .min()
        .reset_index()
        .sort_values(["Customer ID", "InvoiceDate"])
    )

    def _gap_stats(g: pd.DataFrame) -> pd.Series:
        dates = g["InvoiceDate"].values
        first = pd.Timestamp(dates[0])
        tenure = (cutoff - first).days
        if len(dates) < 2:
            return pd.Series({
                "days_since_first_purchase": tenure,
                "avg_days_between_orders": float(tenure),
                "std_days_between_orders": 0.0,
                "is_one_time_buyer": 1,
            })
        gaps = np.diff(dates).astype("timedelta64[D]").astype(float)
        return pd.Series({
            "days_since_first_purchase": tenure,
            "avg_days_between_orders": float(np.mean(gaps)),
            "std_days_between_orders": float(np.std(gaps, ddof=1)) if len(gaps) > 1 else 0.0,
            "is_one_time_buyer": 0,
        })

    return inv_dates.groupby("Customer ID").apply(_gap_stats, include_groups=False)


def _basket_shape_features(df: pd.DataFrame) -> pd.DataFrame:
    """Basket-shape features (5 features).

    - ``avg_basket_size``: average number of line items per invoice.
    - ``avg_basket_value``: average invoice total.
    - ``max_basket_value``: largest single invoice total.
    - ``avg_unique_items_per_basket``: average distinct SKUs per invoice.
    - ``single_item_basket_ratio``: fraction of invoices with exactly 1 line item.
    """
    inv = df.groupby(["Customer ID", "Invoice"]).agg(
        n_lines=("Quantity", "size"),
        total=("LineTotal", "sum"),
        n_unique=("StockCode", "nunique"),
    ).reset_index()

    agg = inv.groupby("Customer ID").agg(
        avg_basket_size=("n_lines", "mean"),
        avg_basket_value=("total", "mean"),
        max_basket_value=("total", "max"),
        avg_unique_items_per_basket=("n_unique", "mean"),
    )
    singles = inv.groupby("Customer ID")["n_lines"].apply(
        lambda s: (s == 1).mean()
    ).rename("single_item_basket_ratio")
    return agg.join(singles)


def _category_mix_features(df: pd.DataFrame) -> pd.DataFrame:
    """Category-mix features (3 features).

    Categories are approximated by the first token of the product description
    (e.g., "red" from "red polkadot mug").

    - ``top_category_share``: fraction of spend in the most-purchased category.
    - ``n_unique_categories``: count of distinct categories.
    - ``category_entropy``: Shannon entropy of the category distribution.
      *Missingness:* 0 if only one category (no diversity).
    """
    tmp = df.copy()
    tmp["category"] = tmp["Description"].str.split().str[0].fillna("unknown")

    cat_spend = (
        tmp.groupby(["Customer ID", "category"])["LineTotal"]
        .sum()
        .reset_index()
    )

    def _cat_stats(g: pd.DataFrame) -> pd.Series:
        total = g["LineTotal"].sum()
        share = g["LineTotal"] / total if total > 0 else g["LineTotal"]
        top_share = share.max()
        n_cats = len(g)
        # Shannon entropy: -Σ p·log(p)
        probs = share.values
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log2(probs))) if len(probs) > 1 else 0.0
        return pd.Series({
            "top_category_share": top_share,
            "n_unique_categories": n_cats,
            "category_entropy": entropy,
        })

    return cat_spend.groupby("Customer ID").apply(_cat_stats, include_groups=False)


def _behavioural_features(
    obs_df: pd.DataFrame, full_df: pd.DataFrame
) -> pd.DataFrame:
    """Behavioural features (5 features).

    Uses the *full* (pre-cleaning) cancellation signal where available.

    - ``has_ever_cancelled``: 1 if the customer has any cancelled invoice.
    - ``cancellation_rate``: fraction of invoices that are cancellations.
    - ``return_rate``: fraction of units returned (Quantity < 0 rows).
      *Missingness:* 0 if no negative-quantity rows survive.
    - ``weekend_purchase_ratio``: fraction of orders placed on Sat/Sun.
    - ``morning_purchase_ratio``: fraction of orders placed before noon.
    """
    # Cancellations — detect from the original Invoice column before cleaning
    inv = obs_df.groupby("Customer ID")["Invoice"].apply(
        lambda s: s.unique()
    )
    has_cancel = inv.apply(
        lambda arr: int(any(str(x).startswith("C") for x in arr))
    ).rename("has_ever_cancelled")
    cancel_rate = inv.apply(
        lambda arr: sum(1 for x in arr if str(x).startswith("C")) / len(arr)
        if len(arr) > 0 else 0.0
    ).rename("cancellation_rate")

    # Return rate from negative-quantity rows in the FULL dataset
    cust_qty = full_df.groupby("Customer ID")["Quantity"].agg(["sum", "count"])
    neg_qty = full_df[full_df["Quantity"] < 0].groupby("Customer ID")["Quantity"].count()
    return_rate = (neg_qty / cust_qty["count"]).fillna(0.0).rename("return_rate")

    # Weekend / morning
    obs_inv = obs_df.drop_duplicates(subset=["Customer ID", "Invoice"])
    obs_inv = obs_inv.assign(
        is_weekend=(obs_inv["InvoiceDate"].dt.dayofweek >= 5).astype(int),
        is_morning=(obs_inv["InvoiceDate"].dt.hour < 12).astype(int),
    )
    time_agg = obs_inv.groupby("Customer ID")[["is_weekend", "is_morning"]].mean()
    time_agg.columns = ["weekend_purchase_ratio", "morning_purchase_ratio"]

    result = pd.DataFrame(index=obs_df["Customer ID"].unique())
    result = result.join(has_cancel).join(cancel_rate).join(return_rate).join(time_agg)
    result["has_ever_cancelled"] = result["has_ever_cancelled"].fillna(0).astype(int)
    result["cancellation_rate"] = result["cancellation_rate"].fillna(0.0)
    result["return_rate"] = result["return_rate"].fillna(0.0)
    result["weekend_purchase_ratio"] = result["weekend_purchase_ratio"].fillna(0.0)
    result["morning_purchase_ratio"] = result["morning_purchase_ratio"].fillna(0.0)
    return result


def _geography_features(df: pd.DataFrame) -> pd.DataFrame:
    """Geography features (3 features).

    - ``country_code``: label-encoded country (integer).
    - ``is_domestic``: 1 if the customer's primary country is 'United Kingdom'.
    - ``n_countries``: number of distinct countries the customer ordered from.
    """
    # Majority country per customer
    country_mode = (
        df.groupby("Customer ID")["Country"]
        .agg(lambda s: s.mode().iloc[0])
        .rename("primary_country")
    )
    n_countries = (
        df.groupby("Customer ID")["Country"]
        .nunique()
        .rename("n_countries")
    )

    result = pd.DataFrame({"primary_country": country_mode, "n_countries": n_countries})
    result["is_domestic"] = (result["primary_country"] == "United Kingdom").astype(int)

    # Label-encode country
    all_countries = sorted(result["primary_country"].unique())
    country_map = {c: i for i, c in enumerate(all_countries)}
    result["country_code"] = result["primary_country"].map(country_map)
    return result[["country_code", "is_domestic", "n_countries"]]


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------
FEATURE_NAMES: list[str] = [
    # RFM
    "recency_days", "frequency", "monetary_total", "monetary_mean", "monetary_std",
    # Cadence
    "days_since_first_purchase", "avg_days_between_orders",
    "std_days_between_orders", "is_one_time_buyer",
    # Basket shape
    "avg_basket_size", "avg_basket_value", "max_basket_value",
    "avg_unique_items_per_basket", "single_item_basket_ratio",
    # Category mix
    "top_category_share", "n_unique_categories", "category_entropy",
    # Behavioural
    "has_ever_cancelled", "cancellation_rate", "return_rate",
    "weekend_purchase_ratio", "morning_purchase_ratio",
    # Geography
    "country_code", "is_domestic", "n_countries",
]


def build_features(
    obs_df: pd.DataFrame,
    full_df: pd.DataFrame,
    cutoff: str,
) -> pd.DataFrame:
    """Compute all feature groups and concatenate them.

    Parameters
    ----------
    obs_df : DataFrame
        Transactions in the observation window (already cleaned).
    full_df : DataFrame
        All transactions (used for return-rate calculation).
    cutoff : str
        ISO date string for the cutoff boundary.

    Returns
    -------
    DataFrame
        One row per customer, columns are ``FEATURE_NAMES``.
    """
    cutoff_ts = pd.Timestamp(cutoff)
    print("[features] Building RFM …")
    rfm = _rfm_features(obs_df, cutoff_ts)
    print("[features] Building cadence …")
    cadence = _cadence_features(obs_df, cutoff_ts)
    print("[features] Building basket shape …")
    basket = _basket_shape_features(obs_df)
    print("[features] Building category mix …")
    cat_mix = _category_mix_features(obs_df)
    print("[features] Building behavioural …")
    behav = _behavioural_features(obs_df, full_df)
    print("[features] Building geography …")
    geo = _geography_features(obs_df)

    X = rfm.join(cadence).join(basket).join(cat_mix).join(behav).join(geo)

    # Ensure all expected features are present
    missing_cols = set(FEATURE_NAMES) - set(X.columns)
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")

    X = X[FEATURE_NAMES]

    # Final missingness audit
    null_counts = X.isnull().sum()
    if null_counts.any():
        print("[features] WARNING — remaining nulls (will be filled per policy):")
        for col, cnt in null_counts[null_counts > 0].items():
            print(f"  {col}: {cnt} nulls")
        # Explicit fill: numeric features get the column median, with a log
        for col in X.columns:
            if X[col].isnull().any():
                median_val = X[col].median()
                print(f"  → filling '{col}' nulls with median = {median_val:.4f}")
                X[col] = X[col].fillna(median_val)

    print(f"[features] Built {len(X.columns)} features for {len(X):,} customers")
    return X


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save_features(X: pd.DataFrame, y: pd.Series, cfg: dict) -> Path:
    """Save feature matrix and labels to Parquet.

    Returns
    -------
    Path
        Directory containing ``X.parquet`` and ``y.parquet``.
    """
    out_dir = PROJECT_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    X.to_parquet(out_dir / "X.parquet")
    y.to_frame().to_parquet(out_dir / "y.parquet")

    # Also save feature metadata
    meta = {
        "feature_names": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "n_customers": len(X),
        "positive_rate": float(y.mean()),
        "cutoff_date": cfg["labelling"]["cutoff_date"],
        "label_window_days": cfg["labelling"]["label_window_days"],
    }
    with open(out_dir / "feature_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[features] Saved to {out_dir}/")
    return out_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run() -> tuple[pd.DataFrame, pd.Series]:
    """Execute the feature-engineering pipeline end to end."""
    cfg = load_config()

    print("=" * 70)
    print("Boréal Marché — Feature Engineering")
    print("=" * 70)

    full_df = load_transactions(cfg)

    obs_df, _, y = create_labels(
        full_df,
        cutoff=cfg["labelling"]["cutoff_date"],
        window_days=cfg["labelling"]["label_window_days"],
    )

    X = build_features(obs_df, full_df, cutoff=cfg["labelling"]["cutoff_date"])

    # Align X and y (intersection of customer IDs)
    common = X.index.intersection(y.index)
    X = X.loc[common]
    y = y.loc[common]

    save_features(X, y, cfg)

    print("=" * 70)
    print("[features] Done.")
    return X, y


def main() -> None:
    """CLI wrapper."""
    run()


if __name__ == "__main__":
    main()
