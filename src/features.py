"""
Boréal Marché — Feature Engineering
=====================================

Builds **25 documented features** from the cleaned transaction data and
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
Structural defaults are explicit: single-order standard deviations are zero,
single-order mean gap is tenure, and unseen countries use -1. Other missing
numeric values remain NaN for the training-only median imputer; feature
construction never estimates medians from all snapshots.

Usage::

    python -m src.features
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.contracts import CONTRACT_VERSION, FEATURE_NAMES
from src.ingest import _sha256

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
        raise FileNotFoundError(f"{path} not found. Run `python -m src.ingest` first.")
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
    label_df = df[(df["InvoiceDate"] >= cutoff_dt) & (df["InvoiceDate"] < label_end)].copy()
    # Only genuine positive purchases establish eligibility or a return label.
    obs_df = purchases(obs_df)
    label_df = purchases(label_df)

    # Customers who appear in the observation window
    obs_customers = set(obs_df["Customer ID"].unique())

    # Which of those ordered in the label window?
    label_customers = set(label_df["Customer ID"].unique()) & obs_customers

    y = pd.Series(
        {cid: cid in label_customers for cid in sorted(obs_customers)},
        name="will_return",
    )

    print(f"[features] Observation window: < {cutoff}")
    print(f"[features] Label window:       {cutoff} … {label_end.date()}")
    print(f"[features] Customers in obs:   {len(obs_customers):,}")
    print(f"[features] Positive labels:    {y.sum():,}  ({y.mean():.1%})")

    return obs_df, label_df, y


def purchases(df: pd.DataFrame) -> pd.DataFrame:
    """Purchase view of the event ledger; cancellations never count as orders."""
    mask = (
        ~df["Invoice"].astype(str).str.upper().str.startswith("C")
        & (df["Quantity"] > 0)
        & (df["Price"] > 0)
    )
    return df.loc[mask].copy()


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
    return agg[["recency_days", "frequency", "monetary_total", "monetary_mean", "monetary_std"]]


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
            return pd.Series(
                {
                    "days_since_first_purchase": tenure,
                    "avg_days_between_orders": float(tenure),
                    "std_days_between_orders": 0.0,
                    "is_one_time_buyer": 1,
                }
            )
        gaps = np.diff(dates).astype("timedelta64[D]").astype(float)
        return pd.Series(
            {
                "days_since_first_purchase": tenure,
                "avg_days_between_orders": float(np.mean(gaps)),
                "std_days_between_orders": float(np.std(gaps, ddof=1)) if len(gaps) > 1 else 0.0,
                "is_one_time_buyer": 0,
            }
        )

    return inv_dates.groupby("Customer ID").apply(_gap_stats, include_groups=False)


def _basket_shape_features(df: pd.DataFrame) -> pd.DataFrame:
    """Basket-shape features (5 features).

    - ``avg_basket_size``: average number of line items per invoice.
    - ``avg_basket_value``: average invoice total.
    - ``max_basket_value``: largest single invoice total.
    - ``avg_unique_items_per_basket``: average distinct SKUs per invoice.
    - ``single_item_basket_ratio``: fraction of invoices with exactly 1 line item.
    """
    inv = (
        df.groupby(["Customer ID", "Invoice"])
        .agg(
            n_lines=("Quantity", "size"),
            total=("LineTotal", "sum"),
            n_unique=("StockCode", "nunique"),
        )
        .reset_index()
    )

    agg = inv.groupby("Customer ID").agg(
        avg_basket_size=("n_lines", "mean"),
        avg_basket_value=("total", "mean"),
        max_basket_value=("total", "max"),
        avg_unique_items_per_basket=("n_unique", "mean"),
    )
    singles = (
        inv.groupby("Customer ID")["n_lines"]
        .apply(lambda s: (s == 1).mean())
        .rename("single_item_basket_ratio")
    )
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

    cat_spend = tmp.groupby(["Customer ID", "category"])["LineTotal"].sum().reset_index()

    def _cat_stats(g: pd.DataFrame) -> pd.Series:
        total = g["LineTotal"].sum()
        share = g["LineTotal"] / total if total > 0 else g["LineTotal"]
        top_share = share.max()
        n_cats = len(g)
        # Shannon entropy: -Σ p·log(p)
        probs = share.values
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log2(probs))) if len(probs) > 1 else 0.0
        return pd.Series(
            {
                "top_category_share": top_share,
                "n_unique_categories": n_cats,
                "category_entropy": entropy,
            }
        )

    return cat_spend.groupby("Customer ID").apply(_cat_stats, include_groups=False)


def _behavioural_features(obs_df: pd.DataFrame, event_history: pd.DataFrame) -> pd.DataFrame:
    """Behavioural features (5 features).

    Both inputs must be strictly before the snapshot cutoff.

    - ``has_ever_cancelled``: 1 if the customer has any cancelled invoice.
    - ``cancellation_rate``: fraction of invoices that are cancellations.
    - ``return_rate``: returned units / all absolute event units (bounded 0–1).
    - ``weekend_purchase_ratio``: fraction of orders placed on Sat/Sun.
    - ``morning_purchase_ratio``: fraction of orders placed before noon.
    """
    # Cancellations — detect from the original Invoice column before cleaning
    inv = event_history.groupby("Customer ID")["Invoice"].apply(lambda s: s.unique())
    has_cancel = inv.apply(lambda arr: int(any(str(x).startswith("C") for x in arr))).rename(
        "has_ever_cancelled"
    )
    cancel_rate = inv.apply(
        lambda arr: (
            sum(1 for x in arr if str(x).startswith("C")) / len(arr) if len(arr) > 0 else 0.0
        )
    ).rename("cancellation_rate")

    cust_qty = (
        event_history.assign(units=event_history["Quantity"].abs())
        .groupby("Customer ID")["units"]
        .sum()
    )
    neg_qty = -event_history[event_history["Quantity"] < 0].groupby("Customer ID")["Quantity"].sum()
    return_rate = (neg_qty / cust_qty).fillna(0.0).rename("return_rate")

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


def _geography_features(df: pd.DataFrame, country_map: dict[str, int]) -> pd.DataFrame:
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
    n_countries = df.groupby("Customer ID")["Country"].nunique().rename("n_countries")

    result = pd.DataFrame({"primary_country": country_mode, "n_countries": n_countries})
    result["is_domestic"] = (result["primary_country"] == "United Kingdom").astype(int)

    # Label-encode country
    result["country_code"] = result["primary_country"].map(country_map).fillna(-1)
    return result[["country_code", "is_domestic", "n_countries"]]


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------
def build_features(
    obs_df: pd.DataFrame,
    full_df: pd.DataFrame,
    cutoff: str,
    country_map: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Compute all feature groups and concatenate them.

    Parameters
    ----------
    obs_df : DataFrame
        Transactions in the observation window (already cleaned).
    full_df : DataFrame
        Event ledger; sliced strictly before cutoff inside this function.
    cutoff : str
        ISO date string for the cutoff boundary.

    Returns
    -------
    DataFrame
        One row per customer, columns are ``FEATURE_NAMES``.
    """
    cutoff_ts = pd.Timestamp(cutoff)
    obs_df = purchases(obs_df[obs_df["InvoiceDate"] < cutoff_ts])
    event_history = full_df[full_df["InvoiceDate"] < cutoff_ts].copy()
    if country_map is None:
        country_map = {c: i for i, c in enumerate(sorted(obs_df["Country"].unique()))}
    print("[features] Building RFM …")
    rfm = _rfm_features(obs_df, cutoff_ts)
    print("[features] Building cadence …")
    cadence = _cadence_features(obs_df, cutoff_ts)
    print("[features] Building basket shape …")
    basket = _basket_shape_features(obs_df)
    print("[features] Building category mix …")
    cat_mix = _category_mix_features(obs_df)
    print("[features] Building behavioural …")
    behav = _behavioural_features(obs_df, event_history)
    print("[features] Building geography …")
    geo = _geography_features(obs_df, country_map)

    X = rfm.join(cadence).join(basket).join(cat_mix).join(behav).join(geo)

    # Ensure all expected features are present
    missing_cols = set(FEATURE_NAMES) - set(X.columns)
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")

    X = X[FEATURE_NAMES]

    # No learned imputation here. The training Pipeline owns medians/scaling.
    X = X.replace([np.inf, -np.inf], np.nan).astype(float)

    print(f"[features] Built {len(X.columns)} features for {len(X):,} customers")
    return X


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save_features(
    X: pd.DataFrame, y: pd.Series, cfg: dict, snapshots: pd.DataFrame, country_map: dict[str, int]
) -> Path:
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
    snapshots.to_parquet(out_dir / "snapshots.parquet")
    # Latest historical snapshot is a demo feature store, not live customer data.
    latest = snapshots["cutoff"].max()
    latest_x = X.loc[snapshots["cutoff"].eq(latest)].copy()
    latest_x.index = snapshots.loc[latest_x.index, "customer_id"].astype(str)
    latest_x.to_parquet(out_dir / "customer_features.parquet")

    # Also save feature metadata
    meta = {
        "feature_names": FEATURE_NAMES,
        "contract_version": CONTRACT_VERSION,
        "country_map": country_map,
        "unknown_country_code": -1,
        "n_features": len(FEATURE_NAMES),
        "n_customers": len(X),
        "positive_rate": float(y.mean()),
        "cutoff_date": cfg["labelling"]["cutoff_date"],
        "label_window_days": cfg["labelling"]["label_window_days"],
        "snapshots": cfg["labelling"]["snapshots"],
        "source_data_sha256": _sha256(PROJECT_ROOT / cfg["data"]["parquet_path"]),
        "snapshot_rows": snapshots.groupby("split").size().to_dict(),
        "missing_counts": X.isna().sum().to_dict(),
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

    configured_features = [f for group in cfg["features"]["groups"].values() for f in group]
    if configured_features != FEATURE_NAMES:
        raise ValueError("pipeline.yaml feature groups differ from canonical contract")
    cutoffs = cfg["labelling"]["snapshots"]
    first = min(cutoffs["train"])
    country_map = {
        c: i
        for i, c in enumerate(
            sorted(full_df.loc[full_df["InvoiceDate"] < first, "Country"].unique())
        )
    }
    frames, targets, rows = [], [], []
    window = cfg["labelling"]["label_window_days"]
    for split, dates in cutoffs.items():
        for cutoff in dates:
            end = pd.Timestamp(cutoff) + pd.Timedelta(days=window)
            if end > full_df["InvoiceDate"].max():
                raise ValueError(
                    f"Incomplete label window for {cutoff}: requires data through {end}"
                )
            obs_df, _, y = create_labels(full_df, cutoff, window)
            X = build_features(obs_df, full_df, cutoff, country_map)
            y = y.reindex(X.index)
            snapshot = pd.DataFrame(
                {
                    "customer_id": X.index,
                    "cutoff": cutoff,
                    "label_end": str(end.date()),
                    "split": split,
                }
            )
            frames.append(X.reset_index(drop=True))
            targets.append(y.reset_index(drop=True))
            rows.append(snapshot)
    X = pd.concat(frames, ignore_index=True)
    y = pd.concat(targets, ignore_index=True)
    snapshots = pd.concat(rows, ignore_index=True)
    save_features(X, y, cfg, snapshots, country_map)

    print("=" * 70)
    print("[features] Done.")
    return X, y


def main() -> None:
    """CLI wrapper."""
    run()


if __name__ == "__main__":
    main()
