"""
Boréal Marché — Association Rule Mining (Head B)
=================================================

Builds a sparse transaction matrix from clean invoices, mines association
rules with Apriori (demonstration) and FP-Growth (production), sweeps
support thresholds, filters by lift, and exports a static rule artifact.

Key design decisions
--------------------
* **Mining is offline; serving is online.** This module runs once (or on a
  schedule).  The API loads the exported JSON — it never re-runs the
  mining algorithm on every request.
* The exported ``artifacts/rules.json`` contains a header with the mining
  date, parameters, source-data hash, and rule count so the serving layer
  can detect staleness.

Usage::

    python -m src.rules
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Transaction matrix
# ---------------------------------------------------------------------------
def build_transaction_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Build a one-hot transaction matrix (invoices × items).

    Parameters
    ----------
    df : DataFrame
        Cleaned transactions with at least ``Invoice`` and ``Description``.

    Returns
    -------
    basket_df : DataFrame
        Boolean DataFrame, one row per invoice, one column per item.
    memory_report : dict
        Dense vs. sparse memory footprint in MB.
    """
    # One-hot encode: each invoice is a row, each Description is a column
    basket = (
        df.groupby(["Invoice", "Description"])["Quantity"]
        .sum()
        .unstack()
        .fillna(0)
    )
    # Convert to boolean (purchased or not)
    basket_bool = (basket > 0)

    # Memory report
    dense_bytes = basket_bool.values.nbytes
    sp = sparse.csr_matrix(basket_bool.values)
    sparse_bytes = sp.data.nbytes + sp.indices.nbytes + sp.indptr.nbytes

    memory_report = {
        "n_invoices": len(basket_bool),
        "n_items": basket_bool.shape[1],
        "sparsity_pct": round(float(1 - sp.nnz / (sp.shape[0] * sp.shape[1])) * 100, 2),
        "dense_mb": round(dense_bytes / 1e6, 2),
        "sparse_mb": round(sparse_bytes / 1e6, 2),
        "compression_ratio": round(dense_bytes / sparse_bytes, 1) if sparse_bytes > 0 else 0,
    }
    print(f"[rules] Transaction matrix: {memory_report['n_invoices']:,} invoices × "
          f"{memory_report['n_items']:,} items")
    print(f"[rules] Sparsity: {memory_report['sparsity_pct']:.1f}%")
    print(f"[rules] Dense: {memory_report['dense_mb']:.1f} MB  "
          f"Sparse: {memory_report['sparse_mb']:.1f} MB  "
          f"(compression {memory_report['compression_ratio']:.0f}×)")

    return basket_bool, memory_report


# ---------------------------------------------------------------------------
# Mining
# ---------------------------------------------------------------------------
def mine_rules_apriori(
    basket_df: pd.DataFrame,
    min_support: float,
    min_confidence: float,
    min_lift: float,
) -> tuple[pd.DataFrame, float]:
    """Mine with Apriori (demonstration / timing comparison).

    Returns (rules_df, elapsed_seconds).
    """
    t0 = time.time()
    freq = apriori(basket_df, min_support=min_support, use_colnames=True)
    if len(freq) == 0:
        return pd.DataFrame(), time.time() - t0
    rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)
    rules = rules[rules["lift"] >= min_lift]
    elapsed = time.time() - t0
    return rules, elapsed


def mine_rules_fpgrowth(
    basket_df: pd.DataFrame,
    min_support: float,
    min_confidence: float,
    min_lift: float,
) -> tuple[pd.DataFrame, float]:
    """Mine with FP-Growth (production).

    Returns (rules_df, elapsed_seconds).
    """
    t0 = time.time()
    freq = fpgrowth(basket_df, min_support=min_support, use_colnames=True)
    if len(freq) == 0:
        return pd.DataFrame(), time.time() - t0
    rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)
    rules = rules[rules["lift"] >= min_lift]
    elapsed = time.time() - t0
    return rules, elapsed


def sweep_support(
    basket_df: pd.DataFrame,
    support_values: list[float],
    min_confidence: float,
    min_lift: float,
) -> pd.DataFrame:
    """Sweep min_support and report rule counts and timings.

    Runs FP-Growth at each threshold and Apriori at the chosen production
    threshold for timing comparison.
    """
    rows = []
    for sup in support_values:
        rules, elapsed = mine_rules_fpgrowth(basket_df, sup, min_confidence, min_lift)
        rows.append({
            "min_support": sup,
            "n_frequent_itemsets": "-",
            "n_rules": len(rules),
            "fpgrowth_time_s": round(elapsed, 2),
        })
        print(f"[rules] support={sup:.3f}  → {len(rules):,} rules  ({elapsed:.1f}s)")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Lift filtering & concrete example
# ---------------------------------------------------------------------------
def filter_and_document_lift(
    rules: pd.DataFrame,
    min_lift: float,
) -> tuple[pd.DataFrame, list[dict]]:
    """Apply lift filter and find a concrete example of a discarded rule.

    Returns
    -------
    filtered : DataFrame
        Rules that survive the lift filter.
    discarded_examples : list[dict]
        At least one example of a high-confidence, low-lift rule that was
        rightfully removed — with an explanation of why.
    """
    # Find high-confidence rules that would be BELOW the lift threshold
    # (for documentation purposes, temporarily include them)
    low_lift = rules[rules["lift"] < min_lift].copy()
    high_conf_low_lift = low_lift.sort_values("confidence", ascending=False)

    discarded_examples: list[dict] = []
    if len(high_conf_low_lift) > 0:
        example = high_conf_low_lift.iloc[0]
        discarded_examples.append({
            "antecedent": sorted(example["antecedents"]),
            "consequent": sorted(example["consequents"]),
            "confidence": round(float(example["confidence"]), 4),
            "lift": round(float(example["lift"]), 4),
            "support": round(float(example["support"]), 6),
            "reason_discarded": (
                f"Lift = {float(example['lift']):.4f} (< {min_lift}).  "
                f"Despite {float(example['confidence']):.1%} confidence, the consequent "
                f"'{sorted(example['consequents'])}' is so popular on its own that "
                f"co-occurrence with '{sorted(example['antecedents'])}' adds no "
                f"predictive value — the customer would likely buy it anyway."
            ),
        })

    filtered = rules[rules["lift"] >= min_lift].copy()
    return filtered, discarded_examples


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def _data_hash(df: pd.DataFrame) -> str:
    """Quick content hash for provenance tracking."""
    h = hashlib.sha256()
    for col in sorted(df.columns):
        h.update(df[col].astype(str).str.cat().encode())
    return h.hexdigest()[:16]


def export_rules(
    rules: pd.DataFrame,
    cfg: dict,
    memory_report: dict,
    sweep_df: pd.DataFrame,
    discarded_examples: list[dict],
    apriori_time: float,
    fpgrowth_time: float,
    data_hash: str,
) -> Path:
    """Export rules to ``artifacts/rules.json``.

    The file contains a header and the rule list, each rule being a dict
    with antecedent (list of item descriptions), consequent, confidence,
    lift, and support.
    """
    out_path = PROJECT_ROOT / "artifacts" / "rules.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rule_list = []
    for _, row in rules.iterrows():
        rule_list.append({
            "antecedent": sorted(row["antecedents"]),
            "consequent": sorted(row["consequents"]),
            "confidence": round(float(row["confidence"]), 6),
            "lift": round(float(row["lift"]), 4),
            "support": round(float(row["support"]), 6),
        })

    # Sort by lift descending for serving efficiency
    rule_list.sort(key=lambda r: -r["lift"])

    # Cap at max_rules
    max_rules = cfg["rules"]["max_rules"]
    if len(rule_list) > max_rules:
        print(f"[rules] Capping from {len(rule_list)} to {max_rules} rules")
        rule_list = rule_list[:max_rules]

    artifact = {
        "header": {
            "mining_date": datetime.now(timezone.utc).isoformat(),
            "algorithm_demo": cfg["rules"]["algorithm_demo"],
            "algorithm_prod": cfg["rules"]["algorithm_prod"],
            "min_support": cfg["rules"]["min_support_sweep"],
            "min_confidence": cfg["rules"]["min_confidence"],
            "min_lift": cfg["rules"]["min_lift"],
            "source_data_hash": data_hash,
            "rule_count": len(rule_list),
            "memory_report": memory_report,
            "timing": {
                "apriori_seconds": round(apriori_time, 2),
                "fpgrowth_seconds": round(fpgrowth_time, 2),
            },
            "discarded_rule_examples": discarded_examples,
        },
        "rules": rule_list,
    }

    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)

    print(f"[rules] Exported {len(rule_list)} rules to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Popularity fallback
# ---------------------------------------------------------------------------
def compute_popularity(df: pd.DataFrame, top_n: int = 20) -> list[str]:
    """Return the top-N most frequently purchased items for fallback."""
    counts = df.groupby("Description")["Invoice"].nunique().sort_values(ascending=False)
    return counts.head(top_n).index.tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run() -> Path:
    """Execute the full rule-mining pipeline."""
    cfg = load_config()

    print("=" * 70)
    print("Boréal Marché — Association Rule Mining (Head B)")
    print("=" * 70)

    # Load clean transactions
    parquet_path = PROJECT_ROOT / cfg["data"]["parquet_path"]
    if not parquet_path.exists():
        raise FileNotFoundError(f"{parquet_path} not found. Run `python -m src.ingest` first.")

    df = pd.read_parquet(parquet_path)
    data_hash = _data_hash(df[["Invoice", "Description"]])

    # Build transaction matrix
    print("\n[rules] === Transaction Matrix ===")
    basket_df, mem_report = build_transaction_matrix(df)

    # Support sweep
    print("\n[rules] === Support Sweep (FP-Growth) ===")
    rules_cfg = cfg["rules"]
    sweep_df = sweep_support(
        basket_df,
        rules_cfg["min_support_sweep"],
        rules_cfg["min_confidence"],
        rules_cfg["min_lift"],
    )
    print(sweep_df.to_string(index=False))

    # Pick the production support level (first one that yields ≤ max_rules)
    chosen_support = rules_cfg["min_support_sweep"][1]  # default to second value
    for sup in rules_cfg["min_support_sweep"]:
        test_rules, _ = mine_rules_fpgrowth(basket_df, sup, rules_cfg["min_confidence"], rules_cfg["min_lift"])
        if len(test_rules) <= rules_cfg["max_rules"]:
            chosen_support = sup
            break

    print(f"\n[rules] Production support level: {chosen_support}")

    # Apriori timing comparison
    print("\n[rules] === Apriori (demonstration) ===")
    apriori_rules, apriori_time = mine_rules_apriori(
        basket_df, chosen_support, rules_cfg["min_confidence"], rules_cfg["min_lift"],
    )
    print(f"[rules] Apriori: {len(apriori_rules)} rules in {apriori_time:.2f}s")

    # FP-Growth production run
    print("\n[rules] === FP-Growth (production) ===")
    prod_rules, fpgrowth_time = mine_rules_fpgrowth(
        basket_df, chosen_support, rules_cfg["min_confidence"], rules_cfg["min_lift"],
    )
    print(f"[rules] FP-Growth: {len(prod_rules)} rules in {fpgrowth_time:.2f}s")

    # Lift filtering
    print("\n[rules] === Lift Filtering ===")
    # For the discarded-rule example, temporarily mine with lift >= 0
    all_rules, _ = mine_rules_fpgrowth(basket_df, chosen_support, rules_cfg["min_confidence"], 0.0)
    filtered, discarded = filter_and_document_lift(all_rules, rules_cfg["min_lift"])
    print(f"[rules] After lift filter (≥ {rules_cfg['min_lift']}): {len(filtered)} rules")
    if discarded:
        ex = discarded[0]
        print(f"[rules] Example discarded rule:")
        print(f"        {ex['antecedent']} → {ex['consequent']}")
        print(f"        confidence={ex['confidence']:.2%}  lift={ex['lift']:.4f}")
        print(f"        Reason: {ex['reason_discarded']}")

    # Compute popularity fallback
    popularity = compute_popularity(df, top_n=20)

    # Export
    print("\n[rules] === Export ===")
    out_path = export_rules(
        prod_rules, cfg, mem_report, sweep_df, discarded,
        apriori_time, fpgrowth_time, data_hash,
    )

    # Also save popularity for the recommendation engine
    pop_path = PROJECT_ROOT / "artifacts" / "popularity.json"
    with open(pop_path, "w") as f:
        json.dump({"items": popularity}, f, indent=2, ensure_ascii=False)
    print(f"[rules] Popularity fallback saved to {pop_path}")

    print("=" * 70)
    print("[rules] Done.")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Boréal Marché rule mining")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
