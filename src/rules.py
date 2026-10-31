"""Offline sparse association mining, chronological evaluation, JSON serving artifacts."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

from src.contracts import ARTIFACTS, PROJECT_ROOT
from src.features import purchases
from src.ingest import _sha256, load_config
from src.recommend import RecommendationEngine, _Rule


def baskets(df: pd.DataFrame) -> list[list[str]]:
    """Use canonical descriptions, explicitly accepting description/SKU ambiguity."""
    grouped: list[list[str]] = (
        df.groupby("Invoice", sort=True)["Description"].agg(lambda s: sorted(set(s))).tolist()
    )
    return grouped


def build_transaction_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Encode directly to CSR; never allocate the full dense invoice/item matrix."""
    rows = baskets(df)
    if not rows:
        raise ValueError("No purchase baskets to mine")
    encoder = TransactionEncoder()
    matrix = encoder.fit(rows).transform(rows, sparse=True).tocsr()
    dense_bytes = matrix.shape[0] * matrix.shape[1] * np.dtype(bool).itemsize
    sparse_bytes = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    report = {
        "n_invoices": matrix.shape[0],
        "n_items": matrix.shape[1],
        "dense_equivalent_bytes": dense_bytes,
        "csr_bytes": sparse_bytes,
        "sparsity_pct": 100 * (1 - matrix.nnz / max(1, dense_bytes)),
        "dense_mb": dense_bytes / 1e6,
        "sparse_mb": sparse_bytes / 1e6,
    }
    return pd.DataFrame.sparse.from_spmatrix(matrix, columns=encoder.columns_), report


def mine(
    basket_df: pd.DataFrame,
    support: float,
    confidence: float,
    lift: float,
    algorithm: str = "fpgrowth",
    max_len: int = 3,
) -> tuple[pd.DataFrame, float, int]:
    start = time.perf_counter()
    miner = apriori if algorithm == "apriori" else fpgrowth
    kwargs = {"low_memory": True} if algorithm == "apriori" else {}
    frequent = miner(basket_df, min_support=support, use_colnames=True, max_len=max_len, **kwargs)
    columns = ["antecedents", "consequents", "confidence", "lift", "support"]
    if frequent.empty or not frequent["itemsets"].map(len).gt(1).any():
        rules = pd.DataFrame(columns=columns)
    else:
        rules = association_rules(
            frequent, metric="confidence", min_threshold=confidence, num_itemsets=len(basket_df)
        )
        rules = rules[rules["lift"] >= lift]
    return rules, time.perf_counter() - start, len(frequent)


def mine_rules_apriori(
    basket_df: pd.DataFrame, min_support: float, min_confidence: float, min_lift: float
) -> tuple[pd.DataFrame, float]:
    rules, seconds, _ = mine(basket_df, min_support, min_confidence, min_lift, "apriori")
    return rules, seconds


def mine_rules_fpgrowth(
    basket_df: pd.DataFrame, min_support: float, min_confidence: float, min_lift: float
) -> tuple[pd.DataFrame, float]:
    rules, seconds, _ = mine(basket_df, min_support, min_confidence, min_lift)
    return rules, seconds


def rule_records(rules: pd.DataFrame, max_rules: int) -> list[dict]:
    records: list[dict[str, Any]] = [
        {
            "antecedent": sorted(r.antecedents),
            "consequent": sorted(r.consequents),
            "confidence": float(r.confidence),
            "lift": float(r.lift),
            "support": float(r.support),
        }
        for r in rules.itertuples()
    ]
    return sorted(
        records, key=lambda r: (-r["lift"], -r["confidence"], r["antecedent"], r["consequent"])
    )[:max_rules]


def compute_popularity(df: pd.DataFrame, top_n: int | None = None) -> list[str]:
    counts = (
        df.groupby("Description")["Invoice"].nunique().sort_values(ascending=False, kind="stable")
    )
    popular: list[str] = counts.head(top_n).index.tolist() if top_n else counts.index.tolist()
    return popular


def engine_from_records(
    records: list[dict[str, Any]], popularity: list[str]
) -> RecommendationEngine:
    return RecommendationEngine(
        [
            _Rule(
                frozenset(r["antecedent"]),
                frozenset(r["consequent"]),
                r["confidence"],
                r["lift"],
                r["support"],
            )
            for r in records
        ],
        popularity,
    )


def temporal_baskets(df: pd.DataFrame, train_end: str, validation_end: str, test_end: str) -> Any:
    """Assign entire invoices by date; reject invoices spanning a split boundary."""
    history = purchases(df)
    dates = history.groupby("Invoice")["InvoiceDate"].agg(["min", "max"])
    for boundary in [train_end, validation_end, test_end]:
        if ((dates["min"] < boundary) & (dates["max"] >= boundary)).any():
            raise ValueError("Invoice straddles a temporal boundary")
    return (
        history[history["InvoiceDate"] < train_end],
        history[(history["InvoiceDate"] >= train_end) & (history["InvoiceDate"] < validation_end)],
        history[(history["InvoiceDate"] >= validation_end) & (history["InvoiceDate"] < test_end)],
    )


def evaluate_baskets(
    engine: Any, heldout: Any, popularity: Any, seed: int = 42, k: int = 5
) -> dict[str, Any]:
    """Mask half each later basket; compare rule/fallback hybrid to popularity.

    This is basket completion, not next-order prediction. Single-item baskets
    cannot supply both context and hidden targets and are counted but excluded.
    """
    rng = np.random.default_rng(seed)
    hits, baseline_hits, evaluated, rule_served = 0, 0, 0, 0
    recommended, baseline_recommended = set(), set()
    all_baskets = baskets(heldout)
    for basket in all_baskets:
        if len(basket) < 2:
            continue
        shuffled = rng.permutation(basket).tolist()
        observed = shuffled[: max(1, len(basket) // 2)]
        hidden = set(shuffled[len(observed) :])
        result = engine.recommend(observed, k)
        recs = result["recommendations"]
        baseline = [item for item in popularity if item not in observed][:k]
        hits += bool(set(recs) & hidden)
        baseline_hits += bool(set(baseline) & hidden)
        recommended.update(recs)
        baseline_recommended.update(baseline)
        rule_served += result["strategy"] == "rule_based"
        evaluated += 1
    return {
        "baskets_total": len(all_baskets),
        "baskets_evaluated": evaluated,
        "single_item_excluded": len(all_baskets) - evaluated,
        "hit_rate_at_5": hits / evaluated if evaluated else None,
        "popularity_hit_rate_at_5": baseline_hits / evaluated if evaluated else None,
        "catalogue_coverage": len(recommended) / len(popularity) if popularity else None,
        "popularity_catalogue_coverage": len(baseline_recommended) / len(popularity)
        if popularity
        else None,
        "rule_firing_coverage": rule_served / evaluated if evaluated else None,
        "coverage_denominator": "distinct training descriptions",
        "k": k,
        "mask_seed": seed,
    }


def run() -> Path:
    cfg = load_config()
    rcfg = cfg["rules"]
    path = PROJECT_ROOT / cfg["data"]["parquet_path"]
    df = pd.read_parquet(path)
    train, validation, test = temporal_baskets(
        df, rcfg["train_end"], rcfg["validation_end"], rcfg["test_end"]
    )
    matrix, memory = build_transaction_matrix(train)
    popularity = compute_popularity(train)
    sweep, candidates = [], {}
    for support in rcfg["min_support_sweep"]:
        all_rules, seconds, n_itemsets = mine(
            matrix,
            support,
            min(rcfg["min_confidence_sweep"]),
            rcfg["min_lift"],
            max_len=rcfg["max_itemset_length"],
        )
        for confidence in rcfg["min_confidence_sweep"]:
            filtered = all_rules[all_rules["confidence"] >= confidence]
            records = rule_records(filtered, rcfg["max_rules"])
            metrics = evaluate_baskets(
                engine_from_records(records, popularity), validation, popularity
            )
            sweep.append(
                {
                    "support": support,
                    "confidence": confidence,
                    "eligible_rules": len(filtered),
                    "exported_rules": len(records),
                    "frequent_itemsets": n_itemsets,
                    "shared_mining_seconds": seconds,
                    **metrics,
                }
            )
            candidates[(support, confidence)] = records
        print(
            f"[rules] support={support} itemsets={n_itemsets} eligible_rules={len(all_rules)}",
            flush=True,
        )
    # Select on validation only; never mine validation or final-test baskets.
    selected = max(
        sweep, key=lambda r: (r["hit_rate_at_5"], r["catalogue_coverage"], -r["exported_rules"])
    )
    support, confidence = selected["support"], selected["confidence"]
    records = candidates[(support, confidence)]
    ap, ap_time, _ = mine(
        matrix, support, confidence, rcfg["min_lift"], "apriori", rcfg["max_itemset_length"]
    )
    fp, fp_time, _ = mine(
        matrix, support, confidence, rcfg["min_lift"], "fpgrowth", rcfg["max_itemset_length"]
    )
    ap_keys = {(tuple(r["antecedent"]), tuple(r["consequent"])) for r in rule_records(ap, len(ap))}
    fp_keys = {(tuple(r["antecedent"]), tuple(r["consequent"])) for r in rule_records(fp, len(fp))}
    final = evaluate_baskets(engine_from_records(records, popularity), test, popularity)
    raw, _, _ = mine(matrix, support, confidence, 0, max_len=rcfg["max_itemset_length"])
    discarded = rule_records(raw[raw["lift"] < rcfg["min_lift"]], 1)
    header = {
        "mining_date": datetime.now(UTC).isoformat(),
        "algorithm_prod": "fpgrowth",
        "algorithm_demo": "apriori",
        "min_support": support,
        "min_confidence": confidence,
        "min_lift": rcfg["min_lift"],
        "max_itemset_length": rcfg["max_itemset_length"],
        "product_identity": "canonical_description",
        "source_data_hash": _sha256(path),
        "train_end_exclusive": rcfg["train_end"],
        "rule_count": len(records),
        "memory_report": memory,
        "timing": {"apriori_seconds": ap_time, "fpgrowth_seconds": fp_time},
        "apriori_fpgrowth_rule_keys_equal": ap_keys == fp_keys,
        "discarded_rule_examples": discarded,
        "popularity_path": "artifacts/popularity.json",
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "rules.json").write_text(
        json.dumps({"header": header, "rules": records}, indent=2)
    )
    (ARTIFACTS / "popularity.json").write_text(
        json.dumps(
            {
                "items": popularity,
                "train_end_exclusive": rcfg["train_end"],
                "source_data_hash": header["source_data_hash"],
            },
            indent=2,
        )
    )
    reports = PROJECT_ROOT / "reports/current_run"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "recommendation_metrics.json").write_text(
        json.dumps(
            {
                "header": header,
                "sweep_validation": sweep,
                "selected_validation": selected,
                "final_test": final,
                "protocol": "train before June; validation June-August; untouched "
                "September-November baskets; deterministic half-basket masking",
            },
            indent=2,
        )
    )
    print(json.dumps({"rules": len(records), "final_test": final}, indent=2))
    return ARTIFACTS / "rules.json"


if __name__ == "__main__":
    run()
