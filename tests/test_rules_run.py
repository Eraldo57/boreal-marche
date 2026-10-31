"""
Rule-mining pipeline test on a SYNTHETIC ledger.

Items are placeholders (item_a, item_b, ...), not products from the workbook. The
ledger is generated so that item_a and item_b co-occur often, giving the miner a
real rule to find, without needing the DVC-tracked transactions.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import rules

ITEMS = [f"item_{letter}" for letter in "abcdefghij"]


def _ledger() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    records, invoice = [], 0

    def add_invoices(count: int, start: str, end: str) -> None:
        nonlocal invoice
        dates = pd.date_range(start, end, periods=count)
        for stamp in dates:
            invoice += 1
            basket = set(rng.choice(ITEMS[2:], size=rng.integers(1, 4), replace=False))
            if rng.random() < 0.4:
                basket |= {"item_a", "item_b"}
            for item in sorted(basket):
                records.append(
                    {
                        "Invoice": str(invoice),
                        "Description": item,
                        "Quantity": 1,
                        "Price": 2.0,
                        "InvoiceDate": stamp,
                    }
                )

    add_invoices(300, "2011-01-01", "2011-05-30")
    add_invoices(80, "2011-06-02", "2011-08-30")
    add_invoices(80, "2011-09-02", "2011-11-30")
    return pd.DataFrame(records)


@pytest.fixture
def rule_run(tmp_path, monkeypatch):
    parquet = tmp_path / "data" / "processed" / "transactions.parquet"
    parquet.parent.mkdir(parents=True)
    _ledger().to_parquet(parquet)
    monkeypatch.setattr(rules, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(rules, "ARTIFACTS", tmp_path / "artifacts")
    return tmp_path, rules.run()


def test_run_writes_rules_with_header(rule_run):
    root, path = rule_run
    payload = json.loads(path.read_text())
    header = payload["header"]
    assert header["rule_count"] == len(payload["rules"]) > 0
    assert header["algorithm_prod"] == "fpgrowth"
    assert header["apriori_fpgrowth_rule_keys_equal"] is True
    # A 10-item toy matrix is too small for CSR to beat dense; just check the report is filled.
    assert header["memory_report"]["n_invoices"] == 300


def test_run_finds_the_planted_association(rule_run):
    _, path = rule_run
    found = {
        (tuple(r["antecedent"]), tuple(r["consequent"]))
        for r in json.loads(path.read_text())["rules"]
    }
    assert (("item_a",), ("item_b",)) in found


def test_run_writes_popularity_and_metrics(rule_run):
    root, _ = rule_run
    popularity = json.loads((root / "artifacts" / "popularity.json").read_text())
    assert set(popularity["items"]) <= set(ITEMS)
    metrics = json.loads((root / "reports/current_run/recommendation_metrics.json").read_text())
    assert metrics["sweep_validation"] and metrics["final_test"]["k"] == 5


def test_mine_apriori_and_fpgrowth_agree():
    matrix, _ = rules.build_transaction_matrix(_ledger())
    apriori_rules, _ = rules.mine_rules_apriori(matrix, 0.05, 0.3, 1.2)
    fpgrowth_rules, _ = rules.mine_rules_fpgrowth(matrix, 0.05, 0.3, 1.2)
    assert len(apriori_rules) == len(fpgrowth_rules) > 0


def test_mine_handles_no_frequent_pairs():
    matrix, _ = rules.build_transaction_matrix(_ledger())
    empty, _, itemsets = rules.mine(matrix, 0.999, 0.1, 1.2)
    assert empty.empty and itemsets >= 0


def test_build_matrix_rejects_empty_ledger():
    empty = _ledger().iloc[0:0]
    with pytest.raises(ValueError, match="No purchase baskets"):
        rules.build_transaction_matrix(empty)


def test_straddling_invoice_is_rejected():
    frame = _ledger()
    first = frame["Invoice"].iloc[0]
    mask = frame["Invoice"] == first
    frame.loc[mask, "InvoiceDate"] = [
        pd.Timestamp("2011-05-31") if i == 0 else pd.Timestamp("2011-06-02")
        for i in range(mask.sum())
    ]
    with pytest.raises(ValueError, match="straddles"):
        rules.temporal_baskets(frame, "2011-06-01", "2011-09-01", "2011-12-01")


def test_evaluate_baskets_with_no_multi_item_baskets():
    single = pd.DataFrame({"Invoice": ["1", "2"], "Description": ["item_a", "item_b"]})
    engine = rules.engine_from_records([], ["item_a"])
    result = rules.evaluate_baskets(engine, single, ["item_a"])
    assert result["baskets_evaluated"] == 0 and result["hit_rate_at_5"] is None
