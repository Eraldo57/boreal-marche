"""
Fairness-audit logic on small SYNTHETIC arrays, plus one integration check on the real
model. The synthetic scores are hand-written to produce a known recall gap; they are not
workbook rows.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import fairness
from src.contracts import METADATA_PATH, PROCESSED


def _synthetic_audit_inputs():
    """Group A: perfect recall. B: recall 0.2 (100 returners). C: recall 1/3 (3 returners)."""
    y = np.r_[np.ones(100), np.zeros(100), np.ones(100), np.zeros(100), np.ones(3), np.zeros(3)]
    p = np.r_[
        np.full(100, 0.9),
        np.full(100, 0.1),
        np.full(20, 0.9),
        np.full(80, 0.1),
        np.full(100, 0.1),
        [0.9, 0.1, 0.1],
        np.full(3, 0.1),
    ]
    region = ["A"] * 200 + ["B"] * 200 + ["C"] * 6
    groups = pd.DataFrame({dimension: "all" for dimension in fairness.DIMENSIONS}, index=range(406))
    groups["region"] = region
    groups["purchase_history"] = ["repeat"] * 200 + ["one-time"] * 206
    return y, p, groups


def _rows(report, dimension):
    return {row["group"]: row for row in report["groups"][dimension]}


def test_region_mapping_uses_workbook_country_names():
    assert fairness.region_of("United Kingdom") == "United Kingdom"
    assert fairness.region_of("EIRE") == "Other Europe"
    assert fairness.region_of("Japan") == "Rest of world"
    assert fairness.region_of(None) == fairness.region_of("Unspecified") == "Unknown"


@pytest.mark.parametrize(
    ("days", "band"),
    [
        (0, "under 6 months"),
        (182, "under 6 months"),
        (183, "6-12 months"),
        (365, "12-18 months"),
        (548, "18 months or more"),
    ],
)
def test_tenure_band_edges(days, band):
    assert fairness.tenure_band(days) == band


def test_wilson_interval_brackets_the_proportion():
    low, high = fairness.wilson(8, 10)
    assert 0 <= low < 0.8 < high <= 1
    assert fairness.wilson(0, 0) is None


def test_audit_flags_large_gap_as_strong_and_small_group_as_inconclusive():
    y, p, groups = _synthetic_audit_inputs()
    report = fairness.audit(y, p, threshold=0.5, groups=groups)
    rows = _rows(report, "region")
    assert report["overall"]["recall"] == pytest.approx(121 / 203)
    assert not rows["A"]["flagged"]
    assert rows["B"]["flagged"] and rows["B"]["evidence"] == "strong"
    assert rows["C"]["flagged"] and rows["C"]["evidence"] == "inconclusive (small group)"
    assert rows["B"]["selection_rate"] == pytest.approx(20 / 200)


def test_threshold_for_recall_returns_the_highest_sufficient_threshold():
    y = np.array([1, 1, 1, 1, 0])
    p = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    assert fairness.threshold_for_recall(y, p, 0.75) == pytest.approx(0.3)
    assert fairness.threshold_for_recall(np.zeros(3), p[:3], 0.5) is None


def test_mitigation_estimate_closes_the_flagged_gap():
    y, p, groups = _synthetic_audit_inputs()
    flagged = [{"dimension": "region", "group": "B"}]
    (estimate,) = fairness.mitigation_estimates(y, p, 0.5, groups, flagged)
    floor = 121 / 203 - fairness.MAX_RECALL_GAP
    assert estimate["threshold_reaching_overall_minus_5pp"] < 0.5
    assert estimate["recall_after"] >= floor > estimate["recall_before"]
    assert estimate["selection_rate_after"] > estimate["selection_rate_before"]


def test_overlap_counts_rows_and_one_time_share():
    _, _, groups = _synthetic_audit_inputs()
    overlap = fairness.flag_overlap(groups, [{"dimension": "region", "group": "B"}])
    assert overlap["rows_in_any_flagged_group"] == 200
    assert overlap["one_time_buyer_share"]["region=B"] == 1.0


def test_markdown_tables_list_every_group_with_its_flag():
    y, p, groups = _synthetic_audit_inputs()
    report = fairness.audit(y, p, 0.5, groups)
    report.update(test_cutoff="2020-01-01", mitigation_estimates=[])
    report["overlap"] = fairness.flag_overlap(groups, [{"dimension": "region", "group": "B"}])
    tables = fairness.markdown_tables(report)
    assert "| A | 200 |" in tables and "| B | 200 |" in tables and "| C | 6 |" in tables
    assert "| strong |" in tables and "| inconclusive (small group) |" in tables
    assert "200 of 406 rows" in tables


@pytest.mark.integration
@pytest.mark.skipif(
    not METADATA_PATH.exists() or not (PROCESSED / "X.parquet").exists(),
    reason="Run the real local pipeline to generate the model and features",
)
def test_real_audit_matches_saved_predictions_and_committed_report():
    report = fairness.run()  # raises if recomputed scores differ from test_predictions
    committed = json.loads((fairness.REPORTS / "fairness_audit.json").read_text())
    assert report["model_sha256"] == committed["model_sha256"], "re-run python -m src.fairness"
    assert report["flagged_groups"] == committed["flagged_groups"]
    assert report["overall"]["recall"] == pytest.approx(committed["overall"]["recall"])
    for dimension in fairness.DIMENSIONS:
        assert sum(row["n"] for row in report["groups"][dimension]) == report["overall"]["n"]
