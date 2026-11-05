"""The presentation must distinguish product diversity, basket coverage and latency."""

import json

import pytest
from streamlit.testing.v1 import AppTest

from frontend import basket


@pytest.fixture
def study(tmp_path, monkeypatch):
    header = {
        "timing": {"apriori_seconds": 1.0, "fpgrowth_seconds": 2.0},
        "apriori_fpgrowth_rule_keys_equal": True,
        "memory_report": {"n_items": 10},
    }
    rules = [
        {"antecedent": ["a", "b"], "consequent": ["c"]},
        {"antecedent": ["b"], "consequent": ["d"]},
    ]
    final = {
        "hit_rate_at_5": 0.4,
        "popularity_hit_rate_at_5": 0.2,
        "rule_firing_coverage": 0.75,
        "catalogue_coverage": 0.3,
        "baskets_evaluated": 20,
        "single_item_excluded": 2,
    }
    report = {
        "header": header,
        "final_test": final,
        "sweep_validation": [
            {
                "support": 0.1,
                "confidence": 0.2,
                "exported_rules": 2,
                "hit_rate_at_5": 0.4,
                "catalogue_coverage": 0.3,
            }
        ],
    }
    report_path = tmp_path / "metrics.json"
    report_path.write_text(json.dumps(report))
    benchmark = {
        "executed_at": "2026-01-01",
        "queries": 20,
        "query_source": "synthetic test baskets",
        "results": [
            {
                "rules": 2,
                "rules_are": "real export",
                "recommend_ms_p50": 0.1,
                "recommend_ms_p95": 0.2,
                "recommend_ms_max": 0.3,
            }
        ],
    }
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(json.dumps(benchmark))
    monkeypatch.setattr(basket, "REPORT", report_path)
    monkeypatch.setattr(basket, "BENCHMARK", benchmark_path)
    source = (
        f"from frontend.basket import _show_conclusions\n_show_conclusions({header!r}, {rules!r})"
    )
    return source, benchmark_path, benchmark


def test_conclusions_compute_separate_product_and_basket_measures(study):
    source, _, _ = study
    app = AppTest.from_string(source).run()
    assert not app.exception
    text = " ".join(item.value for item in app.markdown)
    assert "2.00×" in text
    assert "25.00%" in text and "fallback" in text
    assert "10.00 percentage points below" in text
    product_table = next(t.value for t in app.table if "Product group" in t.value.columns)
    assert product_table["Distinct products"].tolist() == [2, 2, 3]
    assert any("HTTP, network and Streamlit" in c.value for c in app.caption)
    assert any("original rubric" in c.value for c in app.caption)


def test_synthetic_timing_is_not_presented_as_real_export_latency(study):
    source, path, benchmark = study
    benchmark["results"][0]["rules_are"] = "synthetic"
    path.write_text(json.dumps(benchmark))
    app = AppTest.from_string(source).run()
    assert not app.exception
    assert any("No saved real-export" in item.value for item in app.info)
    assert not any("Milliseconds" in t.value.columns for t in app.table)


def test_missing_benchmark_does_not_hide_evaluation_results(study):
    source, path, _ = study
    path.write_text("invalid JSON")
    app = AppTest.from_string(source).run()
    assert not app.exception
    assert any("No saved real-export" in item.value for item in app.info)
    assert any("Product group" in t.value.columns for t in app.table)
