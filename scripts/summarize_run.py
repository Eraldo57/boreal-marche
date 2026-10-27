"""Regenerate an evidence-only report and environment/artifact inventory."""

import json
import subprocess
import sys
from datetime import UTC, datetime

from src.contracts import PROJECT_ROOT
from src.ingest import _sha256


def main():
    root = PROJECT_ROOT
    reports = root / "reports/current_run"

    def read(path):
        return json.loads((root / path).read_text())

    manifest = read("data/processed/manifest.json")
    model = read("reports/current_run/retention_metrics.json")
    rules = read("reports/current_run/recommendation_metrics.json")
    assistant = read("reports/current_run/assistant_regex_metrics.json")
    api = read("reports/current_run/api_smoke.json")
    ui = read("reports/current_run/ui_smoke.json")
    index = read("data/chromadb/manifest.json")
    test = model["final_test"]
    rec = rules["final_test"]
    lines = [
        "# Executed validation results",
        "",
        f"Report generated {datetime.now(UTC).isoformat()}. Sources below are executed outputs.",
        "",
        "## Data and Head A",
        "",
        f"- Raw rows: {manifest['original_row_count']:,}; cleaned event rows: "
        f"{manifest['row_count']:,}; positive purchase rows: {manifest['purchase_rows']:,}.",
        f"- Model: {model['selected_model']}; calibration: {model['selected_calibration']}.",
        f"- Final test rows: {model['splits']['test']['rows']:,}; "
        f"AP: {test['pr_auc']:.6f}; ROC-AUC: {test['roc_auc']:.6f}; "
        f"Brier: {test['brier_score']:.6f}.",
        f"- Recency baseline AP: "
        f"{model['final_test_baselines']['business_heuristic_recency']['pr_auc']:.6f}; "
        f"dummy AP: {model['final_test_baselines']['DummyClassifier_stratified']['pr_auc']:.6f}.",
        "- Six dated snapshots; feature history strictly before cutoff; positive target "
        "in [cutoff, cutoff + 90 days). Repeat customers are allowed across periods.",
        "- Models/tuning use December validation; calibrators fit March; calibration choice "
        "and utility threshold use June; final September holdout is scored after decisions lock.",
        "- Monetary utility assumes cost 5 and margin 25; not observed profit or causal uplift.",
        "- Retention ran twice successfully after an initial dependency failure. Final rerun "
        "fixed timing/report bookkeeping, not model selection based on test results. "
        "Identical held-out AP does not constitute independent validation.",
        "",
        "Sources: data/processed/manifest.json, cleaning_report.json, "
        "retention_metrics.json, artifacts/retention_model_metadata.json.",
        "",
        "## Head B",
        "",
        f"- {rules['header']['rule_count']:,} selected rules; support "
        f"{rules['header']['min_support']}, confidence {rules['header']['min_confidence']}, "
        f"lift {rules['header']['min_lift']}.",
        f"- Evaluated later baskets: {rec['baskets_evaluated']:,}; excluded single-item "
        f"baskets: {rec['single_item_excluded']:,}.",
        f"- Hit-rate@5: {rec['hit_rate_at_5']:.6f}; popularity: "
        f"{rec['popularity_hit_rate_at_5']:.6f}.",
        f"- Catalogue coverage: {rec['catalogue_coverage']:.6f}; rule-firing coverage: "
        f"{rec['rule_firing_coverage']:.6f}.",
        f"- CSR bytes: {rules['header']['memory_report']['csr_bytes']:,}; equivalent dense "
        f"boolean bytes: {rules['header']['memory_report']['dense_equivalent_bytes']:,} "
        "(dense size calculated, not allocated).",
        f"- Actual timing: {rules['header']['timing']}. Equal rule keys: "
        f"{rules['header']['apriori_fpgrowth_rule_keys_equal']}. "
        "FP-Growth was slower in this bounded local run.",
        "- Mine before June; select on June–August; evaluate September–November. "
        "Deterministic half-basket masking, not an online next-purchase or sales experiment.",
        "",
        "Source: recommendation_metrics.json. Timing is machine/workload-specific.",
        "",
        "## Head C and integration",
        "",
        f"- Index manifest: {json.dumps(index, ensure_ascii=False)}",
        f"- Golden questions: {assistant['golden_count']}; "
        f"category counts: {assistant['category_counts']}.",
        f"- Tool/refusal-path accuracy: {assistant['tool_choice_accuracy_all']:.4f}; "
        f"out-of-scope refusal rate: {assistant['out_of_scope_refusal_rate']:.4f}; "
        f"in-scope refusals: {assistant['in_scope_refusal_count']}.",
        f"- Expected policy-source accuracy: {assistant['expected_source_accuracy']:.4f}. "
        "Exact excerpt inclusion is a grounding proxy, not correctness/completeness.",
        f"- French-to-English retrieval smoke: "
        f"{assistant['multilingual_smoke']['english_return_policy_retrieved']}.",
        f"- {api['passed_requests']} live HTTP checks; {len(ui['checks'])} Streamlit/API "
        f"interaction checks; {ui['feature_widgets']} feature widgets, no UI exceptions.",
        "- OpenAI/Ollama were not executed. French output may quote English. Golden set "
        "is a development regression set, not an unseen language-model benchmark.",
        "",
        "Sources: assistant_regex_metrics.json, api_smoke.json, ui_smoke.json.",
        "",
        "## Tests and operations",
        "",
        "- Baseline: 100 passed, 1 failed (medical refusal); see TESTS_BEFORE.md.",
        "- Final: 123 passed, 4 dependency/deprecation warnings; logs/tests_verified.log.",
        "- Lint passed. Coverage: 64% rounded, below the unchanged 80% gate. "
        "Strict mypy: 77 errors in 13 files. CI is NOT claimed green.",
        "- MLflow local experiment and nested tuning runs saved. Registry promotion absent.",
        "- Monitoring demo sent 120 historical feature vectors with a deliberate +365-day "
        "recency perturbation; measured PSI and KS from durable local logs. "
        "This is synthetic demo traffic, not observed production drift or an SLO test.",
        "- Docker Compose config validated; daemon unavailable; no image build or cloud run.",
        "- DVC no-SCM init/commit recorded existing outputs. No remote/full repro validation. "
        "Chroma persistence mutates under serving; current index may be marked modified.",
        "- logs/commands.jsonl and logs/*.log retain successful/failed pipeline attempts.",
        "",
        "See REMAINING_ISSUES.md for the prioritized gaps and RUNBOOK.md for reproduction.",
    ]
    (reports / "VALIDATION_RESULTS.md").write_text("\n".join(lines) + "\n")
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    freeze = "\n".join(line for line in freeze.splitlines() if not line.startswith(("-e ", "#")))
    (reports / "environment_freeze.txt").write_text(freeze + "\n")
    paths = [
        p
        for parent in [root / "artifacts", root / "data/processed"]
        for p in parent.iterdir()
        if p.suffix in {".parquet", ".json", ".joblib"}
    ]
    inventory = [
        {"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": _sha256(p)}
        for p in sorted(paths)
    ]
    (reports / "artifact_inventory.json").write_text(json.dumps(inventory, indent=2))
    print("Regenerated validation report, environment snapshot and artifact inventory")


if __name__ == "__main__":
    main()
