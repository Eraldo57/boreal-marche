"""Basket selection and an explanation grounded in the API's loaded mining artifacts."""
# Developed with AI assistance; verification/provenance: docs/AI_USE.md.

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import streamlit as st

REPORT = Path(__file__).resolve().parents[1] / "reports/current_run/recommendation_metrics.json"
BENCHMARK = Path(__file__).resolve().parents[1] / "reports/current_run/rules_scaling.json"
# Capstone_Project1.pdf, Appendix B (p. 29), quotes above-40% coverage without a definition.
DEFENCE_COVERAGE_REFERENCE = 0.40


@st.cache_data(ttl=60, show_spinner=False)
def load_details(api_base: str) -> dict[str, Any]:
    response = httpx.get(f"{api_base.rstrip('/')}/recommend/details", timeout=30)
    response.raise_for_status()
    return dict(response.json())


def basket_inputs(details: dict[str, Any]) -> list[str]:
    items = details["items"]
    default = ["white hanging heart t-light holder"]
    selected = st.multiselect(
        "Choose basket items (type to search all products)",
        items,
        default=[item for item in default if item in items],
        key="basket_products",
        max_selections=100,
    )
    st.caption(
        f"All {len(items):,} descriptions known to the loaded recommender are available. "
        "Products without a matching rule can still receive popularity suggestions. "
        "Clear the selection for an empty basket."
    )
    with st.expander("Browse or download all available items"):
        triggering = {item for rule in details["rules"] for item in rule["antecedent"]}
        suggested = {item for rule in details["rules"] for item in rule["consequent"]}
        table = pd.DataFrame(
            {
                "Product description": items,
                "Can trigger a rule": [item in triggering for item in items],
                "Can be suggested by a rule": [item in suggested for item in items],
            }
        )
        search = st.text_input("Filter available items", key="basket_item_filter")
        visible = table[table["Product description"].str.contains(search, case=False, regex=False)]
        st.dataframe(visible, hide_index=True, height=300)
        st.download_button(
            "Download all items (CSV)",
            table.to_csv(index=False),
            "recommendation_items.csv",
            "text/csv",
        )
    extra = st.text_area(
        "One item per line; leave empty for popularity",
        value="",
        help="Optional additional descriptions. Popularity is used when the whole basket is empty.",
    )
    return list(
        dict.fromkeys([*selected, *(x.strip().lower() for x in extra.splitlines() if x.strip())])
    )


def _rule_table(rules: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Basket contains (A)": " + ".join(rule["antecedent"]),
                "Suggest (B)": " + ".join(rule["consequent"]),
                "Support (%)": rule["support"] * 100,
                "Confidence (%)": rule["confidence"] * 100,
                "Lift": rule["lift"],
            }
            for rule in rules
        ],
        columns=["Basket contains (A)", "Suggest (B)", "Support (%)", "Confidence (%)", "Lift"],
    )


def show_mining_explanation(details: dict[str, Any]) -> None:
    st.subheader("How the basket rules were learned")
    data, algorithms, conclusions, rules_tab = st.tabs(
        ["Data and timeline", "Apriori vs FP-Growth", "Measured conclusions", "Explore all rules"]
    )
    header, rules = details["metadata"], details["rules"]
    memory = header.get("memory_report", {})
    with data:
        st.markdown(
            "Each cleaned **purchase invoice becomes one basket**. Repeated descriptions within "
            "an invoice count once: mining uses presence, not quantity or price. Returns and "
            "cancellations are excluded. Items are lowercase product descriptions, so different "
            "SKUs can merge and description variants can split the same product."
        )
        if memory:
            cols = st.columns(3)
            cols[0].metric("Training baskets", f"{memory['n_invoices']:,}")
            cols[1].metric("Training descriptions", f"{memory['n_items']:,}")
            cols[2].metric("Stored nonzero entries", f"{100 - memory['sparsity_pct']:.2f}%")
            st.write(
                f"The input is a basket × product presence matrix. A cell is True when an "
                f"invoice contains that description. It uses {memory['sparse_mb']:.2f} MB of "
                f"CSR storage versus {memory['dense_mb']:.2f} MB for equivalent dense booleans."
            )
        st.table(
            pd.DataFrame(
                [
                    {
                        "Period": "Before 2011-06-01",
                        "Use": "Mine itemsets/rules and calculate popularity",
                    },
                    {
                        "Period": "June–August 2011",
                        "Use": "Choose support/confidence using basket completion",
                    },
                    {
                        "Period": "September–November 2011",
                        "Use": "Evaluate the frozen rules on later baskets",
                    },
                ]
            )
        )
        st.caption("These dates describe the saved local study; later baskets were not mined.")
    with algorithms:
        st.markdown(
            "**Apriori** first counts individual items, then builds candidate pairs and triples. "
            "It prunes candidates whose subsets are not frequent: a combination cannot occur "
            "more often than any of its subsets. It counts surviving candidates "
            "against the baskets.\n\n"
            "**FP-Growth** counts frequent items, orders them by frequency, and compresses baskets "
            "into a prefix tree with shared paths. It mines conditional pattern trees to discover "
            "frequent combinations without Apriori's explicit candidate-generation step.\n\n"
            "**Both produce frequent itemsets and their support.** A separate association-rule "
            "step splits each itemset into A → B and computes confidence and lift. FP-Growth "
            "supplies the serving rules; Apriori checks agreement on the same training matrix "
            "and selected settings. Neither runs again when you click Get recommendations."
        )
        if header:
            st.write(
                f"Selected minimum support: {header['min_support']:.1%}; confidence: "
                f"{header['min_confidence']:.0%}; lift: {header['min_lift']:g}; "
                f"maximum combined itemset size: {header['max_itemset_length']} products."
            )
            if memory:
                minimum_baskets = math.ceil(header["min_support"] * memory["n_invoices"])
                st.write(
                    f"Support requires at least {minimum_baskets} "
                    "training baskets containing the entire combination."
                )
        st.latex(r"\mathrm{support}(A\to B)=\frac{n(A\cap B)}{N}")
        st.latex(r"\mathrm{confidence}(A\to B)=\frac{n(A\cap B)}{n(A)}")
        st.latex(r"\mathrm{lift}(A\to B)=\frac{\mathrm{confidence}(A\to B)}{n(B)/N}")
        st.caption(
            "N is all training baskets; n(A), n(B) and n(A∩B) count baskets containing "
            "A, B and both. A and B may each contain several products."
        )
        st.markdown(
            "Algorithm references: [Apriori](https://rasbt.github.io/mlxtend/user_guide/"
            "frequent_patterns/apriori/), [FP-Growth](https://rasbt.github.io/mlxtend/"
            "user_guide/frequent_patterns/fpgrowth/), and [association rules]"
            "(https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/)."
        )
    with conclusions:
        _show_conclusions(header, rules)
    with rules_tab:
        st.write(f"All {len(rules):,} loaded rules, searchable and downloadable.")
        table = _rule_table(rules)
        search = st.text_input("Find a product in the learned rules", key="rule_search")
        mask = table["Basket contains (A)"].str.contains(search, case=False, regex=False) | table[
            "Suggest (B)"
        ].str.contains(search, case=False, regex=False)
        st.dataframe(table[mask], hide_index=True, height=350)
        st.download_button(
            "Download all rules (CSV)",
            table.to_csv(index=False),
            "association_rules.csv",
            "text/csv",
        )
        if not rules:
            st.info("No association rules are loaded; recommendations use popularity.")
            return
        index = st.selectbox(
            "Explain one learned rule",
            options=list(range(len(rules))),
            format_func=lambda i: (
                f"{' + '.join(rules[i]['antecedent'])} → {' + '.join(rules[i]['consequent'])}"
            ),
        )
        rule = rules[index]
        st.write(
            f"Among baskets containing A, {rule['confidence']:.1%} also contained B. "
            f"Both appeared together in {rule['support']:.3%} of all training baskets. "
            f"B was {rule['lift']:.2f} times as common with A as it was overall."
        )
        if memory and rule["confidence"] > 0 and rule["lift"] > 0:
            n = memory["n_invoices"]
            both = round(rule["support"] * n)
            a = round(rule["support"] / rule["confidence"] * n)
            b = round(rule["confidence"] / rule["lift"] * n)
            st.caption(
                f"Counts reconstructed from saved support/confidence/lift (rounded): "
                f"N = {n:,}; n(A) ≈ {a:,}; n(B) ≈ {b:,}; n(A∩B) ≈ {both:,}."
            )
            st.latex(
                rf"\mathrm{{support}}=\frac{{{both}}}{{{n}}}"
                rf"\approx {100 * rule['support']:.3f}\%"
            )
            st.latex(
                rf"\mathrm{{confidence}}=\frac{{{both}}}{{{a}}}"
                rf"\approx {100 * rule['confidence']:.1f}\%"
            )
            st.latex(
                rf"\mathrm{{lift}}=\frac{{{both}/{a}}}{{{b}/{n}}}"
                rf"\approx {rule['lift']:.2f}"
            )
        st.caption(
            "An association describes historical co-occurrence, not causation or a guarantee "
            "that this customer will buy. Serving matches the whole A, excludes basket items, "
            "keeps the highest-lift rule for each suggestion, and ranks by lift."
        )


def _show_conclusions(header: dict[str, Any], rules: list[dict[str, Any]]) -> None:
    rule_count = len(rules)
    if header:
        st.metric("Loaded association rules", f"{rule_count:,}")
        timing = header["timing"]
        st.table(
            pd.DataFrame(
                [
                    {"Algorithm": "Apriori", "Measured seconds": timing["apriori_seconds"]},
                    {"Algorithm": "FP-Growth", "Measured seconds": timing["fpgrowth_seconds"]},
                ]
            )
        )
        same = header["apriori_fpgrowth_rule_keys_equal"]
        st.write(f"Same antecedent/consequent rule keys: {'yes' if same else 'no'}.")
        faster = (
            "Apriori" if timing["apriori_seconds"] < timing["fpgrowth_seconds"] else "FP-Growth"
        )
        st.caption(
            f"{faster} was faster in this bounded local comparison; speed is workload-specific."
        )
    try:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        if not header or report["header"] != header:
            st.info("The saved evaluation report does not match the loaded rules.")
            return
        final = report["final_test"]
    except (OSError, ValueError, KeyError):
        st.info("The saved basket-completion evaluation is unavailable in this installation.")
        return
    st.markdown("**Later-basket completion results**")
    st.write(
        "Half of each eligible test basket is visible, and the other half is hidden. "
        "A hit means at least one of the five suggestions appears in the hidden half. "
        "Single-product baskets cannot be split and are excluded."
    )
    st.bar_chart(
        pd.DataFrame(
            {
                "Hit-rate@5": {
                    "Rules + fallback": final["hit_rate_at_5"],
                    "Popularity only": final["popularity_hit_rate_at_5"],
                }
            }
        )
    )
    st.table(
        pd.DataFrame(
            [
                {
                    "Measure": "Rules + fallback Hit-rate@5",
                    "Result": f"{final['hit_rate_at_5']:.2%}",
                },
                {
                    "Measure": "Popularity Hit-rate@5",
                    "Result": f"{final['popularity_hit_rate_at_5']:.2%}",
                },
                {"Measure": "Evaluated baskets", "Result": f"{final['baskets_evaluated']:,}"},
                {
                    "Measure": "Single-product baskets excluded",
                    "Result": f"{final['single_item_excluded']:,}",
                },
                {
                    "Measure": "Baskets served by matching rules",
                    "Result": f"{final['rule_firing_coverage']:.2%}",
                },
                {
                    "Measure": "Baskets served by popularity fallback",
                    "Result": f"{1 - final['rule_firing_coverage']:.2%}",
                },
                {
                    "Measure": "Training catalogue ever recommended",
                    "Result": f"{final['catalogue_coverage']:.2%}",
                },
            ]
        )
    )
    baseline_hit_rate = final["popularity_hit_rate_at_5"]
    if baseline_hit_rate:
        relative_hit_rate = final["hit_rate_at_5"] / baseline_hit_rate
        st.markdown(f"**Hit-rate@5 is {relative_hit_rate:.2f}× the popularity baseline.**")
        st.latex(
            rf"\mathrm{{relative\ hit\ rate}}="
            rf"\frac{{{final['hit_rate_at_5']:.6f}}}{{{baseline_hit_rate:.6f}}}"
            rf"\approx {relative_hit_rate:.2f}"
        )
        st.caption(
            "This compares the chance of finding at least one hidden item among five "
            "suggestions. It does not mean this many times more sales or more items bought."
        )
    st.write(
        f"Matching rules served {final['rule_firing_coverage']:.2%} of evaluated baskets. "
        f"The other {1 - final['rule_firing_coverage']:.2%} used popularity because no eligible "
        "rule-based suggestion was available after matching and basket-item exclusion. "
        "This basket-level fallback rate is different from catalogue coverage."
    )
    st.markdown("**Which products do the rules cover?**")
    antecedents = {item for rule in rules for item in rule["antecedent"]}
    consequents = {item for rule in rules for item in rule["consequent"]}
    catalogue_size = header["memory_report"]["n_items"]
    recommended = round(final["catalogue_coverage"] * catalogue_size)
    st.table(
        pd.DataFrame(
            [
                {
                    "Product group": "Antecedent products",
                    "Distinct products": len(antecedents),
                    "Meaning": "Appear on the basket-contains side of at least one loaded rule",
                },
                {
                    "Product group": "Consequent products",
                    "Distinct products": len(consequents),
                    "Meaning": "Appear on the suggest side of at least one loaded rule",
                },
                {
                    "Product group": "Actually recommended in test",
                    "Distinct products": recommended,
                    "Meaning": "Returned among the top five across evaluated test baskets",
                },
            ]
        )
    )
    st.caption(
        "These groups overlap; they are not separate pieces of the catalogue. "
        "The actually-recommended count includes both rule and popularity outputs and is "
        "derived from the saved coverage ratio and training catalogue size. A consequent "
        "can exist in the rules without ever reaching the test's top five suggestions."
    )
    st.latex(
        rf"\mathrm{{catalogue\ coverage}}=\frac{{{recommended}}}{{{catalogue_size}}}"
        rf"\times100\%={100 * final['catalogue_coverage']:.2f}\%"
    )
    gap = DEFENCE_COVERAGE_REFERENCE - final["catalogue_coverage"]
    st.write(
        f"Coverage reference: {DEFENCE_COVERAGE_REFERENCE:.0%}. "
        f"Measured catalogue coverage: {final['catalogue_coverage']:.2%}. "
        f"The measured value is {abs(gap) * 100:.2f} percentage points "
        f"{'below' if gap >= 0 else 'above'} that reference."
    )
    st.caption(
        "The original rubric (Appendix B, page 29) quotes coverage above 40% for an excellent "
        "rating but does not specify catalogue coverage or rule-firing coverage. Both are "
        "reported here; clarify the intended definition rather than equating them. "
        "Coverage is not a model accuracy measure."
    )
    _show_lookup_latency(rule_count)
    st.caption(
        "Conclusion: the hybrid beat popularity for this offline basket-completion task. "
        "Catalogue coverage is limited, and these results do not measure increased sales "
        "or predict the next order."
    )
    with st.expander("How the settings were selected (validation only)"):
        sweep = pd.DataFrame(report["sweep_validation"])[
            ["support", "confidence", "exported_rules", "hit_rate_at_5", "catalogue_coverage"]
        ]
        st.dataframe(sweep, hide_index=True)
        st.caption(
            "Choose the highest validation Hit-rate@5, then higher catalogue coverage, "
            "then fewer rules if still tied. The final test baskets do not choose the settings."
        )


def _show_lookup_latency(rule_count: int) -> None:
    st.markdown("**Local recommendation-engine latency**")
    st.write(
        "Serving uses an inverted index from each basket item to candidate rules, checks "
        "the full antecedent, excludes existing basket items, deduplicates suggestions and "
        "ranks them by lift. This benchmark times that engine call."
    )
    try:
        benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
        result = next(
            row
            for row in benchmark["results"]
            if row["rules_are"] == "real export" and row["rules"] == rule_count
        )
    except (OSError, ValueError, KeyError, StopIteration):
        st.info("No saved real-export latency benchmark is available for this rule count.")
        return
    st.table(
        pd.DataFrame(
            [
                {"Measure": "Median (p50)", "Milliseconds": result["recommend_ms_p50"]},
                {"Measure": "95th percentile (p95)", "Milliseconds": result["recommend_ms_p95"]},
                {"Measure": "Maximum observed", "Milliseconds": result["recommend_ms_max"]},
            ]
        )
    )
    st.write(
        f"A p95 of {result['recommend_ms_p95']:.3f} ms means 95% of the measured engine calls "
        "finished within that time. It is a percentile of the observed calls, not a guarantee "
        "for future requests."
    )
    st.caption(
        f"Historical local benchmark recorded {benchmark['executed_at'][:10]}: "
        f"{benchmark['queries']} calls using {benchmark['query_source']}; "
        f"{result['rules']:,} real exported rules. Excludes HTTP, network and Streamlit time. "
        "Larger synthetic rule-set timings are not used for these figures."
    )
