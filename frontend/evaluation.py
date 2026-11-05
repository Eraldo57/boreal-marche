"""Display exported test curves without loading customer data or model libraries."""
# Developed with AI assistance; verification/provenance: docs/AI_USE.md.

import json
from pathlib import Path

import pandas as pd
import streamlit as st

FIGURES = Path(__file__).resolve().parents[1] / "reports/current_run/figures"
DIAGNOSTICS = FIGURES.parent / "retention_diagnostics.json"


def show_ranking_plots() -> None:
    st.subheader("Held-out model evaluation")
    st.caption(
        "Saved test-snapshot results, not the current customer's score. "
        "Positive means a purchase within 90 days. These curves evaluate purchase "
        "prediction, not the effect of a coupon."
    )
    pr, roc, calibration, quality = st.tabs(
        ["Precision–Recall", "ROC", "Calibration", "Threshold and uncertainty"]
    )
    for tab, name, explanation in (
        (
            pr,
            "precision_recall_curve.png",
            "Recall: how many actual returners are selected. Precision: how many selected "
            "customers actually return. AP is average precision, reported as PR-AUC "
            "in this project.",
        ),
        (
            roc,
            "roc_curve.png",
            "Recall: the share of actual returners selected. False-positive rate: the share "
            "of actual non-returners selected. The diagonal represents random ranking.",
        ),
        (
            calibration,
            "reliability_curve.png",
            "The diagonal is perfect calibration: customers assigned a probability near 80% "
            "should return about 80% of the time. Curves compare methods on the June decision "
            "period, after calibrators were fitted on March data.",
        ),
    ):
        with tab:
            path = FIGURES / name
            if not path.exists():
                st.info("This installation does not include the saved test curve.")
                continue
            st.image(str(path), width="stretch")
            st.caption(explanation)
            st.download_button(
                "Download plot",
                data=path.read_bytes(),
                file_name=name,
                mime="image/png",
                key=f"download_{name}",
            )
    try:
        diagnostics = json.loads(DIAGNOSTICS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        with quality:
            st.info("Saved threshold diagnostics are unavailable in this installation.")
        return
    with calibration:
        st.table(
            pd.DataFrame(
                [
                    {"June decision-period method": method, "Brier score": result["brier_score"]}
                    for method, result in diagnostics["calibration_decision_period"].items()
                ]
            )
        )
        st.write(
            f"Selected calibration: **{diagnostics['selected_calibration']}**. "
            f"Later September test Brier: **{diagnostics['model']['brier_score']:.4f}**."
        )
        st.latex(r"\mathrm{Brier}=\frac{1}{N}\sum_i(p_i-y_i)^2")
        st.caption(
            "Lower is better. Brier measures probability quality, not an error percentage "
            "or calibration alone. June and September scores are on different periods."
        )
    with quality:
        model = diagnostics["model"]
        st.write(
            f"Saved model {diagnostics['model_version']}: {diagnostics['feature_count']} "
            f"serving features; {diagnostics['customers']:,} test customers; "
            f"threshold {diagnostics['threshold']:.1%}."
        )
        st.markdown("**Confusion matrix at the saved threshold**")
        st.dataframe(
            pd.DataFrame(
                [[model["tn"], model["fp"]], [model["fn"], model["tp"]]],
                index=["Actually did not return", "Actually returned"],
                columns=["Not selected", "Selected"],
            )
        )
        st.caption(
            "A selected returner is a true positive; a selected non-returner is a false "
            "positive. These are purchase outcomes, not responses caused by coupons."
        )
        intervals = diagnostics["uncertainty"]["ci95"]
        st.table(
            pd.DataFrame(
                [
                    {
                        "Measure": label,
                        "Value": model[key],
                        "95% interval": f"{intervals[key][0]:.4f} to {intervals[key][1]:.4f}",
                    }
                    for key, label in [
                        ("pr_auc", "Average precision"),
                        ("roc_auc", "ROC-AUC"),
                        ("brier_score", "Brier score"),
                        ("recall", "Recall"),
                        ("precision", "Precision"),
                    ]
                ]
            )
        )
        baseline = diagnostics["recency_same_selection_count"]
        st.write(
            f"At the same selection count: model recall {model['recall']:.2%}; "
            f"recency-rule recall {baseline['recall']:.2%}. "
            f"Selected share: {model['selection_rate']:.2%}."
        )
        delta = intervals["ap_difference"]
        st.write(
            f"Paired 95% interval for model AP minus recency AP: "
            f"{delta[0]:+.4f} to {delta[1]:+.4f}."
        )
        st.caption(
            f"{diagnostics['uncertainty']['draws_used']:,} bootstrap draws. "
            + diagnostics["uncertainty"]["limitation"]
        )
