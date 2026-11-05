"""Explain the saved coupon utility experiment without changing the serving policy."""
# Developed with AI assistance; verification/provenance: docs/AI_USE.md.

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import streamlit as st

REPORT_PATH = Path(__file__).resolve().parents[1] / "reports/current_run/retention_metrics.json"


def show_coupon_math(probability: float, active_threshold: float) -> None:
    """Render the report's assumptions, customer calculation and historical utility curve.

    The curve contains measured validation outcomes, not expected customer utilities.
    Exploring it never changes the active threshold or the recorded decision.
    """
    with st.expander("Coupon math: expected value and threshold utility"):
        st.warning(
            "Demonstration assumptions, not validated coupon profit. This formula credits "
            "a returning customer's margin to the offer even if they would buy without it. "
            "It does not measure the extra purchases caused by a coupon."
        )
        try:
            report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
            selection = report["threshold_selection"]
            costs = selection["cost_assumptions"]
            cost = float(costs["cost_coupon"])
            margin = float(costs["margin_retained"])
            missed = float(costs["cost_false_negative"])
            thresholds = [float(t) for t in selection["curve_thresholds"]]
            utilities = [float(u) for u in selection["curve_utility"]]
            if (
                not thresholds
                or len(thresholds) != len(utilities)
                or any(
                    not math.isfinite(v) for v in [cost, margin, missed, *thresholds, *utilities]
                )
                or min(cost, margin, missed) < 0
                or margin + missed <= 0
            ):
                raise ValueError("Invalid utility report")
        except (OSError, ValueError, KeyError, TypeError):
            st.info(
                "Saved validation utility report unavailable; no results have been substituted."
            )
            return

        st.markdown("**1. Assumed value for this customer**")
        st.write(
            f"Coupon cost C = {cost:g}; return margin M = {margin:g}; "
            f"missed-return cost F = {missed:g}. Values are assumed monetary units. "
            "The coupon cost is charged for every offer, including when the customer does not buy."
        )
        st.latex(r"E[U_{\mathrm{offer}}]=pM-C,\qquad E[U_{\mathrm{no\ action}}]=-pF")
        st.latex(
            rf"E[U_{{\mathrm{{offer}}}}]={probability:.6f}\times {margin:g}"
            rf"-{cost:g}={probability * margin - cost:.2f}"
        )
        st.caption(
            f"p = {probability:.2%}, the current customer's predicted purchase probability. "
            f"Assumed value of no action: {-probability * missed:.2f}. "
            "This calculation updates when you change the customer's features."
        )
        break_even = cost / (margin + missed)
        st.latex(
            r"p_{\mathrm{break\ even}}=\frac{C}{M+F}="
            rf"\frac{{{cost:g}}}{{{margin:g}+{missed:g}}}={break_even:.1%}".replace("%", r"\%")
        )

        st.markdown("**2. Utility across actual validation customers**")
        st.latex(
            r"U(t)=\sum_i\left[\mathbf{1}(p_i\geq t)(y_iM-C)"
            r"-\mathbf{1}(p_i<t)y_iF\right]"
        )
        st.write(
            "t is the candidate threshold; pᵢ is a customer's predicted probability; "
            "yᵢ is 1 if they actually purchased in the next 90 days, otherwise 0. "
            "Customers at or above t are selected. The saved experiment tested 201 thresholds "
            "from 0 to 1, in steps of 0.005, on the June 2011 decision period."
        )
        best_index = max(range(len(utilities)), key=utilities.__getitem__)
        best = thresholds[best_index]
        st.info(
            f"Saved validation optimum: {best:.1%}, with assumed utility "
            f"{utilities[best_index]:,.0f}. The theoretical break-even is {break_even:.1%}; "
            "the empirical optimum uses observed outcomes, which differ from "
            "predicted probabilities."
        )
        if not math.isclose(best, active_threshold, abs_tol=1e-9):
            st.warning(
                f"The current API threshold is {active_threshold:.1%}. This historical report's "
                f"optimum is {best:.1%}; it does not explain the current model's threshold."
            )
        curve = pd.DataFrame({"Threshold (%)": [100 * t for t in thresholds], "Utility": utilities})
        st.line_chart(curve, x="Threshold (%)", y="Utility", x_label="Candidate threshold (%)")
        inspected = st.select_slider(
            "Inspect a saved validation threshold",
            options=thresholds,
            value=best,
            format_func=lambda t: f"{t:.1%}",
            key="coupon_utility_threshold",
        )
        st.metric("Assumed validation utility", f"{utilities[thresholds.index(inspected)]:,.0f}")
        st.caption(
            f"Exploration only. Active prediction threshold: {active_threshold:.1%}. "
            "Moving this slider does not change the coupon proposal. "
            "Source: the saved retention validation report."
        )
