"""Local demonstration UI. All inference/retrieval and feature lookup go through FastAPI."""

import os

import httpx
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Boréal Marché", page_icon="🍁", layout="wide")
API_BASE = st.sidebar.text_input(
    "API URL", value=os.getenv("API_BASE_URL", "http://localhost:8000")
)
language = st.sidebar.selectbox("Language / Langue", ["en", "fr"])
provider = st.sidebar.selectbox("Assistant provider", ["regex", "openai", "ollama"])
st.sidebar.caption("Local academic demonstration. No authenticated customer accounts.")


def api_call(method, endpoint, payload=None):
    """Display actionable backend errors, never fabricate a fallback prediction."""
    try:
        response = httpx.request(method, API_BASE + endpoint, json=payload, timeout=45)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        st.error(f"API {exc.response.status_code}: {exc.response.text[:500]}")
    except httpx.HTTPError as exc:
        st.error(f"Backend unavailable at {API_BASE}: {exc}. Start FastAPI using the runbook.")
    return None


st.title("Boréal Marché — Customer Intelligence")
st.caption("Vanier capstone • Integration and validation milestone")
ready = api_call("GET", "/ready")
if ready and ready["ready"]:
    st.success("Backend ready: retention, recommendations and document index loaded.")
else:
    st.warning("Backend/artifacts are unavailable. See /ready and reports/current_run/RUNBOOK.md.")

prediction, recommendation, assistant, monitoring, about = st.tabs(
    ["Prediction", "Recommendations", "Assistant", "Monitoring", "About"]
)

with prediction:
    st.header("90-day purchase / Achat dans les 90 jours")
    st.caption(
        "Probability means WILL RETURN. Risk means NOT returning. "
        "Historical GBP amounts remain GBP."
    )
    customer_id = st.text_input("Historical demo customer ID", value="12347")
    if st.button("Load historical customer features"):
        record = api_call("GET", f"/customers/{customer_id}/features")
        if record:
            st.session_state["feature_record"] = record
    record = st.session_state.get("feature_record")
    if record:
        contract = api_call("GET", "/feature-contract")
        st.info(
            f"Historical snapshot: {record['snapshot']}. What-if: change any value and the "
            "probability below updates as soon as you leave the field."
        )
        features = {}
        columns = st.columns(3)
        for i, name in enumerate(contract["feature_names"] if contract else record["features"]):
            with columns[i % 3]:
                features[name] = st.number_input(
                    name,
                    value=float(record["features"][name]),
                    key=f"feature_{record['customer_id']}_{name}",
                )
        # Live: score again whenever the values change, and only then (no duplicate log rows).
        asked = (record["customer_id"], tuple(features.items()))
        if st.session_state.get("predicted_for") != asked:
            fresh = api_call(
                "POST", "/predict", {"customer_id": record["customer_id"], "features": features}
            )
            if fresh:
                st.session_state["predicted_for"], st.session_state["last_prediction"] = (
                    asked,
                    fresh,
                )
        result = st.session_state.get("last_prediction")
        if result and st.session_state.get("predicted_for") == asked:
            st.metric("Probability of purchase within 90 days", f"{result['probability']:.1%}")
            st.write(
                f"Non-return risk: {result['risk_label']}; model decision: {result['will_return']}"
            )
            st.caption(
                f"Assumption-based threshold: {result['threshold']:.3f}; "
                f"model {result['model_version']}"
            )
            if result["top_factors"]:
                with st.expander("Global importance hints (not individual explanations)"):
                    st.json(result["top_factors"])

            st.subheader("Human review / Révision humaine")
            proposed = "offer_coupon" if result["will_return"] else "no_action"
            st.write(
                f"The model proposes: **{proposed.replace('_', ' ')}**. A person decides; "
                "nothing is sent to the customer."
            )
            st.caption(
                "Why this step exists: under Québec's Law 25, a decision based only on automated "
                "processing brings disclosure and human-review obligations, so the model never "
                "decides alone. The proposal follows the placeholder coupon policy (at or above "
                "the threshold); it predicts returning, not that a coupon changes behaviour."
            )
            with st.form("review_form", clear_on_submit=True):
                reviewer = st.text_input("Reviewer name")
                reason = st.text_area("Reason for the decision")
                decision = st.radio("Decision", ["approved", "rejected"], horizontal=True)
                if st.form_submit_button("Record decision"):
                    saved = api_call(
                        "POST",
                        "/decisions",
                        {
                            "customer_id": result["customer_id"],
                            "probability": result["probability"],
                            "threshold": result["threshold"],
                            "model_version": result["model_version"],
                            "proposed_action": proposed,
                            "decision": decision,
                            "reviewer": reviewer,
                            "reason": reason,
                        },
                    )
                    if saved:
                        st.success(
                            f"Decision {saved['id']} recorded: {saved['decision']} by "
                            f"{saved['reviewer']}. Nothing was sent to the customer."
                        )
            with st.expander("Recent decisions (audit trail)"):
                recent = api_call("GET", "/decisions?limit=10")
                if recent:
                    st.dataframe(pd.DataFrame(recent), hide_index=True)
                else:
                    st.write("No decision recorded yet.")
    else:
        st.info("Load a historical record to obtain the complete canonical feature vector.")

with recommendation:
    st.header("Basket completion")
    st.caption(
        "Use UCI giftware descriptions. The fictional grocery catalogue is "
        "a separate assistant demonstration."
    )
    basket_text = st.text_area(
        "One item per line; leave empty for popularity", value="white hanging heart t-light holder"
    )
    if st.button("Get recommendations"):
        result = api_call(
            "POST",
            "/recommend",
            {"basket": [x.strip() for x in basket_text.splitlines() if x.strip()], "top_k": 5},
        )
        if result:
            st.write(result["recommendations"])
            st.caption(f"Strategy: {result['strategy']}")
            with st.expander("Firing rules: support, confidence and lift"):
                st.json(result["because"])

with assistant:
    st.header("Grounded assistant / Assistant avec sources")
    st.caption(
        "Regex returns source excerpts, which may be English for a French query. "
        "OpenAI/Ollama are optional tool-routing providers."
    )
    chat_customer = st.text_input(
        "Demo customer context (only needed for retention)", value="12347"
    )
    question = st.text_input(
        "Question", value="What is the return policy for non-perishable items?"
    )
    if st.button("Ask assistant"):
        result = api_call(
            "POST",
            "/chat",
            {
                "message": question,
                "customer_id": chat_customer or None,
                "language": language,
                "provider": provider,
            },
        )
        if result:
            st.write(result["reply"])
            st.caption(f"Provider: {result['provider_used']}; refused: {result['refused']}")
            for citation in result["citations"]:
                with st.expander(citation["source"]):
                    st.write(citation["chunk"])
            with st.expander("Typed tool execution"):
                st.json(result["tool_calls"])

with monitoring:
    st.header("Live monitoring")
    st.caption(
        "Computed from the prediction_log table (SQLite locally, PostgreSQL in Docker Compose) "
        "against the training reference. Rules: configs/pipeline.yaml; design: docs/MONITORING.md."
    )
    if st.button("Refresh metrics"):
        result = api_call("GET", "/metrics")
        alerts = api_call("GET", "/alerts")
        if result:
            show = {"green": st.success, "yellow": st.warning, "red": st.error}
            show.get(result["status"], st.info)(
                f"Status: {result['status']} (the worst level among the checks below)"
            )
            fire_rate = result["recommendation"].get("rule_fire_rate")
            kpis = st.columns(4)
            kpis[0].metric("Requests", result["request_count"])
            kpis[1].metric("Error rate", f"{result['error_rate']:.1%}")
            kpis[2].metric("p95 latency", f"{result['latency_p95_ms']:.0f} ms")
            kpis[3].metric("Rule fire rate", "n/a" if fire_rate is None else f"{fire_rate:.0%}")
            st.subheader("Checks, each with its rule")
            st.dataframe(pd.DataFrame(result["checks"]), hide_index=True)
            drift = result["drift_metrics"] + result["prediction_drift"]
            if drift:
                st.subheader("Drift against the training reference")
                table = pd.DataFrame(drift).pivot_table(
                    index="feature", columns="statistic", values="value"
                )
                st.dataframe(table.sort_values("psi", ascending=False))
            else:
                st.info(f"Drift not measured yet ({result['drift_status']}).")
            st.subheader("Data quality and request volume")
            quality, volume = st.columns(2)
            quality.json(result["data_quality"])
            volume.bar_chart(pd.Series(result["requests_by_route"], name="requests"))
        if alerts:
            st.subheader("Automated alerts")
            st.caption(
                f"Evaluated in the background; delivered to: {', '.join(alerts['channels'])}. "
                f"Webhook configured: {alerts['webhook_configured']}."
            )
            if alerts["alerts"]:
                st.dataframe(pd.DataFrame(alerts["alerts"]), hide_index=True)
            else:
                st.write("No alert raised so far.")
        st.caption(
            "True accuracy needs labels that only exist 90 days after each prediction; "
            "the backfill design is in docs/MONITORING.md."
        )

with about:
    st.header("Implementation and evidence")
    version = api_call("GET", "/version")
    if version:
        st.json(version)
    st.write(
        "Public UCI Online Retail II transactions are re-skinned as fictional Boréal Marché. "
        "They are not Québec grocery customer data. Original currency: GBP."
    )
    st.write(
        "The authored policy/product corpus describes a fictional retailer. "
        "It does not prove actual business operations or legal compliance."
    )
    st.write(
        "Human review, authenticated access, retention/deletion enforcement, cloud deployment "
        "and full fairness evaluation remain next-phase work."
    )
    st.write(
        "OpenAI may send the message and retrieved context to an external provider. "
        "Regex runs locally; Ollama requires its own configured server."
    )
    st.caption("Evidence: reports/current_run/VALIDATION_RESULTS.md and REMAINING_ISSUES.md")
