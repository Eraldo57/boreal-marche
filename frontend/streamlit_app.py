"""Demonstration UI. All inference/retrieval and feature lookup go through FastAPI."""

import json
import os

import httpx
import pandas as pd
import streamlit as st
from about import DIAGRAM_CSS, mermaid_to_dot, read_document
from basket import basket_inputs, load_details, show_mining_explanation
from coupon import show_coupon_math
from evaluation import show_ranking_plots
from notes import effect_lines, reason_lines, strategy_note

st.set_page_config(page_title="Boréal Marché", page_icon="🍁", layout="wide")
HOSTED_DEMO = os.getenv("HOSTED_DEMO", "").lower() == "true"
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
if not HOSTED_DEMO:
    API_BASE = st.sidebar.text_input("API URL", value=API_BASE)
language = st.sidebar.selectbox("Language / Langue", ["en", "fr"])
providers = os.getenv("ASSISTANT_PROVIDERS", "regex,openai,ollama").split(",")
provider = st.sidebar.selectbox("Assistant provider", providers)
st.sidebar.caption(
    "Academic demonstration using historical public data. No authenticated customer accounts."
)


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


def stream_chat(payload, final):
    """Yield the answer's words as they arrive from /chat/stream; the closing event fills
    *final* with the citations, the tool call and who wrote the answer."""
    try:
        with httpx.stream("POST", API_BASE + "/chat/stream", json=payload, timeout=180) as response:
            if response.status_code >= 400:
                response.read()
                final["error"] = f"API {response.status_code}: {response.text[:500]}"
                return
            for line in response.iter_lines():
                event = json.loads(line) if line else {}
                if event.get("type") == "token":
                    yield event["text"]
                elif event.get("type") == "fallback":
                    final["fallback"] = event["reason"]
                elif event.get("type") == "done":
                    final.update(event)
    except httpx.HTTPError as exc:
        final["error"] = f"Backend unavailable at {API_BASE}: {exc}"


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
            reasons = effect_lines(result)
            if reasons:
                st.subheader("Why this score / Pourquoi ce score")
                st.markdown("\n".join(f"- {line}" for line in reasons))
                st.caption(
                    "Compared with a typical customer (the training median) on each feature, "
                    "one at a time. Features are related, so these do not add up. This is "
                    "association in past data, not a cause and not the effect of a coupon."
                )
            if result["top_factors"]:
                with st.expander("Global importance (the same for every customer)"):
                    st.json(result["top_factors"])

            show_coupon_math(result["probability"], result["threshold"])

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
    try:
        details = load_details(API_BASE)
    except httpx.HTTPError:
        details = None
        st.info("Product picker unavailable. You can still enter product descriptions below.")
    if details:
        basket_items = basket_inputs(details)
    else:
        basket_text = st.text_area(
            "One item per line; leave empty for popularity",
            value="white hanging heart t-light holder",
        )
        basket_items = [x.strip() for x in basket_text.splitlines() if x.strip()]
    top_k = 5
    recommendation_request = (tuple(basket_items), top_k)
    if st.button("Get recommendations"):
        fresh = api_call("POST", "/recommend", {"basket": basket_items, "top_k": top_k})
        st.session_state["last_recommendation"] = fresh
        st.session_state["recommended_for"] = recommendation_request
    result = st.session_state.get("last_recommendation")
    if result and st.session_state.get("recommended_for") == recommendation_request:
        if result["strategy"] != "rule_based":
            level, note = strategy_note(result)
            getattr(st, level)(note)
        lines = reason_lines(result)
        st.markdown("\n".join(f"- {line}" for line in lines or result["recommendations"]))
        st.caption(f"Strategy: {result['strategy']}")
        with st.expander("Firing rules: support, confidence and lift"):
            st.json(result["because"])
    if details:
        show_mining_explanation(details)

with assistant:
    st.header("Grounded assistant / Assistant avec sources")
    st.caption(
        "Answers stream word by word. With regex, a policy answer quotes the source excerpt, "
        "which may be English for a French question. With OpenAI or Ollama, the model writes "
        "the answer from the cited excerpts in the chosen language. Scores, recommendations and "
        "products always come from the tools, never from a model."
    )
    chat_customer = st.text_input(
        "Demo customer context (only needed for retention)", value="12347"
    )
    question = st.text_input(
        "Question", value="What is the return policy for non-perishable items?"
    )
    if st.button("Ask assistant"):
        final = {}
        payload = {
            "message": question,
            "customer_id": chat_customer or None,
            "language": language,
            "provider": provider,
        }
        st.write_stream(stream_chat(payload, final))
        if "error" in final:
            st.error(final["error"])
        if "fallback" in final:
            st.warning(final["fallback"])
        by_model = final.get("answer_written_by") not in {"source text", "template"}
        if final.get("passage_confidence") == "uncertain" and by_model:
            # A model wrote this answer. When it quotes the source instead, the notice is
            # already the first lines of the reply.
            st.warning(
                "The best policy passage matched this question only weakly, so the answer "
                "may not be about what you asked. Check the excerpt below, or contact support."
            )
        figures = final.get("figure_check", {})
        if figures.get("figures_not_in_sources"):
            st.warning(
                "The answer states figures no cited source contains: "
                f"{', '.join(figures['figures_not_in_sources'])}. Trust the excerpts below."
            )
        if figures.get("top_source_figures_missing"):
            st.warning(
                "The main source states figures the answer leaves out: "
                f"{', '.join(figures['top_source_figures_missing'])}. Read the excerpt below."
            )
        if "reply" in final:
            st.caption(
                f"Provider: {final['provider_used']}; refused: {final['refused']}; "
                f"answer written by: {final['answer_written_by']}"
            )
            for citation in final["citations"]:
                with st.expander(citation["source"]):
                    st.write(citation["chunk"])
            with st.expander("Typed tool execution"):
                st.json(final["tool_calls"])

with monitoring:
    st.header("Live monitoring")
    st.caption(
        "Computed from the prediction_log table (SQLite locally, PostgreSQL when hosted) "
        "against the training reference. Rules: configs/pipeline.yaml; design: docs/MONITORING.md."
    )
    if st.button("Refresh metrics"):
        result = api_call("GET", "/metrics")
        alerts = api_call("GET", "/alerts")
        if result:
            # A status, not an application error: coloured text rather than an error box.
            colour = {"green": "green", "yellow": "orange", "red": "red"}.get(
                result["status"], "gray"
            )
            st.markdown(
                f"### :{colour}[Status: {result['status']}]  \n"
                "The worst level among the checks below."
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


def show_document(name, missing):
    text = read_document(name, language)
    if text is None:
        st.warning(missing)
    else:
        st.markdown(text)


with about:
    st.header("About / À propos")
    notice, card, privacy, provenance, architecture, version_tab = st.tabs(
        [
            "Customer notice / Avis",
            "Model card",
            "Privacy notice / Confidentialité",
            "Data provenance",
            "Architecture",
            "Version",
        ]
    )
    with notice:
        show_document("notice", "Customer notice missing from this installation.")
    with card:
        show_document("model_card", "MODEL_CARD.md missing from this installation.")
        show_ranking_plots()
    with privacy:
        show_document("privacy", "Privacy notice missing from this installation.")
    with provenance:
        show_document("provenance", "DATA_CARD.md missing from this installation.")
    with architecture:
        source = read_document("architecture")
        dot = mermaid_to_dot(source) if source else None
        if dot:
            st.markdown(DIAGRAM_CSS, unsafe_allow_html=True)  # a constant, not user input
            st.graphviz_chart(dot, width="stretch")
        st.caption("Source and rationale: ARCHITECTURE.md")
    with version_tab:
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
            "OpenAI may send the message and retrieved context to an external provider. "
            "Regex runs locally; Ollama requires its own configured server."
        )
        st.write(
            "Customer identity verification and enforced retention and deletion remain "
            "outside this academic demonstration."
        )
        st.caption("Evidence: reports/current_run/VALIDATION_RESULTS.md and REMAINING_ISSUES.md")
