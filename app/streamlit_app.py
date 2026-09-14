"""
Boréal Marché — Customer Intelligence Platform
Streamlit front end with five sections.

Launch:  streamlit run app/streamlit_app.py
"""

import json
import time

import httpx
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE = st.sidebar.text_input("API URL", value="http://localhost:8000")
CLIENT_TIMEOUT = 30.0

# ---------------------------------------------------------------------------
# Internationalisation — minimal EN / FR toggle
# ---------------------------------------------------------------------------
STRINGS = {
    "en": {
        "app_title": "🍁 Boréal Marché — Customer Intelligence",
        "tab_predict": "📊 Prediction",
        "tab_recommend": "🛒 Recommendations",
        "tab_chat": "💬 Assistant",
        "tab_monitor": "📈 Monitoring",
        "tab_about": "ℹ️ About",
        "predict_header": "Customer Retention Prediction",
        "predict_run": "Predict",
        "predict_prob": "Probability of return",
        "predict_risk": "Risk label",
        "predict_factors": "Top contributing factors",
        "recommend_header": "Basket Recommendations",
        "recommend_add": "Add product to basket",
        "recommend_run": "Get recommendations",
        "recommend_clear": "Clear basket",
        "recommend_because": "Because you added",
        "recommend_empty": "Your basket is empty. Add items above.",
        "chat_header": "Customer Support Assistant",
        "chat_placeholder": "Ask about delivery, returns, allergens, privacy…",
        "chat_citations": "Sources",
        "monitor_header": "System Monitoring",
        "monitor_requests": "Total requests",
        "monitor_errors": "Error rate",
        "monitor_p50": "Latency p50",
        "monitor_p95": "Latency p95",
        "monitor_drift": "Feature Drift Status",
        "about_header": "About This Platform",
        "about_provenance": "Data Provenance",
        "about_provenance_text": (
            "This application uses the UCI Online Retail II dataset, a public dataset "
            "re-skinned as Boréal Marché for a pedagogical capstone project at Vanier College. "
            "No real Boréal Marché customer data exists."
        ),
        "about_privacy": "Privacy Notice",
        "about_privacy_text": (
            "This system processes purchase history to predict retention risk, generate "
            "product recommendations, and answer policy questions. No personal data leaves "
            "your deployment. Under Québec's Law 25, you have the right to know how automated "
            "decisions are made, to request a human review, and to access or correct your data."
        ),
        "about_architecture": "Architecture",
        "lang_label": "Language / Langue",
        "provider_label": "LLM Provider",
        "sidebar_title": "Settings",
    },
    "fr": {
        "app_title": "🍁 Boréal Marché — Intelligence Client",
        "tab_predict": "📊 Prédiction",
        "tab_recommend": "🛒 Recommandations",
        "tab_chat": "💬 Assistant",
        "tab_monitor": "📈 Surveillance",
        "tab_about": "ℹ️ À propos",
        "predict_header": "Prédiction de rétention client",
        "predict_run": "Prédire",
        "predict_prob": "Probabilité de retour",
        "predict_risk": "Niveau de risque",
        "predict_factors": "Facteurs les plus importants",
        "recommend_header": "Recommandations de panier",
        "recommend_add": "Ajouter un produit au panier",
        "recommend_run": "Obtenir des recommandations",
        "recommend_clear": "Vider le panier",
        "recommend_because": "Parce que vous avez ajouté",
        "recommend_empty": "Votre panier est vide. Ajoutez des articles ci-dessus.",
        "chat_header": "Assistant de support client",
        "chat_placeholder": "Questions sur la livraison, retours, allergènes, vie privée…",
        "chat_citations": "Sources",
        "monitor_header": "Surveillance du système",
        "monitor_requests": "Requêtes totales",
        "monitor_errors": "Taux d'erreur",
        "monitor_p50": "Latence p50",
        "monitor_p95": "Latence p95",
        "monitor_drift": "État de dérive des caractéristiques",
        "about_header": "À propos de cette plateforme",
        "about_provenance": "Provenance des données",
        "about_provenance_text": (
            "Cette application utilise le jeu de données UCI Online Retail II, un jeu de "
            "données public habillé sous le nom Boréal Marché pour un projet de fin d'études "
            "au Collège Vanier. Aucune donnée réelle de Boréal Marché n'existe."
        ),
        "about_privacy": "Avis de confidentialité",
        "about_privacy_text": (
            "Ce système traite l'historique d'achat pour prédire le risque de rétention, "
            "générer des recommandations de produits et répondre à des questions de politique. "
            "Aucune donnée personnelle ne quitte votre déploiement. En vertu de la Loi 25 du "
            "Québec, vous avez le droit de savoir comment les décisions automatisées sont "
            "prises, de demander une révision humaine et d'accéder ou de corriger vos données."
        ),
        "about_architecture": "Architecture",
        "lang_label": "Language / Langue",
        "provider_label": "Fournisseur LLM",
        "sidebar_title": "Paramètres",
    },
}


def t(key: str) -> str:
    """Return the translated string for the current language."""
    return STRINGS.get(st.session_state.get("lang", "en"), STRINGS["en"]).get(key, key)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def api_post(endpoint: str, payload: dict) -> dict | None:
    """POST to the FastAPI backend. Returns parsed JSON or None on error."""
    try:
        with httpx.Client(base_url=API_BASE, timeout=CLIENT_TIMEOUT) as client:
            r = client.post(endpoint, json=payload)
            r.raise_for_status()
            return r.json()
    except httpx.HTTPStatusError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text}")
        return None
    except httpx.ConnectError:
        st.error(f"Cannot reach the API at {API_BASE}. Is the server running?")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


def api_get(endpoint: str) -> dict | None:
    """GET from the FastAPI backend."""
    try:
        with httpx.Client(base_url=API_BASE, timeout=CLIENT_TIMEOUT) as client:
            r = client.get(endpoint)
            r.raise_for_status()
            return r.json()
    except httpx.ConnectError:
        st.error(f"Cannot reach the API at {API_BASE}. Is the server running?")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Boréal Marché", page_icon="🍁", layout="wide")

with st.sidebar:
    st.header("⚙️ " + t("sidebar_title"))

    lang = st.selectbox(
        t("lang_label"),
        options=["en", "fr"],
        format_func=lambda x: "English" if x == "en" else "Français",
        index=0,
    )
    st.session_state["lang"] = lang

    provider = st.selectbox(
        t("provider_label"),
        options=["regex", "openai", "ollama"],
        index=0,
    )
    st.session_state["provider"] = provider

    st.divider()
    st.caption("Boréal Marché v0.1.0")

# ---------------------------------------------------------------------------
# Main title
# ---------------------------------------------------------------------------
st.title(t("app_title"))

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_predict, tab_recommend, tab_chat, tab_monitor, tab_about = st.tabs([
    t("tab_predict"),
    t("tab_recommend"),
    t("tab_chat"),
    t("tab_monitor"),
    t("tab_about"),
])

# ===========================================================================
# TAB 1 — Prediction (what-if sliders)
# ===========================================================================
with tab_predict:
    st.header(t("predict_header"))

    col_sliders, col_result = st.columns([2, 1])

    with col_sliders:
        st.subheader("Customer Features")
        feat = {}
        c1, c2, c3 = st.columns(3)
        with c1:
            feat["recency_days"] = st.slider("Recency (days)", 0, 365, 30)
            feat["frequency"] = st.slider("Frequency (orders)", 1, 200, 10)
            feat["monetary_total"] = st.slider("Monetary total ($)", 0, 50000, 2000, step=100)
            feat["monetary_mean"] = st.slider("Avg. order value ($)", 0, 5000, 200, step=10)
            feat["monetary_std"] = st.slider("Order value std ($)", 0, 2000, 80, step=10)
        with c2:
            feat["avg_basket_size"] = st.slider("Avg. basket size (items)", 1, 100, 12)
            feat["avg_basket_value"] = st.slider("Avg. basket value ($)", 0, 5000, 250, step=10)
            feat["max_basket_value"] = st.slider("Max basket value ($)", 0, 10000, 800, step=50)
            feat["avg_unique_items_per_basket"] = st.slider("Avg. unique items/basket", 1, 80, 8)
            feat["single_item_basket_ratio"] = st.slider("Single-item basket %", 0.0, 1.0, 0.15)
        with c3:
            feat["days_since_first_purchase"] = st.slider("Tenure (days)", 1, 730, 180)
            feat["avg_days_between_orders"] = st.slider("Avg. days between orders", 1, 365, 30)
            feat["n_unique_categories"] = st.slider("Unique categories", 1, 50, 8)
            feat["cancellation_rate"] = st.slider("Cancellation rate", 0.0, 1.0, 0.02)
            feat["weekend_purchase_ratio"] = st.slider("Weekend purchase %", 0.0, 1.0, 0.25)

        # Convert all to float for the API
        features = {k: float(v) for k, v in feat.items()}

    with col_result:
        if st.button(t("predict_run"), type="primary", use_container_width=True):
            result = api_post("/predict", {
                "customer_id": "WHAT-IF-001",
                "features": features,
            })
            if result:
                prob = result["probability"]
                st.metric(t("predict_prob"), f"{prob:.1%}")
                st.progress(prob)

                risk = result.get("risk_label", "unknown")
                emoji = {"high": "🔴", "medium": "🟠", "low": "🟢"}.get(risk, "⚪")
                st.metric(t("predict_risk"), f"{emoji} {risk.capitalize()}")

                if result.get("top_factors"):
                    st.subheader(t("predict_factors"))
                    for f in result["top_factors"]:
                        st.write(f"- **{f.get('feature', '?')}**: {f.get('importance', 0):.3f}")


# ===========================================================================
# TAB 2 — Basket & Recommendations
# ===========================================================================
with tab_recommend:
    st.header(t("recommend_header"))

    # Initialise basket in session state
    if "basket" not in st.session_state:
        st.session_state["basket"] = []

    col_add, col_basket = st.columns([2, 1])

    with col_add:
        new_item = st.text_input(t("recommend_add"), placeholder="e.g. WHITE HANGING HEART T-LIGHT HOLDER")
        c_add, c_clear = st.columns(2)
        with c_add:
            if st.button("➕ Add", use_container_width=True) and new_item.strip():
                st.session_state["basket"].append(new_item.strip().upper())
                st.rerun()
        with c_clear:
            if st.button(t("recommend_clear"), use_container_width=True):
                st.session_state["basket"] = []
                st.rerun()

    with col_basket:
        st.subheader("🧺 Basket")
        if st.session_state["basket"]:
            for i, item in enumerate(st.session_state["basket"]):
                st.write(f"{i+1}. {item}")
        else:
            st.info(t("recommend_empty"))

    st.divider()

    if st.session_state["basket"] and st.button(t("recommend_run"), type="primary"):
        result = api_post("/recommend", {
            "basket": st.session_state["basket"],
            "top_k": 5,
        })
        if result:
            st.subheader("✨ Recommended Products")
            for rec in result.get("recommendations", []):
                st.success(f"🏷️ {rec}")

            if result.get("because"):
                with st.expander(t("recommend_because") + "…"):
                    for rule in result["because"]:
                        ant = ", ".join(rule["antecedent"])
                        con = ", ".join(rule["consequent"])
                        st.write(
                            f"- {ant} → **{con}** "
                            f"(confidence {rule['confidence']:.0%}, lift {rule['lift']:.1f})"
                        )

            strategy = result.get("strategy", "unknown")
            if strategy == "popularity_fallback":
                st.caption("ℹ️ No matching rules — showing popular items instead.")
            elif strategy == "empty_basket":
                st.caption("ℹ️ Empty basket — showing trending items.")


# ===========================================================================
# TAB 3 — Assistant (chat)
# ===========================================================================
with tab_chat:
    st.header(t("chat_header"))

    # Initialise chat history
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    # Render history
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander(f"📎 {t('chat_citations')} ({len(msg['citations'])})"):
                    for cit in msg["citations"]:
                        st.caption(f"**{cit['source']}** (score {cit['score']:.2f})")
                        st.text(cit["chunk"][:300])

    # Input
    if user_input := st.chat_input(t("chat_placeholder")):
        st.session_state["chat_messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("…"):
                result = api_post("/chat", {
                    "message": user_input,
                    "provider": st.session_state.get("provider", "regex"),
                    "language": st.session_state.get("lang", "en"),
                })

            if result:
                reply = result.get("reply", "Sorry, I could not generate a response.")
                st.markdown(reply)

                citations = result.get("citations", [])
                assistant_msg = {"role": "assistant", "content": reply, "citations": citations}
                st.session_state["chat_messages"].append(assistant_msg)

                if citations:
                    with st.expander(f"📎 {t('chat_citations')} ({len(citations)})"):
                        for cit in citations:
                            st.caption(f"**{cit['source']}** (score {cit['score']:.2f})")
                            st.text(cit["chunk"][:300])

                if result.get("refused"):
                    st.warning(f"⚠️ {result.get('refusal_reason', 'Out of scope')}")
            else:
                fallback = "I couldn't reach the backend. Please check the API connection."
                st.warning(fallback)
                st.session_state["chat_messages"].append(
                    {"role": "assistant", "content": fallback}
                )


# ===========================================================================
# TAB 4 — Monitoring
# ===========================================================================
with tab_monitor:
    st.header(t("monitor_header"))

    if st.button("🔄 Refresh", key="refresh_monitor"):
        pass  # triggers rerun

    result = api_get("/metrics")

    if result:
        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(t("monitor_requests"), f"{result.get('request_count', 0):,}")
        k2.metric(t("monitor_errors"), f"{result.get('error_rate', 0):.2%}")
        k3.metric(t("monitor_p50"), f"{result.get('latency_p50_ms', 0):.0f} ms")
        k4.metric(t("monitor_p95"), f"{result.get('latency_p95_ms', 0):.0f} ms")

        # Traffic-light status
        status = result.get("status", "green")
        light = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(status, "⚪")
        st.subheader(f"Overall Status: {light} {status.upper()}")

        # Drift table
        drift_metrics = result.get("drift_metrics", [])
        if drift_metrics:
            st.subheader(t("monitor_drift"))
            for dm in drift_metrics:
                col_feat, col_stat, col_val, col_alert = st.columns([3, 1, 1, 1])
                col_feat.write(dm["feature"])
                col_stat.write(dm["statistic"].upper())
                col_val.write(f"{dm['value']:.4f}")
                if dm["alert"]:
                    col_alert.error("⚠️ ALERT")
                else:
                    col_alert.success("✅ OK")
        else:
            st.info("No drift data available yet. Send traffic to populate.")
    else:
        st.info("Connect to the API to view monitoring data.")


# ===========================================================================
# TAB 5 — About
# ===========================================================================
with tab_about:
    st.header(t("about_header"))

    # Model card summary
    st.subheader("🏷️ Model Card Summary")
    st.markdown("""
| Field | Value |
|---|---|
| **Model type** | HistGradientBoostingClassifier (calibrated) |
| **Task** | Binary classification — 90-day customer retention |
| **Primary metric** | PR-AUC (Average Precision) |
| **Training data** | UCI Online Retail II (re-skinned as Boréal Marché) |
| **Features** | 25 engineered features (RFM, basket, cadence, category, behavioural, geography) |
| **Threshold** | Profit-optimised, not default 0.5 |
| **Calibration** | Isotonic regression |
| **Framework** | scikit-learn, MLflow, Optuna |
    """)

    st.divider()

    # Data provenance
    st.subheader(f"📦 {t('about_provenance')}")
    st.info(t("about_provenance_text"))

    st.divider()

    # Privacy notice
    st.subheader(f"🔒 {t('about_privacy')}")
    st.warning(t("about_privacy_text"))

    st.divider()

    # Architecture
    st.subheader(f"🏗️ {t('about_architecture')}")
    st.markdown("""
```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Front End                       │
│  Prediction │ Recommendations │ Assistant │ Monitoring │ About │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP / JSON
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI (Pydantic v2)                    │
│  /predict  /predict/batch  /recommend  /chat  /metrics      │
│  /health   /ready          /version                         │
└──────┬──────────┬────────────┬───────────────┬──────────────┘
       │          │            │               │
       ▼          ▼            ▼               ▼
  ┌─────────┐ ┌────────┐ ┌──────────┐  ┌────────────┐
  │ Head A  │ │ Head B │ │  Head C  │  │ Monitoring │
  │Retention│ │ Rules  │ │   RAG    │  │   Drift    │
  │ Model   │ │ Engine │ │Assistant │  │  Detector  │
  │(joblib) │ │ (JSON) │ │(3 provs) │  │  (PSI/KS)  │
  └─────────┘ └────────┘ └──────────┘  └────────────┘
       │          │            │               │
       ▼          ▼            ▼               ▼
  ┌─────────────────────────────────────────────────┐
  │              MLflow Tracking Server              │
  │     Experiments · Model Registry · Artifacts     │
  └─────────────────────────────────────────────────┘
```
    """)

    st.divider()
    st.caption(
        "Boréal Marché is a pedagogical project for Vanier College course 420-977-VA. "
        "Built by Eraldo Ibrushi."
    )
