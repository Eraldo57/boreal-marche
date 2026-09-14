# Automated-Decision Register

**Boréal Marché — Customer Intelligence Platform**
**Version:** 0.1.0 | **Last updated:** 2026-09-14

This register is required under Québec's Law 25 (s.12.1) for any decision based
exclusively on automated processing of personal information.

---

## Register Table

| # | Decision Point | System Component | What Is Decided | Personal Info Used | Personal Info NOT Used | Fully Automated? | Human Review Point | Customer Rights Mechanism | Risk Level | Frequency |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Retention risk scoring** | Head A — Prediction API (`/predict`) | Probability (0–1) that a customer will not return within 90 days; binary risk label (high/medium/low) | Purchase history: recency, frequency, monetary value, basket composition, cadence, cancellation/return rates, country | Name, email, phone, payment info, IP, demographics | **No** — score is a recommendation; coupon dispatch requires human approval | Marketing analyst reviews flagged-customer batch before any action | (1) "About" tab disclosure; (2) `/predict` returns `top_factors`; (3) Human review on request via support; (4) Data access/correction via support | Medium | Per-customer, on-demand or batch |
| 2 | **Product recommendation** | Head B — Recommendation API (`/recommend`) | Ranked list of up to 5 suggested products | Current basket contents (session-level, not stored) | Individual purchase history at recommendation time; personal identifiers | **Yes** — displayed in real time | Rule set reviewed by merchandising analyst before deployment; individual recommendations not individually reviewed | Firing rules returned in API response (`because` field) | Low | Per-request, real-time |
| 3 | **Policy question answering** | Head C — Assistant API (`/chat`) | Natural-language response to customer query, grounded in policy documents | Customer message text; optionally `customer_id` for tool calls | Personal identifiers beyond customer_id | **Partially** — guardrails enforce refusal and human handoff when confidence is low | Mandatory handoff to human agent when: (a) similarity below floor; (b) out-of-scope question; (c) cross-customer data requested | Human agent available for all questions; assistant discloses automation | Low | Per-message, real-time |
| 4 | **Drift alert** | Monitoring (`/metrics`) | Whether feature distributions have shifted beyond thresholds (PSI > 0.2, KS > 0.05) | Aggregated, anonymised feature statistics | Individual customer data | **Yes** — alert is logged automatically | ML engineer reviews alerts and decides whether to retrain | Alert is internal; not customer-facing | Low | Continuous |
| 5 | **Scheduled retraining** | `scripts/retrain.py` + CD workflow | Whether to propose a new model version | Aggregated training metrics | Individual customer data at decision time | **No** — retraining creates a PR; a human must review and merge | Engineer reviews MLflow comparison, approves PR | Not customer-facing | Weekly |

---

## Decision Flow Summary

```
Customer interaction
        │
        ├─→ /predict   ──→ Score ──→ [HUMAN REVIEW] ──→ Action (coupon)
        │
        ├─→ /recommend  ──→ Suggestions ──→ Displayed immediately
        │
        └─→ /chat       ──→ Answer ──→ Displayed (with guardrails + handoff)
                              │
                              └─→ [HUMAN HANDOFF if low confidence]
```

---

## Review Schedule

This register is reviewed:
- **Quarterly** by the project owner
- **On every model retrain** (to verify features and assumptions are still current)
- **When new decision points are added** to the platform

---

## Contact

For questions about automated decisions: contact customer support or the responsible
person identified in `governance/GOVERNANCE_LOI25.md`.
