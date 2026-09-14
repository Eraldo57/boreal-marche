# Governance — Loi 25 / Law 25 Compliance

**Boréal Marché — Customer Intelligence Platform**
**Version:** 0.1.0
**Last updated:** 2026-09-14
**Responsible person (Loi 25 s.3.1):** Eraldo Ibrushi

---

## 1. Automated-Decision Register

Québec's Law 25 (formerly Bill 64) requires that any enterprise using personal information
to make a decision based exclusively on automated processing must inform the individual
concerned, at or before the time the decision is made.

The following table documents every automated decision point in the Boréal Marché platform.

### 1.1 Decision Point: Retention Prediction (Head A)

| Field | Details |
|---|---|
| **Decision** | Whether a customer is at risk of churning (not returning within 90 days) |
| **What is decided** | A retention-risk score (0–1 probability) and a risk label (high / medium / low) |
| **What action follows** | Customers above the profit-optimised threshold are flagged for a retention coupon campaign |
| **Personal information used** | Purchase history (recency, frequency, monetary value), basket composition, geographic region (country), purchase cadence, cancellation/return rates |
| **Personal information NOT used** | Name, email, phone, payment method, IP address, device information, demographic data (age, gender, ethnicity) |
| **Is the decision fully automated?** | **No.** The model produces a score and a recommendation. A human marketing analyst reviews the flagged list before any coupon is sent. See Section 2 below. |
| **Human review step** | The flagged-customer list is surfaced on the dashboard. A marketing analyst reviews, approves, or overrides each batch before dispatch. No coupon is sent without human approval. |
| **How a customer exercises their rights** | A customer may: (1) Request to know that an automated decision was made — the "About" section of the application discloses this; (2) Request the principal factors behind the decision — the API returns `top_factors` with each prediction; (3) Request human review — the customer contacts support, and a human re-evaluates; (4) Correct their data — contact support to update or delete records. |
| **Retention period** | Prediction logs are retained for 12 months for monitoring and audit, then deleted. |

### 1.2 Decision Point: Product Recommendation (Head B)

| Field | Details |
|---|---|
| **Decision** | Which products to suggest to a customer based on their current basket |
| **What is decided** | A ranked list of up to 5 recommended products |
| **What action follows** | Recommendations are displayed in the "you might also like" section of the interface |
| **Personal information used** | Current basket contents only. Association rules are mined from aggregated, anonymised transaction patterns — no individual purchase history is used at recommendation time. |
| **Is the decision fully automated?** | **Yes**, but the decision is a product suggestion, not a consequential decision affecting the customer's rights or interests. It is analogous to shelf placement. |
| **Human review step** | The rule set is reviewed by a merchandising analyst before deployment. Individual recommendations are shown in real time and not editable per-request. |
| **How a customer exercises their rights** | A customer may request to see the rules that produced a recommendation (the API returns the firing rule with confidence, lift, and support). |
| **Retention period** | Recommendation logs are retained for 6 months. |

### 1.3 Decision Point: Customer Support Assistant (Head C)

| Field | Details |
|---|---|
| **Decision** | How to answer a customer's natural-language question about policies, products, or their account |
| **What is decided** | A grounded text response with citations to company policy documents |
| **What action follows** | The answer is displayed to the customer. If the question cannot be answered from the corpus, the assistant refuses and offers to hand off to a human agent. |
| **Personal information used** | The customer's message text. If a `customer_id` is provided, the assistant may call the retention and recommendation tools, which use purchase history. |
| **Is the decision fully automated?** | **Partially.** The assistant generates responses, but it is designed to refuse out-of-scope questions and to hand off to human support when confidence is low. It never modifies customer data. |
| **Human review step** | The assistant includes a mandatory human-handoff path: when the similarity score is below the floor, or when the question is classified as out-of-scope, the response is "I cannot answer this — let me connect you with a human agent." |
| **Guardrails** | (1) Out-of-scope questions are refused; (2) Answers below the similarity floor are not returned; (3) The LLM never produces a probability or a product ID from its own weights — it must call a tool; (4) Cross-customer data is never disclosed; (5) Every answer includes citations. |
| **How a customer exercises their rights** | Contact support to request human review of any automated answer. |
| **Retention period** | Chat logs are retained for 3 months. |

---

## 2. Design Consequence — Human-in-the-Loop

Because the retention prediction (Head A) is a decision based on automated processing of
personal information, Loi 25 s.12.1 requires that the individual be informed and given the
right to present observations and request a review by a natural person.

### Where the human approval step is inserted

```
Customer data  →  Feature pipeline  →  Model inference  →  Score + risk label
                                                                    │
                                                                    ▼
                                                        ┌─────────────────┐
                                                        │  HUMAN REVIEW   │
                                                        │  Marketing team │
                                                        │  reviews batch  │
                                                        │  before action  │
                                                        └────────┬────────┘
                                                                 │
                                                                 ▼
                                                          Coupon sent
                                                       (only if approved)
```

**Why this design:**
- The model score is a recommendation, not a final decision.
- The marketing analyst can override, exclude customers, or adjust the threshold.
- No coupon is ever sent by the system alone.
- This satisfies Loi 25's requirement for human review of automated decisions.

### Disclosure

The application's "About" tab displays a plain-language notice in both French and English
explaining:
1. That automated processing is used.
2. What personal information feeds the model.
3. How to request a human review.
4. How to access or correct personal data.

---

## 3. Data-Minimisation Audit

Loi 25 requires that only the personal information necessary for the stated purposes be
collected and used. The following audit examines every feature used by the model.

| Feature | Category | Necessary? | Justification |
|---|---|---|---|
| `recency_days` | RFM | ✅ Yes | Core predictor of retention — recent buyers are more likely to return |
| `frequency` | RFM | ✅ Yes | Number of past orders is directly predictive |
| `monetary_total` | RFM | ✅ Yes | Total spend indicates engagement depth |
| `monetary_mean` | RFM | ✅ Yes | Average order value differentiates customer segments |
| `monetary_std` | RFM | ✅ Yes | Variability in spending is a churn signal |
| `days_since_first_purchase` | Cadence | ✅ Yes | Tenure is a known retention factor |
| `avg_days_between_orders` | Cadence | ✅ Yes | Purchase rhythm predicts next-purchase timing |
| `std_days_between_orders` | Cadence | ✅ Yes | Irregular timing signals disengagement |
| `is_one_time_buyer` | Cadence | ✅ Yes | Strong negative predictor of return |
| `avg_basket_size` | Basket | ✅ Yes | Larger baskets correlate with loyalty |
| `avg_basket_value` | Basket | ✅ Yes | Higher value baskets indicate committed customers |
| `max_basket_value` | Basket | ✅ Yes | Peak spend indicates willingness-to-pay |
| `avg_unique_items_per_basket` | Basket | ✅ Yes | Variety in basket indicates exploration |
| `single_item_basket_ratio` | Basket | ✅ Yes | High ratio signals low engagement |
| `top_category_share` | Category | ✅ Yes | Category concentration indicates purchase pattern |
| `n_unique_categories` | Category | ✅ Yes | Breadth of interest predicts retention |
| `category_entropy` | Category | ✅ Yes | Information-theoretic measure of purchase diversity |
| `has_ever_cancelled` | Behavioural | ✅ Yes | Past cancellations predict future disengagement |
| `cancellation_rate` | Behavioural | ✅ Yes | Rate is more informative than binary flag |
| `return_rate` | Behavioural | ✅ Yes | Distinguishes quality dissatisfaction from general churn |
| `weekend_purchase_ratio` | Behavioural | ✅ Yes | Temporal pattern differentiates customer types |
| `morning_purchase_ratio` | Behavioural | ✅ Yes | Same reasoning — minimal personal sensitivity |
| `country_code` | Geography | ⚠️ Caution | Used to control for shipping-time effects. Encoded as a numeric code, not stored as personally identifiable location data. Retained because international vs. domestic shipping meaningfully affects return rates. |
| `is_domestic` | Geography | ✅ Yes | Binary simplification of geography — minimal privacy impact |

### Features explicitly NOT used (and why)

| Feature | Why excluded |
|---|---|
| Customer name | Not predictive; personally identifiable |
| Email address | Not predictive; personally identifiable |
| Phone number | Not predictive; personally identifiable |
| Payment method | Sensitive financial data; not necessary for retention prediction |
| IP address | Personally identifiable; not relevant |
| Device/browser info | Not available in transaction data; would be surveillance |
| Age / gender / ethnicity | Demographic data not available and would introduce fairness risks |
| Individual product descriptions | Too granular; category-level features suffice |
| Free-text order notes | Unstructured personal content; not necessary |

### Conclusion

All 24 features used by the model are derived from transactional behaviour (what was
purchased, when, and how much). No directly identifying information (name, email, phone)
enters the model. The geographic feature (`country_code`) is the most sensitive and is
retained only because it controls for a legitimate business factor (shipping logistics).

---

## 4. Retention and Deletion Policy

| Data type | Retention period | Deletion method |
|---|---|---|
| Raw transaction data | Duration of service + 12 months | Secure deletion from storage |
| Feature table | Duration of model lifecycle | Overwritten on retrain |
| Prediction logs | 12 months | Automated purge job |
| Recommendation logs | 6 months | Automated purge job |
| Chat logs | 3 months | Automated purge job |
| Model artifacts | Indefinite (for audit trail) | Versioned in MLflow; old versions archived |

---

## 5. References

- Loi 25: [Loi modernisant des dispositions législatives en matière de protection des renseignements personnels](https://www.publicationsduquebec.gouv.qc.ca/fileadmin/Fichiers_client/lois_et_reglements/LoisAnnuelles/fr/2021/2021C25F.PDF)
- Commission d'accès à l'information du Québec: [cai.gouv.qc.ca](https://www.cai.gouv.qc.ca/)
- EU AI Act (for reference; not directly applicable): [artificialintelligenceact.eu](https://artificialintelligenceact.eu/)
