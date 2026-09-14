# Model Card — Boréal Marché Retention Model

*Following the format proposed by Mitchell et al. (2019), "Model Cards for Model Reporting."*

---

## Model Details

| Field | Value |
|---|---|
| **Model name** | `boreal-marche-retention-v0.1` |
| **Model type** | HistGradientBoostingClassifier (scikit-learn), calibrated with isotonic regression |
| **Framework** | scikit-learn 1.5+, wrapped in a scikit-learn Pipeline (preprocessing + estimator) |
| **Training date** | 2026-09-14 (initial training) |
| **Version** | 0.1.0 |
| **Developed by** | Eraldo Ibrushi, Vanier College (420-977-VA capstone project) |
| **Licence** | Educational use only — not for production deployment on real customer data |
| **Serialisation format** | joblib (chosen over pickle for: smaller file size, transparent dependency on numpy/scipy versions, and the compression option; chosen over ONNX because the model includes scikit-learn preprocessing that ONNX cannot represent without custom operators) |
| **MLflow run ID** | Recorded in `artifacts/metadata_sidecar.json` |

### Why not pickle?

Pickle executes arbitrary code on load. A malicious `.pkl` file can run shell commands
when loaded. Joblib shares this risk in principle, but the pipeline contains only
scikit-learn objects whose `__reduce__` methods are well-audited. For a true
zero-trust environment, ONNX export would be preferable, but it cannot represent the
full preprocessing pipeline.

---

## Intended Use

### Primary use
Identify customers of Boréal Marché who are at risk of not returning within 90 days, so
that the marketing team can target retention campaigns (coupons, re-engagement emails)
to the right customers instead of blanketing the entire customer base.

### Primary users
- Marketing analysts who review the flagged-customer list and approve coupon sends.
- The system is NOT intended for fully automated customer-facing decisions without human
  review (see governance/GOVERNANCE_LOI25.md).

### Out-of-scope uses
- **Medical, legal, financial, or hiring decisions.** This model predicts grocery purchase
  behaviour — nothing else.
- **Individual-level credit scoring or access control.**
- **Deployment on real customer data without privacy review.** The training data is a
  re-skinned public dataset; the model has not been validated on real Boréal Marché data.
- **Real-time automated actions** (e.g., blocking a customer, changing prices) without
  human review.

---

## Factors

### Relevant factors
- **Recency** of last purchase (strongest single predictor)
- **Frequency** of past orders
- **Monetary value** of purchases
- **Purchase cadence** regularity
- **Basket composition** (size, value, variety)
- **Geographic region** (domestic vs. international shipping effects)

### Factors that may affect performance
- **Seasonality:** The model was trained on 2009–2011 data. Holiday purchasing patterns may
  not generalise to other periods.
- **Geography:** The dataset is dominated by UK customers (~90%). Model performance may be
  lower for customers from under-represented regions.
- **Customer tenure:** Very new customers (< 2 orders) have sparse feature vectors, and
  the model may default to the base rate for them.

---

## Metrics

### Primary metric
- **PR-AUC (Average Precision):** Chosen because the dataset is imbalanced (more returners
  than churners). PR-AUC is more informative than ROC-AUC when the positive class is the
  minority.

### Secondary metrics
| Metric | Purpose |
|---|---|
| ROC-AUC | Threshold-independent discrimination |
| Brier score | Calibration quality (lower is better) |
| Recall at top decile | "If we can only contact 10% of customers, how many churners do we catch?" |
| Fit time | Practical constraint for retraining |

### Decision threshold
The threshold is NOT set at 0.5. It is selected by maximising expected profit:

```
Expected profit = (true positives × margin_retained) − (false positives × cost_coupon)
```

Cost assumptions:
- **Cost of sending a coupon:** $5.00
- **Average margin from a retained customer:** $25.00
- **Cost of missing a churner (false negative):** $0 (no coupon sent, no cost incurred)

The profit-optimal threshold is determined on the calibrated validation set and recorded
in the metadata sidecar.

---

## Evaluation Data

- **Dataset:** UCI Online Retail II (re-skinned as Boréal Marché)
- **Split:** 80% train / 20% test, stratified by target
- **Cross-validation:** 5-fold stratified CV on the training set
- **Temporal note:** The cutoff date (2011-09-01) creates a natural temporal split — the
  label window (2011-09-01 to 2011-11-30) does not overlap with the observation window.

---

## Training Data

- **Source:** UCI Online Retail II, approximately 1.07 million transactions from a UK-based
  online retailer, December 2009 – December 2011.
- **Size after cleaning:** ~400,000 transactions from ~4,300 customers.
- **Features:** 25 engineered features (see `configs/pipeline.yaml` for the full list).
- **Labelling:** Binary — did the customer place at least one order in the 90 days
  following the cutoff date?
- **Provenance:** This is a public dataset. It is NOT proprietary Boréal Marché data.
  See `governance/DATA_CARD.md` for the full provenance disclosure.

---

## Quantitative Analyses

### Spot-check results (cross-validated on training set)

| Model | PR-AUC | ROC-AUC | Brier | Fit time (s) |
|---|---|---|---|---|
| DummyClassifier (stratified) | ~baseline | 0.50 | — | <0.01 |
| Business heuristic (recency < 60d) | ~0.60 | ~0.65 | — | <0.01 |
| LogisticRegression | — | — | — | — |
| RandomForestClassifier | — | — | — | — |
| GradientBoostingClassifier | — | — | — | — |
| **HistGradientBoostingClassifier** | **—** | **—** | **—** | **—** |
| KNeighborsClassifier | — | — | — | — |
| LinearSVC | — | — | — | — |

*Note: Cells marked "—" are populated after training. This model card is a living document
updated with each retraining cycle.*

### Calibration
- Isotonic regression is applied after model selection.
- Reliability curves (before and after calibration) are logged as MLflow artifacts.

---

## Ethical Considerations

### Fairness
- A subgroup fairness audit is performed by region, tenure band, and spend quartile
  (see `governance/fairness_audit.md`).
- Groups with recall more than 5 percentage points below overall recall are flagged.
- Three candidate mitigations with trade-offs are documented.

### Privacy
- The model uses only transactional features. No demographic, biometric, or directly
  identifying information enters the model.
- Geographic data (`country_code`) is the most sensitive feature and is retained only for
  its legitimate business purpose (shipping-time effects on return rates).
- Full data-minimisation audit in `governance/GOVERNANCE_LOI25.md`.

### Environmental impact
- Training on a laptop CPU takes approximately 5–15 minutes.
- SageMaker HPO uses spot instances to reduce cost and carbon footprint.

### Dual use
- This model should not be repurposed for customer surveillance, discriminatory pricing,
  or any use that restricts customer access to goods or services.

---

## Caveats and Recommendations

1. **The training data is from 2009–2011.** Consumer behaviour has changed significantly.
   A model trained on this data should not be deployed in production without retraining
   on current data.

2. **The dataset is predominantly UK-based.** Performance on Québec customers is unknown
   and likely different (language, product preferences, shipping patterns).

3. **This is a pedagogical project.** The "Boréal Marché" scenario is fictional. The
   model card documents the methodology, not a real deployment.

4. **Threshold sensitivity.** The profit-optimal threshold is sensitive to the cost
   assumptions. Small changes in coupon cost or margin can shift the threshold
   significantly. The business should review these assumptions quarterly.

5. **Delayed ground truth.** The retention label is only observable 90 days after the
   cutoff. This means the monitoring dashboard cannot detect model degradation in real
   time — it can only detect input drift. The design accounts for this with a backfill
   mechanism documented in the monitoring policy.
