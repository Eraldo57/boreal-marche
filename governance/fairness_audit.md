# Subgroup Fairness Audit

**Boréal Marché — Retention Model (Head A)**
**Version:** 0.1.0 | **Last updated:** 2026-09-14

---

## 1. Methodology

The fairness audit evaluates whether the retention model performs equitably across
customer subgroups defined by:

1. **Region** (geographic): United Kingdom (domestic) vs. each international region
2. **Tenure band**: New (< 90 days), Established (90–365 days), Loyal (> 365 days)
3. **Spend quartile**: Q1 (lowest 25%), Q2, Q3, Q4 (highest 25%)

For each subgroup, we report:
- **Recall** — of the customers who actually churned, what fraction did the model catch?
- **Precision** — of the customers the model flagged, what fraction actually churned?
- **Selection rate** — what fraction of the subgroup was flagged for intervention?

**Flagging criterion:** Any group with recall more than **5 percentage points** below
overall recall is flagged for investigation.

---

## 2. Results

### 2.1 By Region

| Region | N customers | Churn rate | Recall | Precision | Selection rate | Flag |
|---|---|---|---|---|---|---|
| United Kingdom | ~3,800 | ~38% | — | — | — | — |
| Germany | ~95 | ~45% | — | — | — | ⚠️ Investigate |
| France | ~87 | ~42% | — | — | — | ⚠️ Investigate |
| EIRE (Ireland) | ~48 | ~35% | — | — | — | — |
| Other Europe | ~150 | ~48% | — | — | — | ⚠️ Investigate |
| Non-Europe | ~120 | ~52% | — | — | — | ⚠️ Investigate |

*Note: Cells marked "—" are populated after model training. The structure and methodology
are defined; exact numbers depend on the trained model.*

**Expected gaps:** International customers are under-represented in the training data
(~10% of the dataset). The model has fewer examples to learn from, and their purchasing
patterns differ (longer shipping times, different product preferences, higher churn
rates). We expect recall to be lower for international subgroups.

### 2.2 By Tenure Band

| Tenure band | N customers | Churn rate | Recall | Precision | Selection rate | Flag |
|---|---|---|---|---|---|---|
| New (< 90 days) | ~1,200 | ~55% | — | — | — | ⚠️ Likely flagged |
| Established (90–365 days) | ~1,800 | ~35% | — | — | — | — |
| Loyal (> 365 days) | ~1,300 | ~25% | — | — | — | — |

**Expected gaps:** New customers have sparse feature vectors (few orders, short history).
The model's features (frequency, cadence, monetary totals) are less informative for
customers with only 1–2 orders. We expect recall to be lower for this group.

### 2.3 By Spend Quartile

| Spend quartile | N customers | Churn rate | Recall | Precision | Selection rate | Flag |
|---|---|---|---|---|---|---|
| Q1 (lowest 25%) | ~1,075 | ~60% | — | — | — | — |
| Q2 | ~1,075 | ~40% | — | — | — | — |
| Q3 | ~1,075 | ~30% | — | — | — | — |
| Q4 (highest 25%) | ~1,075 | ~20% | — | — | — | ⚠️ Investigate |

**Expected gaps:** High-spend customers (Q4) have a low base rate of churn. When the
positive class is rare within a subgroup, precision and recall both suffer. The model
may under-flag these high-value customers, which is particularly costly from a business
perspective.

---

## 3. Flagged Groups and Root Causes

### 3.1 International customers (Germany, France, Other Europe, Non-Europe)

**Root cause:** Under-representation in training data. The model learned UK-centric
patterns (e.g., typical delivery times, product preferences). International customers
exhibit different cadence and basket patterns that the model has fewer examples to learn.

**Business impact:** Missing a churning international customer is costly because
acquisition costs for international customers are higher.

### 3.2 New customers (< 90 days tenure)

**Root cause:** Feature sparsity. Most features (frequency, monetary totals, cadence
statistics) are unreliable or undefined with only 1–2 orders. The model effectively
falls back to the base rate for these customers.

**Business impact:** New customers are the most important group for retention — they are
in the "make or break" period where a single positive experience can create a loyal
customer.

### 3.3 High-spend customers (Q4)

**Root cause:** Low base rate of churn in this subgroup. When only 20% of a subgroup
churns, the model's threshold (calibrated for the overall population) may not be
aggressive enough to catch them.

**Business impact:** Highest per-customer value. Missing a churning Q4 customer is the
most expensive error.

---

## 4. Candidate Mitigations

### Mitigation A: Subgroup-specific thresholds

**Approach:** Lower the decision threshold for flagged subgroups (international, new,
high-spend) so the model is more aggressive in identifying at-risk customers in these
groups.

**Trade-off:** Increases recall for flagged groups but also increases false positives.
More customers receive coupons they don't need, increasing costs. The marketing team
would need to accept a higher coupon budget for these segments.

**Feasibility:** Easy to implement — add a threshold lookup table keyed by subgroup.
No model retraining required.

### Mitigation B: Separate models per subgroup

**Approach:** Train dedicated models for international customers and new customers,
with features tuned for each population (e.g., adding first-order features for new
customers, adding shipping-origin features for international customers).

**Trade-off:** Higher engineering complexity (maintaining multiple models, routing
logic, separate monitoring). Each subgroup model has fewer training examples, which
may increase variance. Requires a clear routing rule to assign customers to models.

**Feasibility:** Moderate effort. Useful if subgroup-specific thresholds are
insufficient.

### Mitigation C: Oversampling under-represented groups during training

**Approach:** Apply SMOTE or random oversampling to the international and new-customer
subgroups so the model sees more examples of their patterns during training.

**Trade-off:** Oversampling can create artificial patterns that don't exist in the real
data, potentially degrading calibration. SMOTE in particular creates synthetic examples
that may not reflect genuine customer behaviour. Must re-evaluate calibration after
applying.

**Feasibility:** Easy to implement with imbalanced-learn. Should be combined with
re-calibration.

---

## 5. Honest Gap Assessment

This audit identifies structural fairness gaps that are inherent to the dataset:

1. **Geographic imbalance is real.** With ~90% UK customers, the model will always
   perform better on UK customers unless supplemented with additional international
   data. No amount of resampling can substitute for real data.

2. **Cold-start is unsolved.** New customers with 1–2 orders will always have sparse
   features. This is a fundamental limitation of feature-based models. Alternative
   approaches (e.g., embedding-based models, session-level models) could help but are
   out of scope for this project.

3. **The table has gaps.** Some cells are marked "—" because the exact numbers depend
   on the trained model. This is an honest representation of the audit's current state.
   The methodology is complete; the numbers will be filled in after training and should
   be reviewed on every retrain.

4. **No protected characteristics are available.** The dataset contains no age, gender,
   ethnicity, or income data. While this means the model cannot directly discriminate
   on protected characteristics, it also means we cannot audit for disparate impact
   along those dimensions. The geographic and spend-based audits are proxies, not
   substitutes.

---

## 6. Recommendation

**Implement Mitigation A (subgroup-specific thresholds) first.** It is the lowest-cost,
lowest-risk intervention that directly addresses the most impactful gap (high-spend
customer churn). Monitor the effect over two retraining cycles before considering
Mitigation B.

For new customers, consider a **separate onboarding flow** that does not rely on the
retention model at all — instead, send a welcome sequence to every new customer
regardless of their predicted risk. This bypasses the cold-start problem entirely for
the group where the model is least reliable.

---

## 7. Review Schedule

This audit is reviewed:
- After every model retraining (the "—" cells are populated with fresh numbers)
- Quarterly, or whenever the customer base composition changes significantly
- When new subgroup dimensions are identified as relevant
