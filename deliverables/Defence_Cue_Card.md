# Defence cue card — current model 0.3.0

Use the PowerPoint speaker notes; do not memorise every decimal. Twenty slides are grouped by Head. Keep each Head’s method, evaluation and limitations together.

| Section | What to say | Streamlit proof |
|---|---|---|
| Purpose | Three tasks: purchase prediction, basket completion, grounded policy assistance. | Prediction / Basket / Assistant |
| Data | UK giftware supports a fictional Québec demo. Six dated cutoffs across five stages. | About / data card |
| Head A | AP **0.760 vs 0.671** recency; ROC **0.786**; Brier **0.1940**. | About → Model card → four evaluation tabs |
| Decision | **20% theoretical**, **20.5% June optimum**; costs **5/25 assumed**; no coupon effect measured. | Prediction → Coupon math |
| Head B | White holder A → red holder B: **23.1% confidence**, **4.70 lift**, **3.54% support**. | Basket → Get recommendations → rule explorer |
| Head B test | Hit@5 **48.7% vs 17.2%**, rule-fire **83.7%**, catalogue **7.7%**. | Basket → Mining explanation → Measured conclusions |
| Algorithms | Same 3,567 rule keys. Apriori **1.22 s**, FP-Growth **64.45 s** in this measured run. | Basket → Apriori vs FP-Growth |
| Head C | Routing **97.5 / 95 / 92.5%** is not written-answer correctness. | Assistant → provider selector |
| Governance | 25 engineered; **12 served**. Thin-history gaps remain. Human review; no automatic dispatch. | Human review / About |
| Engineering | Explain tests and version evidence; distinguish local containers from cloud deployment. | Monitoring / About → Version |

At 20.5%, model recall is **82.42%** versus **81.93%** for equal-count recency. Ranking is the
clearer gain. AP interval: **0.742–0.775**. The test period was reused; no fresh-holdout claim.
Sigmoid was selected on June Brier; March fits calibration, June chooses it, September tests it.

French demo: “Combien coûte la livraison standard ?” Set French explicitly. Do not promise
identical answers from the three providers. Historical answer review: OpenAI 11/18 correct,
Ollama 3/18; one AI rater and development questions, not independent validation.

If challenged: “That is an assumption in the prototype. I would validate it with a coupon/control
experiment and a new future-period holdout before treating it as a business result.”

Current slide order: overview1–3; Head A4–10; Head B11–14; Head C15–17; engineering18; conclusion19; sources20.
