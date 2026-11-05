# Defence script — grouped by Head

Twenty slides, grouped so each Head includes its method, evaluation and limitations.
Use the PowerPoint speaker notes in Presenter View and the cue card for short reminders.

| Slides | Section |
|---|---|
| 1–3 | Purpose, architecture and data overview |
| 4–10 | Head A: chronology, ranking, calibration, utility, threshold, fairness and monitoring |
| 11–14 | Head B: worked rule, thresholds, algorithms and evaluation |
| 15–17 | Head C: workflow, guardrails and answer evaluation |
| 18 | Engineering and delivery status |
| 19–20 | Conclusion and source references |

GitHub/AWS integration code is ready, awaiting owner sign-in and actual execution.
The local implementation and saved scientific results are unchanged by this regrouping.

## Slide 1: Three customer questions. One inspectable prototype.

Opening · 40 seconds. This is Boréal Marché, an academic customer-intelligence prototype. It brings together a purchase-probability model, basket association rules and a policy assistant. I will show what each head does, what the measured results mean, and the limits that remain. The historical data is UK giftware, while the Québec business is fictional. This is not a live campaign, and the interface sends no coupons. The central theme is that a prediction, an association and a generated answer each need different evidence. The presentation now completes Head A before moving to Head B, then Head C, and ends with engineering status.

## Slide 2: Offline artifacts support three online paths

Architecture · 55 seconds. Offline processing creates trusted, versioned artifacts: the retention estimator, mined rules and policy index. Online, Streamlit calls FastAPI rather than loading models itself. The retention path uses the estimator, recommendations use an antecedent index, and the assistant retrieves policy text or calls typed tools. Prediction logs feed monitoring. Local Docker Compose separates the API, UI, database and Ollama. Health means the process is running; readiness checks the loaded dependencies. The current API serves the local candidate artifact, and version reports that fallback. Cloud deployment is a separate delivery step.

## Slide 3: Six cutoffs across five chronological stages

Data and chronology · 65 seconds. There are six dated snapshots, not five snapshots: two training cutoffs and four later cutoffs for validation, calibration, decision and test. That creates five stages. Each customer’s features use only events before the cutoff. The label asks whether a positive purchase occurs during the next 90 days, with mature labels before the following stage. Customers may recur, so this is chronological evaluation, not unseen-customer evaluation. We still engineer and document all 25 features, but the model now uses 12 after the original June minimisation audit. The September test period has been reused, and I disclose that rather than calling it new.

## Slide 4: Calibration fitting and selection use different periods

Validation selects model family and parameters. The estimator is then refit on train plus validation before calibrators fit on March data. June selects calibration and threshold. September evaluates that locked protocol. The feature-removal decision used June and was archived before the new September result. Reusing a previously inspected test period limits independence even when the new protocol is not selected from its results. Demo: About → Model card; chronological split table in the saved report.

## Slide 5: The clearest gain is in ranking returners

Ranking · 65 seconds. Head A predicts a purchase within 90 days. Average precision is 0.760 against 0.671 for a recency ranking and 0.431 for a stratified dummy. We call it PR-AUC in the app, but the implementation is sklearn average precision. ROC-AUC is also reported, at 0.786. AP is primary because it keeps the precision of selected returners visible as recall changes. It is not because ROC is a false-negative metric: ROC uses true-positive and false-positive rates. A paired customer bootstrap gives the AP interval shown. These intervals condition on this selected model and one historical test period, not future performance or model-selection uncertainty. Demo: About → Model card → Precision–Recall and ROC.

## Slide 6: Calibration is selected before the test period

Calibration · 55 seconds. A probability near 80% should correspond to roughly 80% returners among comparable customers. March fits the calibration mappings, and June chooses between raw, isotonic and sigmoid predictions. Sigmoid has the lowest June Brier score, 0.15631. Brier is the mean squared difference between probability and outcome; it is not an error percentage or calibration alone. The later September Brier is 0.19398. June and September contain different customer-period outcomes, so those numbers do not establish a controlled before-and-after improvement. Streamlit About, Model card now shows the calibration plot, the Brier comparison and the threshold diagnostics. Demo: About → Model card → Calibration.

## Slide 7: 20.5% is an assumed-utility policy

Coupon policy · 85 seconds. Let p be the purchase probability, C the offer cost, M the retained margin, and F the missed-return cost. The coded scenario compares p times M minus C with minus p times F. Under these assumptions, the mathematical break-even is C divided by M plus F: 5 divided by 25, or 20%. June’s empirical sweep selected 20.5%. The values 5 and 25 remain placeholders. This does not measure the effect of sending a coupon, because it does not estimate purchases without the offer. Giving the same offer at 23% and 90% is therefore a simplified policy, not an optimal causal decision. At this threshold recall is 82.42% against 81.93% for equal-count recency, a small binary gain. Next comes a coupon/control experiment and budget-aware incremental-profit policy. Demo: Prediction → Coupon math.

## Slide 8: The binary gain over recency is small

Selected means probability at least 20.5%, not a measured coupon responder. The confusion matrix is based on actual later purchases. There are 2,236 actual returners and 2,997 non-returners. The equal-count recency comparison selects the same 3,234 customers by most recent purchase. It nearly matches recall while the forest has stronger ranking across thresholds. The bootstrap conditions on the fixed model and historical period. Demo: About → Model card → Threshold and uncertainty.

## Slide 9: Inputs were reduced; fairness gaps remain

Governance · 65 seconds. The original June audit justified removing 13 inputs jointly, including location inputs and duplicate measures. The 25 engineered features remain documented offline; only 12 are served under contract 2.0. A new audit of those 12 shows why individual ablation is not enough: nine individually inconclusive signals lose AP when all are removed together. Five strong recall disparities remain among thin-history customers. Removing location does not prove fairness. The current operational mitigation is human review and no automatic dispatch, not a statistical repair. Reviewer identity is still declared rather than authenticated. MLflow version 2 remains a candidate and is not promoted without the owner acknowledging these flags. Demo: Prediction → human review; About → customer notice.

## Slide 10: Performance evidence arrives after the label matures

Monitoring · 45 seconds. Right after scoring, we can monitor latency, volume, missing values and feature drift. We cannot know 90-day return performance immediately. The documented backfill design waits until the label window matures, joins purchase outcomes by customer and scoring cutoff, checks label completeness and then computes ranking, probability quality and subgroup recall. Drift is an alert for investigation, not proof of accuracy loss. Local alert logging is implemented. Delayed-label backfill and external alert delivery remain design or integration work. Retraining should produce a reviewed proposal rather than automatically replacing the champion. Demo: Monitoring → Refresh metrics.

## Slide 11: White holder → red holder

Rule example · 65 seconds. If the visible basket contains the white hanging heart t-light holder, one learned suggestion is the red hanging heart t-light holder. A is the visible antecedent; B is the suggested consequent. About 23.1% of training baskets containing the white holder also contain the red holder. That is the confidence. The red holder appears about 4.7 times as often conditional on the white one as it does overall: that is lift. The pair appears in 3.54% of all training baskets: that is support. This describes association, not a causal effect. On Streamlit I can search either product in all 3,567 rules and show the exact A, B and denominators. Demo: Basket → select white hanging heart t-light holder → Get recommendations.

## Slide 12: A rule must clear three different tests

A and B are itemsets. A union B is the combined product set whose co-occurrence is counted in support; it is not the probability of either event. Confidence is the share of antecedent baskets that also contain the full consequent. Lift divides that conditional frequency by consequent frequency overall. The support/confidence grid was selected on validation; lift and itemset cap were fixed. The measured discarded low-lift rules do not include a genuinely high-confidence example, so we do not fabricate one. Demo: Basket → Mining explanation → selected thresholds and learned rules.

## Slide 13: FP-Growth serves; Apriori was faster to mine here

Algorithms and serving · 50 seconds. Both algorithms operate on the same sparse transaction matrix and produce the same 3,567 rule keys under the selected settings. We demonstrate Apriori and export the FP-Growth rules, but the measured bounded run was faster with Apriori: 1.22 seconds versus 64.45 seconds. I would not claim a generic speed advantage overrides that evidence. Sparse storage is 2.71 MB against about 120 MB dense equivalent. Online serving does not mine again. It looks up candidate rules through an antecedent inverted index. The local engine p95 is 0.222 milliseconds; this excludes HTTP, retrieval and Streamlit overhead. Demo: Basket → Mining explanation → Apriori vs FP-Growth.

## Slide 14: Better recovery, but a narrow catalogue

Basket evaluation · 65 seconds. We hide roughly half the distinct items in each eligible later basket. A hit means at least one of the five suggestions is among the hidden items. The system hits in 48.7% of 5,884 baskets, versus 17.2% for popularity. It does not mean one correct item in every five suggestions, or a 48.7% online sales increase. Single-item baskets cannot support this masking task and are excluded. Rule firing is 83.7% of evaluated baskets. Catalogue coverage is 7.7%, or 360 distinct suggested descriptions out of 4,687. The denominators differ. Low catalogue coverage does not mean most baskets fall back. Streamlit labels an unmatched basket as popularity fallback. Demo: Basket → Mining explanation → Measured conclusions.

## Slide 15: The assistant routes before it answers

Assistant design · 65 seconds. Head C first selects a typed tool. Policy search retrieves authored sources; retention score calls the saved model; recommendations call the rule engine; product lookup searches the catalogue. The language model does not invent those model probabilities or rule scores. For policy questions, retrieved sections support generated prose and citations. Regex is a pattern-based baseline, while OpenAI and Ollama offer alternative routing and generation providers. The index uses multilingual embeddings, but that alone does not guarantee correct French answers. Low similarity causes refusal and a bilingual handoff explicitly marked as fictional. All this remains an academic demo without authenticated customer accounts. Demo: Assistant → select provider and language; try the customer-score tool.

## Slide 16: Retrieval similarity is a routing heuristic

The thresholds control retrieval behaviour, not model confidence in truth. A high similarity can still retrieve an incomplete or incorrect context, and a generator can add unsupported claims. Bilingual refusal/handoff paths now have explicit tests. The support contact is fictional because this is an academic demonstration. Historical provider metrics predate this small handoff addition and do not substitute for a new independent answer audit. Demo: Assistant → show refusal or caution and citations.

## Slide 17: Correct routing does not ensure correct prose

Assistant evaluation · 65 seconds. On the 40 development questions, routing accuracy is 97.5% for regex, 95% for OpenAI and 92.5% for Ollama. Every provider refused the eight out-of-scope cases. But tool choice and answer correctness are different tasks. In the historical written-answer review, only 11 of 18 OpenAI answers and three of 18 Ollama answers were marked correct. The report identifies a single AI rater, not an independent human panel. The questions also informed development. I therefore treat these as diagnostic results, not deployment certification. French is a visible weakness. I would collect new questions and independent bilingual judgements before claiming reliable policy assistance. Demo: Assistant → ask “Combien coûte la livraison standard ?” with French selected.

## Slide 18: Local evidence is strong; cloud evidence is pending

Engineering and delivery status. The project has working local code and recorded local test evidence. GitHub workflows and AWS integration code are prepared, awaiting the owner’s sign-in and actual cloud execution. Do not present those as green remote CI or completed deployment. This slide distinguishes implemented local work from outstanding external evidence. Demo: About → Version; local verification log.

## Slide 19: Three heads, three kinds of evidence

Conclusion. Head A improves historical purchase ranking, but coupon incrementality remains unmeasured. Head B improves hidden-item recovery through basket associations, while catalogue coverage is narrow. Head C makes policy sources and typed-tool results inspectable, but written-answer correctness still needs independent bilingual validation. Each Head has now been shown together with its evaluation and limitations. GitHub and AWS code remain ready for sign-in; no cloud execution is claimed.

## Slide 20: Every headline maps to a saved source

Source reference for questions. Every headline maps to a saved report or code module. This is the only end-of-deck reference slide; the technical explanations and evaluations are now grouped with their owning Head. Local reports under reports/current_run provide the exact metrics and dates. The model metadata contains hashes, dirty source state, library versions and the selected feature contract. The old model is preserved under reports/archive. The source data is UCI Online Retail II, DOI 10.24432/C5CG6D. Primary privacy references are CAI private-enterprise automated-decision guidance and the private-sector statute section 12.1; these are design references, not a compliance certification. AI tools assisted implementation, verification and preparation; the owner should review and explain the final work.
