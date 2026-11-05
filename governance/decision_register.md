# Automated-decision inventory — implementation and plans

Updated 2026-11-01. An engineering inventory, not evidence of legal compliance.

| Component | Actual output | Actual downstream action | Missing controls |
|---|---|---|---|
| Retention | P(will_return in 90 days), thresholded return decision, inverse risk label | A proposed action (offer a coupon or not), which a named reviewer approves or rejects with a reason; stored in the `decision_log` table; nothing is sent | Verified reviewer identity (no authentication), customer appeal and correction process, explanations stored with each decision and reviewed wording for customers (a per-customer explanation is computed on request), causal validation of coupons |
| Recommendations | Ranked UCI descriptions, firing rules and fallback strategy | Display basket suggestions | Curated product identity, merchandising approval, online impact evaluation |
| Assistant | Cited policy excerpts or actual typed-tool results; refusal where checks fail. When streamed with OpenAI or Ollama, a model writes the policy answer from the cited excerpts, with a figure check | Display only | Robust adversarial defence, reliable model-written answers (the 3B local model wrote one false policy answer during tuning), real human handoff |
| Monitoring | Error rate, `/predict` latency, input and prediction drift (PSI and KS), data quality and rule fire rate, each with its rule | Automated alerts: log, `alert_log` table, optional webhook | Label backfill (designed, not built), log purge, multi-worker aggregation, production validation |
| Retraining | A candidate version in the MLflow registry with gate results; the weekly workflow opens a pull request | No automatic promotion: a named approver moves the champion alias; nothing has been promoted yet | A genuinely new holdout period, a shared registry server, executed cloud automation |

Drift alert conditions are PSI > 0.2 or **KS p-value < 0.05**. They are engineering investigation thresholds, not guarantees. Multiple comparisons and large-sample sensitivity are not corrected.

Scores and suggestions are generated automatically, but no real business action is wired downstream.

**The human approval step, and why it exists.** A decision based exclusively on automated processing triggers disclosure and human-review obligations under Québec's Law 25. So the retention score never decides on its own. In the Prediction tab, the model's output is shown as a proposal, and a named reviewer records "approved" or "rejected" with a reason (`POST /decisions`). Who decided what, when and why is kept in `decision_log` (`GET /decisions`).

That table is the only one holding a customer identifier, because a reviewed decision is about one person and must be traceable for that person's rights. Its retention period still has to be set. The step records a decision; it does not send anything. Without authentication, the reviewer's name is declared, not verified.

Review this inventory when contracts, providers, data retention or deployment scope change.

## 2026-11-05 implementation decision

The original 13 unnecessary features were removed using June development evidence before the new test evaluation. Contract 2.0/model 0.3.0 serves 12 inputs. The September snapshot was reused; the archive preserves prior results. Operational mitigation remains human review of every proposed action and no automatic dispatch. This does not fix the measured subgroup recall gap or verify reviewer identity. No champion promotion is authorized by this implementation decision.
