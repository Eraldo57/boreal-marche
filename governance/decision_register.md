# Automated-decision inventory — implementation and plans

Updated 2026-09-30. An engineering inventory, not evidence of legal compliance.

| Component | Actual output | Actual downstream action | Missing controls |
|---|---|---|---|
| Retention | P(will_return in 90 days), thresholded return decision, inverse risk label | A proposed action (offer a coupon or not), which a named reviewer approves or rejects with a reason; stored in the `decision_log` table; nothing is sent | Verified reviewer identity (no authentication), customer appeal and correction process, individual explanations, causal validation of coupons |
| Recommendations | Ranked UCI descriptions, firing rules and fallback strategy | Display basket suggestions | Curated product identity, merchandising approval, online impact evaluation |
| Assistant | Cited policy excerpts or actual typed-tool result; refusal where checks fail | Display only | Robust adversarial defense, fluent bilingual generation, real human handoff |
| Monitoring | Local latency/error summary and measured PSI / KS p-value | Dashboard status only | Alert delivery, label backfill, log purge and production validation |
| Retraining | Local candidate artifact directory; dry-run command path checked | No automatic promotion | New holdout, registry alias workflow, human review and tested cloud automation |

Drift alert conditions are PSI > 0.2 or **KS p-value < 0.05**. They are engineering investigation thresholds, not guarantees. Multiple comparisons and large-sample sensitivity are not corrected.

Scores and suggestions are generated automatically, but no real business action is wired downstream.

**The human approval step, and why it exists.** A decision based exclusively on automated processing triggers disclosure and human-review obligations under Québec's Law 25. So the retention score never decides on its own. In the Prediction tab, the model's output is shown as a proposal, and a named reviewer records "approved" or "rejected" with a reason (`POST /decisions`). Who decided what, when and why is kept in `decision_log` (`GET /decisions`).

That table is the only one holding a customer identifier, because a reviewed decision is about one person and must be traceable for that person's rights. Its retention period still has to be set. The step records a decision; it does not send anything. Without authentication, the reviewer's name is declared, not verified.

Review this inventory when contracts, providers, data retention or deployment scope change.
