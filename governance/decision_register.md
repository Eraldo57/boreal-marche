# Automated-decision inventory — implementation and plans

Updated 2026-09-30. An engineering inventory, not evidence of legal compliance.

| Component | Actual output | Actual downstream action | Missing controls |
|---|---|---|---|
| Retention | P(will_return in 90 days), thresholded return decision, inverse risk label | Display only; no coupons or approval workflow | Authentication, individual explanations, review/appeal and causal intervention validation |
| Recommendations | Ranked UCI descriptions, firing rules and fallback strategy | Display basket suggestions | Curated product identity, merchandising approval, online impact evaluation |
| Assistant | Cited policy excerpts or actual typed-tool result; refusal where checks fail | Display only | Robust adversarial defense, fluent bilingual generation, real human handoff |
| Monitoring | Local latency/error summary and measured PSI / KS p-value | Dashboard status only | Alert delivery, label backfill, log purge and production validation |
| Retraining | Local candidate artifact directory; dry-run command path checked | No automatic promotion | New holdout, registry alias workflow, human review and tested cloud automation |

Drift alert conditions are PSI > 0.2 or **KS p-value < 0.05**. They are engineering investigation thresholds, not guarantees. Multiple comparisons and large-sample sensitivity are not corrected.

Scores and suggestions are generated automatically, but no real business action is wired downstream. A proposed human review step must not be described as implemented. Review this inventory when contracts, providers, data retention or deployment scope change.
