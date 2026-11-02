# Governance — Law 25 design and implementation status

Updated 2026-11-01. Academic design record, **not legal advice or a compliance certification**.

## Legal reference and scope

The Québec private-sector privacy framework addresses transparency and review for decisions based exclusively on automated processing. See the [CAI enterprise guidance](https://www.cai.gouv.qc.ca/protection-renseignements-personnels/information-entreprises-privees/utilisation-communication-renseignements-personnels) and [current private-sector law](https://www.legisquebec.gouv.qc.ca/fr/document/lc/p-39.1), particularly section 12.1. Applicability and operating procedures require qualified review before any real deployment.

Boréal Marché is fictional. Public source identifiers and transaction histories still merit care; lack of names does not establish anonymity. No real customer should upload personal information to this unauthenticated demo.

## Implemented controls

- Strict prediction/tool input validation; exactly 25 features; no names/emails/payment details are model inputs.
- Shared assistant scope/privacy checks precede tool routing; policy answers cite retrieved text.
- Retention tools require matching caller-supplied demo customer context.
- No coupon dispatch or customer-data mutation exists.
- **Human approval step.** Section 12.1 attaches disclosure and human-review obligations to decisions based exclusively on automated processing, so the retention model only proposes an action. A named reviewer approves or rejects it with a reason in the Prediction tab, and the decision is stored in the `decision_log` table (`POST` and `GET /decisions`). This is the design consequence the brief asks for. See `governance/decision_register.md`.
- The durable monitoring log (`prediction_log` database table) omits customer IDs and chat text, but includes prediction feature vectors. In Compose the database port is not published outside the stack's internal network.
- Interface discloses public-data provenance, the historical demo, and optional external provider use.
- **Customer notice, in plain language, in French and English** (`governance/customer_notice_en.md`, `customer_notice_fr.md`). It is rendered in the interface's About tab and follows the language toggle. It says what the model estimates, which information it uses and does not use, that a person decides, and the customer's rights, with the fictional privacy officer's contact.
- Monitoring alerts carry no customer identifiers. The `alert_log` table stores check names, values and rules only.

These controls are limited: a supplied ID is **not authentication**, regex guardrails are not a security boundary, and feature vectors can be identifying. Direct API feature lookup is public to anyone with local API access. Bind to localhost.

## Planned, not enforced

| Control | Current gap / next action |
|---|---|
| Authentication and authorization | Add authenticated identity, per-customer permissions, rate limits and access audit |
| Human review | The approval step exists, but the reviewer's name is declared, not authenticated. There is no customer-facing appeal or correction process, no support handoff, and no retention period yet for `decision_log`, which holds customer IDs |
| Explanation and correction | Global feature hints are not individual explanations; implement reviewed explanations and correction process |
| Minimization | Audit done (`governance/FEATURE_INVENTORY.md`). A drop-column test on development data finds 13 of 25 features with no reliable contribution, including all three location features, and a 12-feature model scores higher. Removing them is recommended but not applied: it needs the owner's decision and a retrain. Predictive necessity is not legal necessity |
| Retention/deletion | No scheduled purge, erasure request workflow, backup/cache/model lineage deletion or proof of deletion |
| External processing | OpenAI would transmit message/retrieved context; no privacy impact assessment or provider agreement completed |
| Fairness | Subgroup audit done (`governance/fairness_audit.md`). There are five strong recall flags, all customers with a thin purchase history; five mitigations are listed, none chosen. Promotion gates stop a new model from adding or widening a flag. The groups are behavioural proxies only, so there are no protected-characteristic conclusions |
| Incident management | No tested incident response, notification, authorization or escalation workflow |

## Retention/deletion policy proposal

For the academic demo, retain artifacts only as needed for assessment; the student must review stored files before sharing. No automatic retention deadline is implemented. For a future service, define justified periods separately for source data, derived features, prediction telemetry, chats, model versions and backups, obtain review, then implement and test purge/erasure across DVC caches and MLflow as well as primary storage.

Do not interpret illustrative periods in the fictional policy corpus as enforced application behavior. There is no operational customer-support contact or formal privacy-officer appointment in this repository.

## Feature minimization

See governance/FEATURE_INVENTORY.md for the canonical 25 names, order and definitions, including n_countries, and for the data-minimisation audit.

- **Location proxies.** Geographic features can stand in for location; integer encoding does not anonymize it.
- **What the ablation found.** The location features and the monetary duplicates show no reliable predictive contribution.
- **Not a legal claim.** No statement that every feature is legally necessary is made.
