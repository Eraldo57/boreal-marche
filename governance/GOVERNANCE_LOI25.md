# Governance — Law 25 design and implementation status

Updated 2026-09-30. Academic design record, **not legal advice or a compliance certification**.

## Legal reference and scope

The Québec private-sector privacy framework addresses transparency and review for decisions based exclusively on automated processing. See the [CAI enterprise guidance](https://www.cai.gouv.qc.ca/protection-renseignements-personnels/information-entreprises-privees/utilisation-communication-renseignements-personnels) and [current private-sector law](https://www.legisquebec.gouv.qc.ca/fr/document/lc/p-39.1), particularly section 12.1. Applicability and operating procedures require qualified review before any real deployment.

Boréal Marché is fictional. Public source identifiers and transaction histories still merit care; lack of names does not establish anonymity. No real customer should upload personal information to this unauthenticated demo.

## Implemented controls

- Strict prediction/tool input validation; exactly 25 features; no names/emails/payment details are model inputs.
- Shared assistant scope/privacy checks precede tool routing; policy answers cite retrieved text.
- Retention tools require matching caller-supplied demo customer context.
- No coupon dispatch or customer-data mutation exists.
- Durable monitoring JSONL omits customer IDs and chat text, but includes prediction feature vectors.
- Interface discloses public-data provenance, the historical demo, and optional external provider use.

These controls are limited: a supplied ID is **not authentication**, regex guardrails are not a security boundary, and feature vectors can be identifying. Direct API feature lookup is public to anyone with local API access. Bind to localhost.

## Planned, not enforced

| Control | Current gap / next action |
|---|---|
| Authentication and authorization | Add authenticated identity, per-customer permissions, rate limits and access audit |
| Human review | No approval queue, override, dispatch or support-handoff service; design before consequential use |
| Explanation and correction | Global feature hints are not individual explanations; implement reviewed explanations and correction process |
| Minimization | All 25 features are inventoried; necessity is not established by predictive use alone |
| Retention/deletion | No scheduled purge, erasure request workflow, backup/cache/model lineage deletion or proof of deletion |
| External processing | OpenAI would transmit message/retrieved context; no privacy impact assessment or provider agreement completed |
| Fairness | Limited segment diagnostics only; no protected-characteristic conclusions |
| Incident management | No tested incident response, notification, authorization or escalation workflow |

## Retention/deletion policy proposal

For the academic demo, retain artifacts only as needed for assessment; the student must review stored files before sharing. No automatic retention deadline is implemented. For a future service, define justified periods separately for source data, derived features, prediction telemetry, chats, model versions and backups, obtain review, then implement and test purge/erasure across DVC caches and MLflow as well as primary storage.

Do not interpret illustrative periods in the fictional policy corpus as enforced application behavior. There is no operational customer-support contact or formal privacy-officer appointment in this repository.

## Feature minimization

See FEATURE_INVENTORY.md for the canonical 25 names, order and definitions, including n_countries. Geographic features can act as proxies; integer encoding does not anonymize location. Monetary and average-basket-value duplication should be tested via ablation. No statement that every feature is legally necessary is made.
