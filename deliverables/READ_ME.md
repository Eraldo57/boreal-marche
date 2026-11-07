# Boréal Marché — defence package

> Deployment update, 2026-11-06: Google Cloud hosting, public GitHub CI and merge-triggered deployment were completed after this local report. See [the current Google Cloud runbook](https://github.com/Eraldo57/boreal-marche/blob/main/docs/GOOGLE_CLOUD.md) and reports/current_run/google_cloud/ for subsequent execution evidence. Earlier pending-cloud statements below describe the dated local snapshot.


Open Boreal_Marche_Defence.pptx: 20 slides grouped by Head, all with speaker notes.
Use Presenter View to read the notes; Defence_Cue_Card.md is the short rehearsal aid.
Boreal_Marche_Defence.pdf is the printable deck. Text, bars and diagrams are editable in PowerPoint;
the scientific calibration plot is embedded from the executed report.

The technical report is 8 pages; the bilingual executive summary is exactly 2 pages.
Editable Word files accompany their PDFs. The personal reflection is a 2-page DRAFT that Eraldo
must review, correct and approve before submitting. No personal actions or cloud evidence should
be inferred from an unreviewed draft.

Current figures: model0.3.0, contract2.0, 12 model inputs from25 engineered, sigmoid calibration,
AP0.7596, ROC0.7863, Brier0.1940, threshold20.5%. Earlier25-feature artifacts are archived.
The September test period was reused. Coupon cost5/margin25 are scenarios, not measured uplift.

428 tests pass with90% source/API coverage. Current local Docker UI: http://127.0.0.1:18501;
API: http://127.0.0.1:18000. The original native UI is http://127.0.0.1:8501.

External requirements remain: GitHub authentication/green CI/public release, AWS credentials and
spend cap for actual jobs/endpoints/costs, hosting/OIDC, independent fresh-machine reproduction,
video and owner-verified personal reflection. See Completion_Status.md. This package does not
claim a fully deployed or fully submitted capstone. AI assistance is disclosed.

## Revised slide organisation

Overview1–3 → Head A4–10 → Head B11–14 → Head C15–17 → engineering18 → conclusion19 → sources20. The revised deck replaces the earlier main/appendix arrangement mentioned in the technical report. All underlying scientific values remain unchanged. GitHub and AWS are left code-ready awaiting sign-in.
