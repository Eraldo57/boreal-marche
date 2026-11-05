# Completion verification — 5 November 2026

Current model 0.3.0, contract 2.0: 12 serving features / 25 engineered, sigmoid calibration,
threshold 20.5%, test AP 0.7596308, ROC-AUC 0.7863254, Brier 0.1939829.
The original model is archived; test reuse is disclosed, not an independent-holdout claim.

| Check | Result / evidence |
|---|---|
| Unit suite | 421 passed; completion_final_unit_20261105.log |
| Artifact integration | 7 passed; completion_final_integration_20261105.log |
| Combined coverage run | 428 passed; 90%, 3,014 statements / 308 missed; completion_coverage_summary.txt |
| Lint + format | clean; 92 files formatted; completion_lint_20261105.log |
| Strict type check | clean, 33 source files; completion_mypy_20261105.log |
| Real HTTP / widget checks | 16 HTTP and 7 interaction checks; completion_api_smoke / completion_ui_smoke logs |
| Evaluation UI | Four tabs, 3 images, current threshold and sigmoid selection; completion_evaluation_ui log |
| Notebooks | All code cells in all 3 notebooks executed successfully; completion_notebooks log |
| Containers | Native arm64 and linux/amd64 API/frontend images built; sizes recorded in completion_container_sizes.txt |
| amd64 inference | Current joblib loaded, exact 12-feature contract checked, real probability returned |
| DVC | Raw pointer and explicit dependency; cache/lockfile snapshot of existing manual outputs; up to date |
| Submission QA | 20 rendered slides, native editable text/diagrams, 20 speaker-note parts; zero layout errors/warnings; PPTX cross-rendered; report8 / executive2 / reflection-draft2 pages inspected |

No current remote CI, public deployment or AWS execution is claimed. Invalid GitHub credentials,
absent AWS credentials and an unspecified remaining spend limit prevent those steps. No release
tag, champion promotion, coupon dispatch or historical deliverable evidence was fabricated.

The first retraining attempt failed during audit generation, then the audit was corrected and the
locked protocol rerun successfully. Both training logs remain. Two additional diagnostic UI checks
used a wrong AppTest element name before the current `image` element was verified. These are
local verification corrections, not independent model evaluations.
