# Model promotion policy — retention model

Updated 2026-11-01. Applies to the MLflow registered model `boreal-retention`. Implemented by `src/registry.py`; the tolerances live in `configs/pipeline.yaml` under `registry.promotion`.

## Roles

- **Builder.** Trains a candidate, with `python -m src.train` or the weekly retraining workflow.
  - Every training run registers a new model version with the status `candidate`.
  - Registering never changes what is served.
- **Approver.** A named person who decides a promotion or a rollback.
  - In this project that is the project owner. In a real deployment it would be someone accountable for the campaign, not the builder alone.
  - The approver's name, the time, the reason and the gate results are stored on the model version and in `reports/current_run/registry/`.

## What "served" means

At startup, the API loads the version the `champion` alias points to. It first verifies that version's checksum and feature contract.

If any of the following happens, the API serves `artifacts/` instead and reports the reason in `/version` under `model_source`:

- the registry is unreachable;
- the registry has no champion;
- the champion's files fail their checks.

## Gates

`python -m src.registry evaluate --version N` checks the gates. They run on one shared holdout: the test snapshot of the processed data the candidate was built from. Each model is scored at its own decision threshold. All gates must pass.

| Gate | Rule | Why |
|---|---|---|
| Contract and integrity | The checksum matches the version's metadata. Feature names, their order and the contract version match the API. | Never serve a model the API cannot feed correctly. |
| Beats the baseline (first champion only) | PR-AUC is above the recency heuristic's, on the same rows. | A model must beat the business rule it would replace. |
| Same feature encoding | Candidate and champion map country codes to the same countries. | Otherwise the two are scored on different inputs. |
| PR-AUC not worse | Candidate PR-AUC ≥ champion's − 0.01. | The primary metric. The tolerance absorbs noise without inviting real regressions. |
| Calibration not worse | Candidate Brier score ≤ champion's + 0.01. | The decision threshold relies on calibrated probabilities. |
| No new fairness flags | No group may be strongly flagged for the candidate unless it is also flagged for the champion. | Applies the brief's five-point recall rule (section 3.7) to every change. |
| No widened fairness gaps | No existing strong flag may widen by more than 2 percentage points. | Known gaps may not get worse. |
| Known flags acknowledged | The approver must acknowledge every strong flag the candidate has, with `--acknowledge-fairness-flags`. The flags are recorded on the version. | Promoting a model with a known gap becomes a deliberate, recorded decision. |

The tolerances (0.01, 0.01 and 2 percentage points) are judgement values for this project. They are not derived from data.

## Procedure

1. **A candidate is registered.** This happens at the end of every training run.
2. **The weekly retraining workflow runs** (`.github/workflows/retrain.yml`). It re-runs the data pipeline, re-mines the rules and retrains a candidate. Only if every gate passes, it opens a pull request with the new metrics and the gate results (`scripts/retrain_report.py`). It never deploys and never promotes.
3. **A person reviews.** They read the metrics, the fairness tables and the gate report, then merge or close the pull request.
4. **The approver promotes the version.**

   ```bash
   .venv/bin/python -m src.registry promote --version N --approver "Name" --reason "why" [--acknowledge-fairness-flags]
   ```

   The command refuses, and changes nothing, if any gate fails or the approver or reason is missing.
5. **The API is restarted, or the same image redeployed,** so it serves the new champion. No rebuild is needed.

## Rollback

```bash
.venv/bin/python -m src.registry rollback --approver "Name" --reason "why"
```

- **What it does.** It moves the alias back to the previous champion, which is recorded on the current version.
- **No gates.** This is the emergency path.
- **To serve the old model again,** restart the API. No new image is built.
- **The record.** The rollback is recorded on both versions and in `reports/current_run/registry/`.

## Not allowed

- **Tuning against the gate holdout.** Do not re-tune a candidate against the holdout used for its gates and then run the gates again.
- **Moving the alias by hand in MLflow.** Use the command, so the approval is recorded. MLflow itself cannot enforce this: anyone with write access to the registry can move an alias. A real deployment must restrict who can write to the registry.
- **Overriding a failed gate.** Changing a tolerance is a policy change. Make it in `configs/pipeline.yaml` through a reviewed pull request that says why.

## Current state (2026-11-01)

- **Version 1** is the model in `artifacts/` (trained 2026-09-30), registered as a candidate.
- **Its gate check passes.** On the 2011-09-01 test snapshot its PR-AUC is 0.7430, against 0.6711 for the recency heuristic.
- **It carries five strong fairness flags** (`governance/fairness_audit.md`).
- **It is not promoted yet.** The first promotion needs the project owner as approver; until then the API reports the local fallback.
