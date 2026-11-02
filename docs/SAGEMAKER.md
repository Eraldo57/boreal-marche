# SageMaker training, tuning and endpoint

Updated 2026-11-02. Covers brief section 3.5. **Nothing here has run on AWS yet.** The code and image are built and were verified locally, as recorded below; the AWS steps need an account.

## What exists

| Piece | File | What it does |
|---|---|---|
| Training script | `src/train.py` | One file, run locally or as a SageMaker job (script mode). It takes hyperparameters as arguments and reads its data and model folders from `SM_CHANNEL_TRAIN` and `SM_MODEL_DIR`, defaulting to `data/processed` and `artifacts/`. `--mode hpo-trial` runs one tuning trial and prints `validation_pr_auc=…`. |
| Container contract | `scripts/sagemaker_entry.py` | SageMaker starts the image as `train` or `serve`. For `train` it reads `hyperparameters.json`, sets `SM_CHANNEL_TRAIN` and `SM_MODEL_DIR`, and runs `src/train.py`; a failure is written to `/opt/ml/output/failure`. For `serve` it starts the endpoint app. |
| Endpoint app | `api/sagemaker_serve.py` | `GET /ping` and `POST /invocations` on port 8080. It applies the same checksum, contract and input validation as the API. |
| Image | `docker/Dockerfile`, target `sagemaker` | Python 3.12 with training and serving libraries only (`docker/sagemaker-requirements.txt`): no PyTorch, embeddings or Chroma. |
| Launcher | `scripts/sagemaker_jobs.py` | Commands `upload-data`, `train`, `tune`, `deploy`, `invoke`, `teardown` and `compare`, all in `ca-central-1`. Each saves what AWS returned in `reports/current_run/sagemaker/`. |

**Why a custom image.** AWS's prebuilt scikit-learn images run an older Python. This project needs Python 3.12 and scikit-learn 1.6 or later: the calibration uses `FrozenEstimator`, and the code uses `datetime.UTC`.

## Verified locally (linux/arm64, Docker Desktop)

These ran the image exactly as SageMaker would, with `/opt/ml` folders mounted from disk, but **not on AWS**. The re-run of 2026-11-02, with the image built on 2026-11-01, is recorded with its commands, timings and outputs in `reports/current_run/sagemaker_local.json`:

- **Tuning trial.** With hyperparameters `n-estimators=121`, `max-depth=6`, `min-samples-leaf=2` on the real processed data, it printed `validation_pr_auc=0.673018` (December 2010 validation snapshot).
- **Full training job.** With 12 Optuna trials per family it took 18.2 seconds, including container start, and selected a random forest. It reproduced the reported test metrics exactly (PR-AUC 0.7430379). The model file is not byte-identical to `artifacts/`, so the match rests on the metrics; the image has numpy 2.5.4, where `artifacts/` was built with 2.5.3.
- **Endpoint.** Serving that job's model folder, `/ping` returned 200. `/invocations` for historical customer 12347 returned probability 0.711864, "will return" at threshold 0.22, the same as the API.

## Running it on AWS

1. **Prerequisites (one-time, in `ca-central-1`).**
   - An S3 bucket.
   - An ECR repository, for example `boreal-marche-sagemaker`.
   - A SageMaker execution role that can read the bucket, pull from ECR and write CloudWatch logs.
2. **Build and push for Intel/AMD.** SageMaker's standard instances (ml.m5, ml.t2) are x86_64, but the local build above was ARM. Build with `make sagemaker-image` (`linux/amd64`), then tag and push to ECR.
3. **Set the three variables** in `.env.example`: `SAGEMAKER_ROLE_ARN`, `SAGEMAKER_IMAGE_URI` and `SAGEMAKER_BUCKET`. Then install boto3 with `pip install -e '.[sagemaker]'`.
4. **Run the commands:**

   ```bash
   python -m scripts.sagemaker_jobs upload-data
   python -m scripts.sagemaker_jobs train                     # full pipeline, managed spot
   python -m scripts.sagemaker_jobs tune --max-jobs 12 --parallel 3
   python -m scripts.sagemaker_jobs deploy --training-job <bm-train-…>
   python -m scripts.sagemaker_jobs invoke --endpoint <bm-endpoint-…>
   python -m scripts.sagemaker_jobs teardown --endpoint <bm-endpoint-…>   # the same session
   python -m scripts.sagemaker_jobs compare --tuning-evidence reports/current_run/sagemaker/<bm-hpo-…>.json \
       --price-per-hour <USD from the AWS pricing page> --price-source "<URL, date>"
   ```

The tuning job is Bayesian and runs on managed spot instances. It uses the same random-forest ranges as the local Optuna study, so the two can be compared directly.

## Cost reasoning (to complete with real prices)

- **No prices are assumed in this repository.** `compare` takes the hourly price you read on the AWS pricing page that day, and records its source.
- **Spot capacity.** Training and tuning use managed spot capacity. The saving is measured from SageMaker's own `BillableTimeInSeconds` against the training time, not assumed.
- **The endpoint.** It is billed for every hour it is in service. That is why the brief asks for it to be deployed once, invoked, recorded and torn down in one session. `teardown` records that no endpoint remains.
- **Serving.** It stays on App Runner or Render (section 3.6), which is cheaper for a portfolio link than a permanent endpoint.

## Crossover: when does managed tuning pay off?

Locally, a whole training run, including 24 Optuna trials, took about 13 seconds (`reports/current_run/logs/commands.jsonl`). So one trial takes well under a second, while each managed job pays minutes of start-up. On this dataset (about 10,000 training rows), managed tuning is slower and costs more.

`compare` turns this into numbers. It times the best local trial on this machine, then computes the per-trial duration at which parallel managed trials would finish first (`local_trial_seconds_at_crossover`). Managed tuning becomes worthwhile when a single trial takes minutes: a much larger dataset, heavier model families or many more trials.

## Not done

- No AWS command has run. There are no job ARNs, invocation or teardown records, real costs, or spot savings yet.
- The image has not been built for `linux/amd64` or pushed to ECR.
