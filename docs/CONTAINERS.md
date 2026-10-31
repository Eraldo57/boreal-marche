# Containers: design, sizes and how to run them

The brief (§3.6, Containers) asks for a multi-stage Dockerfile with a digest-pinned base image,
a non-root runtime user, a HEALTHCHECK, a `.dockerignore`, a reported image size before and
after optimisation, and a Compose file bringing up API, interface, database and local LLM with
a single command. §7.2 places Dockerfiles and the Compose stack in `docker/`. This page states
where each requirement is met, why each choice was made, and what was and was not verified.

| Requirement | Where |
|---|---|
| Multi-stage build | `docker/Dockerfile`: `api-deps` → `api`, plus a separate `frontend` stage |
| Digest-pinned base | `ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:e697c818…` |
| Non-root runtime user | `USER appuser` in both runtime stages |
| HEALTHCHECK | API probes `/ready`; interface probes Streamlit's `/_stcore/health` |
| `.dockerignore` | repository root (the build context) |
| Size before and after | [below](#image-size-before-and-after); raw numbers in `reports/current_run/container_sizes.json` |
| One-command stack | `docker/compose.yml`: `db`, `api`, `frontend`, `ollama`, `ollama-pull` |

`tests/test_containers.py` checks these properties on every CI run; the CI `docker-build` job
builds both images and reports their size.

## Run the whole stack

```bash
make setup && make all     # once: model, rules, features, index and embedding model on the host
make up                    # = docker compose -f docker/compose.yml up --build -d
```

Then open http://127.0.0.1:8501 (interface) and http://127.0.0.1:8000/docs (API).
`make down` stops everything and keeps the database volume;
`docker compose -f docker/compose.yml down -v` also deletes the database and the Ollama models.

`make up` always rebuilds (from cache, seconds) and then restarts the API and interface
containers, about 10–15 s, even when nothing changed: Compose's builder wraps the unchanged image
in a new index each time, and the new index digest counts as a new image. To start existing
containers without rebuilding, use `docker compose -f docker/compose.yml up -d`.

Read the stored prediction log:

```bash
docker compose -f docker/compose.yml exec db psql -U boreal -d boreal \
  -c "select path, status_code, count(*) from prediction_log group by 1, 2;"
```

Before the first run:

- **Artifacts come from the host.** The API mounts `artifacts/`, `data/processed/`,
  `data/chromadb/` and `data/model_cache/`. DVC has no remote, so a fresh clone must run the
  pipeline (`make all`), which downloads the UCI workbook and verifies its SHA-256. Without
  them the API stays up but `/ready` returns 503 and names what is missing.
- **macOS:** if the repository is under Desktop, Documents or Downloads, Docker Desktop needs
  permission to read it (System Settings → Privacy & Security → Files and Folders, or Full Disk
  Access, for Docker), or the clone must live elsewhere. Without it Compose fails with
  `mkdir /host_mnt/Users/<you>/Desktop: operation not permitted`.
- **Linux:** set `LOCAL_UID` and `LOCAL_GID` (output of `id -u` / `id -g`) so the API can
  write its index and model-cache mounts, and create `data/chromadb` and `data/model_cache`
  first so Docker does not create them as root.
- **Compose 2.24 or newer** (the optional `env_file` entry needs it).
- **Downloads.** The Ollama image is 2.8 GB compressed on linux/arm64 and 3.4 GB on
  linux/amd64. The configured model, `llama3.2:3b`, is a further 2.02 GB from the Ollama
  registry (the 8B `llama3.1` would be 4.92 GB; `llama3.3` and `llama4` are 42.5 and 67.4 GB).
  Set `OLLAMA_MODEL` in `.env` to use another model; the API and the one-shot `ollama-pull`
  service both read it.
- **Ports.** Published ports bind to 127.0.0.1 only, because these demo endpoints have no
  authentication. If 8000, 8501 or 5000 are taken, set `API_PORT`, `STREAMLIT_PORT` or
  `MLFLOW_PORT` in `.env`. PostgreSQL and Ollama are not published, so a PostgreSQL or Ollama
  already running on the host does not conflict.

## Image size before and after

Measured on 2026-10-31 with Docker 29.4.1 on Apple Silicon, so these are **linux/arm64**
images. CI and AWS build linux/amd64, where sizes differ; the CI job prints the amd64 size.

| Image | Uncompressed | Compressed | GPU-only packages | PyTorch |
|---|---|---|---|---|
| **Before**: one image, original root Dockerfile | 7,376 MB | 3,691 MB | 4,394 MB | 1,117 MB |
| **After**: `api` target | 1,993 MB | 534 MB | 0 MB | 517 MB |
| **After**: `frontend` target | 635 MB | 180 MB | 0 MB | — |

*Uncompressed* is the sum of layer sizes (`docker history`), what the container's filesystem
holds. *Compressed* is the image content size (`docker image inspect`), what a registry such as
ECR stores and each host downloads. Docker Desktop's "disk usage" (11.1 GB for the before image)
counts both copies and is not an image size. The API image shrank by 73.0% uncompressed and
85.5% compressed. The brief's size bands (above 2 GB / below 1 GB / below 700 MB) do not say
which measure they mean: the API image is below 2 GB but above 1 GB uncompressed, and below
700 MB compressed.

Installed Python packages went from 7,005 MB to 1,702 MB, measured the same way in both
images. The main contributions (each measured on its own, so they do not add up exactly):

| Change | Saved |
|---|---|
| CPU-only PyTorch: no `nvidia-*` CUDA libraries or `triton` | 4,394 MB |
| PyTorch itself: CUDA build 1,117 MB → CPU build 517 MB, after removing its compiled C++ tests (`torch/test`, 116 MB) and C++ headers (`torch/include`, 64 MB) | 600 MB |
| Training and interface packages out of the API image (`mlflow`, `optuna`, `mlxtend`, `openpyxl`, `streamlit` and what only they pull in) | 416 MB |

What the API image still holds, largest first (`du`, approximate): PyTorch 517 MB, pyarrow
153 MB, scipy 149 MB with its bundled libraries, kubernetes 119 MB, transformers 115 MB,
pandas 77 MB, sympy 74 MB, onnxruntime 59 MB, scikit-learn 57 MB. kubernetes and onnxruntime are declared
dependencies of chromadb that this service does not call, but deleting a declared dependency
would break in whichever code path was not tested, so they stay. A much smaller image would mean
replacing PyTorch with an ONNX embedding runtime, which changes Head C's runtime and requires
rebuilding the index and re-running the assistant evaluation.

## Design choices and why

| Choice | Why |
|---|---|
| Multi-stage: `api-deps` builds a virtualenv, `api` copies only `/opt/venv` | pip's work files never reach the runtime image; the runtime stage holds the venv, the code and nothing else |
| Base `python:3.12-slim-bookworm`, pinned by digest | A tag moves when Docker Hub republishes it; a digest names the same bytes every time. The digest is the multi-arch index, so it resolves to amd64 (AWS) and arm64 (Apple Silicon). Debian slim uses glibc, so the manylinux wheels install as-is; Alpine's musl would force source builds |
| CPU-only PyTorch, installed first from PyTorch's CPU index | The default Linux wheel brings 4.4 GB of CUDA libraries a GPU-less service never loads. Installed first, it already satisfies sentence-transformers, so pip does not fetch the CUDA build |
| Only `[project].dependencies` installed | `pyproject.toml` separates what the API imports from the `train` and `frontend` extras. `pip install -e ".[dev]"` still installs everything for development and CI |
| `pyproject.toml` copied before any code | The dependency layer is rebuilt only when dependencies change; a code edit reuses it |
| No compiler in the build | Every dependency has a prebuilt wheel for linux/amd64 and linux/arm64 (checked with `pip install --dry-run --only-binary=:all:` for both), so `gcc`/`g++` were removed |
| Separate `frontend` image | The interface only calls the API over HTTP. Its image is 635 MB instead of carrying PyTorch and the model libraries |
| `USER appuser`; code root-owned; only `data/` writable | A compromised process cannot rewrite the application code |
| HEALTHCHECK on `/ready`, not `/health` | Compose starts the interface only once the API reports model, rules, index and prediction log loaded. The start period (120 s in the image, 300 s in Compose) covers a first-run download of the embedding model; a passing check ends it early |
| `.dockerignore` | Keeps `.env` secrets, raw data, the virtualenv, tests, git history and the per-customer `test_predictions.parquet` out of the build context |
| `APP_ENV=production`, `PYTHONUNBUFFERED=1`, `PYTHONDONTWRITEBYTECODE=1` | JSON log lines, written immediately; no bytecode written into the read-only code |
| `ANONYMIZED_TELEMETRY=False` | Turns off chromadb's anonymous usage reporting inside the container |
| `GIT_SHA` build argument and OCI labels | `/version` reports the commit the image was built from (CD passes `github.sha`, `make up` passes `git rev-parse HEAD`) |
| PostgreSQL service | The brief requires every prediction logged into a database table. The API writes one row per request to `prediction_log` (SQLite outside Docker, PostgreSQL in Compose) and replays it on restart |
| Ollama plus one-shot `ollama-pull` | Brings up the local LLM and its model with the same single command; the pull is instant once the model is in the volume |
| Third-party images pinned by digest; ports on 127.0.0.1; database and Ollama unpublished | Same reproducibility argument as the base image; no unauthenticated endpoint reachable from the network |

## Verified on 2026-10-31

On the machine above. Until Docker Desktop was given access to the Desktop folder (see the macOS
note), the runs used clones of the mounted folders stored outside `~/Desktop`. From the `make up`
bullet onwards, and for the third latency run, the stack used the real mounts:

- Both images build: the API in 146 s on its first build, the interface in 36 s.
- The API image run alone, with no mounts, returns 200 on `/health` and 503 on `/ready`, naming
  the missing Chroma index and customer feature store. Model and rules load. The process runs
  as `appuser` (uid 999).
- The API was healthy within 18 s of `docker compose up` (database first, embedding model already
  cached), and the interface right after. A
  smoke test passed 10 of 10 checks: `/health`, `/ready`, `/version`, historical customer 12347's
  features, `/predict` (probability 0.712 at threshold 0.22), `/predict/batch`, a rule-based
  `/recommend`, and `/chat` in English and French with the regex provider.
- 100 sequential `/predict` calls measured by the client, including the PostgreSQL write, in three
  runs: p50 41.9, 41.4 and 41.4 ms; p95 45.7, 57.8 and 54.4 ms. The p95 is therefore not reliably
  under the brief's 50 ms target on this machine. All calls used the same customer, so this is a
  latency check, not a load test; the same repetition is why the drift monitor raised a PSI alert.
- PostgreSQL held all 105 requests in `prediction_log`; after an API restart, `/metrics` still
  reported 105.
- The interface reached the API at `http://api:8000` and served its page; the API wrote JSON
  log lines.
- `/version` reported the commit passed as `GIT_SHA`. Rebuilding with only a different commit
  took 1 s with every step cached, because the commit is stamped in the last layers.
- One `docker compose up` started all five services; `ollama-pull` exited 0; `/chat` with
  `provider: "ollama"` returned 200 through the Ollama container. This first run used the small
  `qwen2.5:0.5b` model to test the wiring only.
- After Docker Desktop was allowed to read the Desktop folder, the documented `make up` ran from
  the repository itself with its real mounts: all five services healthy, the same smoke test
  10/10, and the interface (port 8501) and `/docs` (port 8000) both returned 200 on the default
  ports. `ollama-pull` fetched the configured `llama3.2:3b` in 60 s.
- Three `/chat` questions through `llama3.2:3b` returned 200 and chose `policy_search`,
  `policy_search` (French) and `recommendations`, in 17.8, 11.6 and 7.4 s (the first includes
  loading the model). Three questions are a wiring check, not an evaluation.
- Ollama in Docker runs on the CPU only (its log reports `inference compute id=cpu`): Docker on a
  Mac gives containers no GPU, whereas the native Ollama app reports the Apple M4 GPU (Metal).
  Memory in use with the whole stack up: Ollama 3.0 GB, API 1.2 GB, PostgreSQL and interface under
  0.1 GB each, of the 7.75 GB Docker VM.

Not verified: a linux/amd64 build (the CI job does it); a fresh clone on a different machine (the
brief's pre-deployment checklist); an evaluation of the Ollama provider; any cloud deployment.

## Known gaps

- **The cloud image is not self-sufficient.** The Chroma index, the customer feature store and
  the 470 MB embedding model are mounted, not baked in, so the API image on its own is live but
  not ready. Deploying it needs those three either baked into the image (larger) or fetched at
  start-up; that decision belongs with the AWS deployment.
- Chroma opens its index read-write, so that mount cannot be read-only.
- The prediction log is single-process: run one uvicorn worker per container.
