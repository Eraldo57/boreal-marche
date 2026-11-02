# syntax=docker/dockerfile:1
# ===========================================================================
# Boréal Marché — container images. The build context is the repository root:
#   docker build -f docker/Dockerfile --target api      -t boreal-marche/api .
#   docker build -f docker/Dockerfile --target frontend -t boreal-marche/frontend .
# docker/compose.yml builds both. Measured sizes and the reason for each choice
# are in docs/CONTAINERS.md.
# ===========================================================================

# Base image pinned by digest. A tag such as 3.12-slim-bookworm is re-pointed whenever
# Docker Hub republishes it; a digest always names the same bytes. This digest is the
# multi-arch index, so it resolves to linux/amd64 (AWS) or linux/arm64 (Apple Silicon).
# To move to a newer base: docker buildx imagetools inspect python:3.12-slim-bookworm
ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337

# ---------------------------------------------------------------------------
# Stage 1: api-deps — install the API's Python dependencies into a virtualenv
# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS api-deps
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /build

# CPU-only PyTorch first. The default Linux wheel brings ~4.4 GB of NVIDIA CUDA libraries
# (nvidia-*, triton) that a service without a GPU never loads. Once installed, it already
# satisfies sentence-transformers, so the next step does not fetch the CUDA build.
ARG TORCH_VERSION=2.15.0+cpu
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}"

# Only pyproject.toml is copied, so this layer is rebuilt when dependencies change, not on
# every code edit. Only [project].dependencies are installed: no training or UI packages.
# Every dependency ships a prebuilt wheel for both architectures, so no compiler is needed.
COPY pyproject.toml ./
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']))" > requirements.txt \
    && pip install -r requirements.txt

# PyTorch's compiled C++ unit tests (torch/test) and C++ headers (torch/include, used only to
# build extensions or by torch.compile) are never loaded when the model runs: ~180 MB.
RUN rm -rf /opt/venv/lib/python3*/site-packages/torch/test \
           /opt/venv/lib/python3*/site-packages/torch/include

# ---------------------------------------------------------------------------
# Stage 2: api — the FastAPI service (the image CD deploys)
# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS api
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production \
    ANONYMIZED_TELEMETRY=False \
    HF_HOME=/app/data/model_cache

LABEL maintainer="Eraldo Ibrushi <eraldo.ibrushi@gmail.com>" \
      org.opencontainers.image.title="boreal-marche-api" \
      org.opencontainers.image.description="Boréal Marché retention, recommendation and assistant API"

# Non-root runtime user. Code stays root-owned (read-only to it); only data/ is writable.
RUN groupadd --system appuser && useradd --system --gid appuser --no-create-home appuser
WORKDIR /app
COPY --from=api-deps /opt/venv /opt/venv

# Least-changing first, so a code edit reuses the layers above it.
COPY configs/ ./configs/
COPY docs/corpus/ ./docs/corpus/
COPY artifacts/ ./artifacts/
COPY src/ ./src/
COPY api/ ./api/
RUN mkdir -p data/logs data/chromadb data/model_cache data/processed data/registry_cache \
    && chown -R appuser:appuser data

# The commit changes on every build, so it is stamped last: only these metadata layers
# are rebuilt per commit, not the dependency and code layers above.
ARG GIT_SHA=unknown
ENV GIT_SHA=$GIT_SHA
LABEL org.opencontainers.image.revision=$GIT_SHA
USER appuser

EXPOSE 8000
# Healthy means ready: model, rules, index and prediction log all loaded (/ready, not /health).
# The long start period covers the first-run download of the embedding model.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/ready').raise_for_status()" || exit 1
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------------------------------------------------------------------------
# Stage 3: frontend — the Streamlit interface. It only calls the API over HTTP,
# so it carries none of the model libraries.
# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS frontend
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

LABEL maintainer="Eraldo Ibrushi <eraldo.ibrushi@gmail.com>" \
      org.opencontainers.image.title="boreal-marche-frontend" \
      org.opencontainers.image.description="Boréal Marché Streamlit interface"

RUN groupadd --system appuser && useradd --system --gid appuser --create-home appuser
WORKDIR /app
COPY frontend/requirements.txt ./frontend/requirements.txt
RUN pip install -r frontend/requirements.txt
COPY frontend/ ./frontend/
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8501/_stcore/health').raise_for_status()" || exit 1
CMD ["streamlit", "run", "frontend/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
