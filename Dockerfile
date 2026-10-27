# ===========================================================================
# Boréal Marché — Multi-stage Dockerfile
# ===========================================================================
# Image size has not been measured. Training/RAG dependencies remain substantial.
# ===========================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder — install Python dependencies
# ---------------------------------------------------------------------------
# Pinned to a specific tag for reproducibility.
# To pin to a digest, run: docker pull python:3.11-slim-bookworm
# and replace the tag with the sha256 from: docker inspect --format='{{index .RepoDigests 0}}'
FROM python:3.12-slim-bookworm AS builder

WORKDIR /build

# Install build-time system deps (gcc for some wheels)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy only requirements first for layer caching
COPY pyproject.toml ./
COPY src/ ./src/
COPY api/ ./api/
COPY app/ ./app/
COPY scripts/ ./scripts/
RUN pip install --no-cache-dir --prefix=/install .

# ---------------------------------------------------------------------------
# Stage 2: Runtime — lean image with only what we need
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime
ARG GIT_SHA=unknown
ENV GIT_SHA=$GIT_SHA
ENV HF_HOME=/app/data/model_cache

LABEL maintainer="Eraldo Ibrushi <eraldo.ibrushi@gmail.com>"
LABEL description="Boréal Marché Customer Intelligence Platform"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Create non-root user
RUN groupadd --system appuser && \
    useradd --system --gid appuser --no-create-home appuser

# Copy application code
COPY src/ ./src/
COPY api/ ./api/
COPY app/ ./app/
COPY configs/ ./configs/
COPY artifacts/ ./artifacts/
COPY docs/ ./docs/
COPY governance/ ./governance/
COPY scripts/ ./scripts/

# Own the working directory
RUN mkdir -p /app/data/logs /app/data/chromadb /app/data/model_cache && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose API and Streamlit ports
EXPOSE 8000 8501

# Health check against the API
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/ready').raise_for_status()" || exit 1

# Default: run the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
