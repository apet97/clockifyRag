# Multi-stage build to keep the runtime image small and reproducible
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Build-time system dependencies for numpy/torch/faiss wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libblas-dev \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files (pyproject is the source of truth for deps)
COPY . .

# Install the package and runtime dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir .

# -----------------------------------------------------------------------------
# Runtime image
# -----------------------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Runtime libraries required by faiss / numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed site-packages (includes console scripts)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application sources (data, scripts, docs)
COPY --from=builder /app /app

# Create directories for runtime artifacts
RUN mkdir -p /app/data /app/logs

# Non-root user for safety
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

# Default configuration aligns with clockify_rag.config
ENV RAG_OLLAMA_URL=http://host.docker.internal:11434 \
    OLLAMA_URL=http://host.docker.internal:11434 \
    RAG_LOG_FILE=/app/logs/rag_queries.jsonl

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from clockify_rag.error_handlers import print_system_health; import sys; sys.exit(0 if print_system_health() else 1)" > /dev/null

EXPOSE 8000

# Default to serving the FastAPI application (override for CLI usage)
CMD ["uvicorn", "clockify_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
