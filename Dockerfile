FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_CACHE_DIR=/app/models

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install git (needed for GitPython if used at runtime)
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (cache-friendly layer ordering)
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev --no-install-project 2>/dev/null || uv sync --no-dev --no-install-project

# Copy source code and configs
COPY src ./src
COPY configs ./configs

# Install the project itself
RUN uv sync --frozen --no-dev 2>/dev/null || uv sync --no-dev

# Download embedding, BM25, and reranker models at build time
# so cold starts don't re-download them (saves 30-60s on first request)
RUN .venv/bin/python -m docs_rag.retrieval.encoders --warmup

# Expose the port (Render sets PORT automatically)
EXPOSE 8000

# Run the API server
CMD ["sh", "-c", ".venv/bin/uvicorn docs_rag.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
