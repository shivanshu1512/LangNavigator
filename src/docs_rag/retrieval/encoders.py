"""
Dense and sparse encoders — loaded once at startup, reused per request.

Design decisions:
- fastembed (ONNX, CPU) keeps us free-tier and avoids PyTorch overhead.
- Models are loaded lazily (on first use) but cached as module-level singletons
  so we don't re-download or re-load on each request.
- The --warmup CLI entry point pre-loads all models at Docker build time,
  so cold starts don't include download time.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

from docs_rag.config import get_config, get_env_settings

logger = logging.getLogger(__name__)

# Module-level singletons — loaded once, reused forever
_dense_model: Any = None
_sparse_model: Any = None


def get_dense_model():
    """Return the singleton dense embedding model."""
    global _dense_model
    if _dense_model is None:
        from fastembed import TextEmbedding

        config = get_config()
        settings = get_env_settings()
        logger.info("Loading dense model: %s", config.retrieval.dense_model)
        _dense_model = TextEmbedding(
            model_name=config.retrieval.dense_model,
            cache_dir=settings.model_cache_dir,
        )
    return _dense_model


def get_sparse_model():
    """Return the singleton BM25 sparse embedding model."""
    global _sparse_model
    if _sparse_model is None:
        from fastembed import SparseTextEmbedding

        config = get_config()
        settings = get_env_settings()
        logger.info("Loading sparse model: %s", config.retrieval.sparse_model)
        _sparse_model = SparseTextEmbedding(
            model_name=config.retrieval.sparse_model,
            cache_dir=settings.model_cache_dir,
        )
    return _sparse_model


def encode_query_dense(question: str) -> list[float]:
    """Encode a question into a dense vector for retrieval."""
    model = get_dense_model()
    # query_embed adds the "query: " prefix for asymmetric models like BGE
    vector = next(model.query_embed(question))
    return vector.tolist()


def encode_query_sparse(question: str) -> tuple[list[int], list[float]]:
    """Encode a question into sparse BM25 indices and values."""
    model = get_sparse_model()
    sparse = next(model.query_embed(question))
    return sparse.indices.tolist(), sparse.values.tolist()


# ---------------------------------------------------------------------------
# Warmup CLI — called during Docker build to pre-cache models
# ---------------------------------------------------------------------------

def warmup():
    """Pre-load all models so Docker image includes them in the layer cache.

    Called by: RUN .venv/bin/python -m docs_rag.retrieval.encoders --warmup
    """
    logger.info("Warming up models...")
    get_dense_model()
    get_sparse_model()

    # Also warm up the reranker if configured
    config = get_config()
    settings = get_env_settings()
    if config.rerank.provider == "local":
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        logger.info("Loading reranker model: %s", config.rerank.model)
        TextCrossEncoder(
            model_name=config.rerank.model,
            cache_dir=settings.model_cache_dir,
        )
    logger.info("All models warmed up ✓")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()
    if args.warmup:
        warmup()
