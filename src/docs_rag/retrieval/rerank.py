"""
Reranker — cross-encoder rescoring of search candidates.

Design decisions:
- Protocol-based interface: any reranker just needs a `score(query, docs)` method.
  This makes it trivial to swap between local, hosted, or no-op rerankers.
- LocalCrossEncoder uses fastembed's TextCrossEncoder (ONNX, CPU).
  It's small (~80MB) and runs fast enough on 20 candidates.
- NoopReranker keeps the fused order unchanged (for configs A and B).
- The refusal threshold check is separate from reranking — it's applied
  after reranking in the pipeline, not inside the reranker.

Why rerank only 20 candidates: a cross-encoder reads (question + chunk) jointly,
which is much more accurate than comparing separate embeddings, but its cost
grows linearly with candidates. 20 is a sweet spot: enough to recover good chunks
that ranked poorly in one of the two search methods, few enough to stay fast.
"""

from __future__ import annotations

import logging
from typing import Protocol

from docs_rag.config import YAMLConfig, get_env_settings
from docs_rag.models import RetrievedChunk

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    """Interface: any reranker must implement score()."""

    def score(self, query: str, docs: list[str]) -> list[float]:
        """Return a relevance score for each (query, doc) pair."""
        ...


class LocalCrossEncoder:
    """Cross-encoder reranker using fastembed (ONNX, CPU).

    Loads the model once and reuses it. About 80MB in memory.
    """

    def __init__(self, model_name: str, cache_dir: str | None = None):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(
            model_name=model_name,
            cache_dir=cache_dir,
        )
        logger.info("Loaded local cross-encoder: %s", model_name)

    def score(self, query: str, docs: list[str]) -> list[float]:
        """Score each (query, doc) pair with the cross-encoder."""
        if not docs:
            return []
        results = list(self._model.rerank(query, docs))
        if results and isinstance(results[0], (int, float)):
            return [float(s) for s in results]

        scores = [0.0] * len(docs)
        for i, r in enumerate(results):
            if isinstance(r, dict):
                idx = r.get("index", i)
                sc = r.get("score", 0.0)
            elif hasattr(r, "score"):
                idx = getattr(r, "index", i)
                sc = getattr(r, "score", 0.0)
            else:
                idx = i
                sc = float(r)
            scores[idx] = float(sc)
        return scores


class NoopReranker:
    """Pass-through reranker — keeps the fused rank order.

    Used for ablation configs A (dense only) and B (hybrid, no rerank).
    Returns descending scores so the original order is preserved.
    """

    def score(self, query: str, docs: list[str]) -> list[float]:
        return [float(len(docs) - i) for i in range(len(docs))]


# Module-level singleton
_reranker: Reranker | None = None


def get_reranker(config: YAMLConfig | None = None) -> Reranker:
    """Return the singleton reranker based on config."""
    global _reranker
    if _reranker is not None:
        return _reranker

    if config is None:
        from docs_rag.config import get_config
        config = get_config()

    provider = config.rerank.provider
    if provider == "local":
        settings = get_env_settings()
        _reranker = LocalCrossEncoder(
            model_name=config.rerank.model,
            cache_dir=settings.model_cache_dir,
        )
    elif provider == "none":
        _reranker = NoopReranker()
    else:
        logger.warning("Unknown reranker provider '%s', using NoopReranker", provider)
        _reranker = NoopReranker()

    return _reranker


def rerank_chunks(
    query: str,
    candidates: list[RetrievedChunk],
    config: YAMLConfig,
) -> list[RetrievedChunk]:
    """Rerank candidates and return the top_n by rerank score.

    Args:
        query: The user's question.
        candidates: Retrieved chunks from hybrid search.
        config: YAML config with rerank settings.

    Returns:
        Top-N candidates sorted by rerank score (highest first).
    """
    if not candidates:
        return []

    reranker = get_reranker(config)
    top_n = config.rerank.top_n

    # Extract texts for scoring
    docs = [c.chunk.embed_text for c in candidates]
    scores = reranker.score(query, docs)

    # Annotate candidates with rerank scores
    for candidate, score in zip(candidates, scores):
        candidate.rerank_score = score

    # Sort by rerank score (highest first) and take top_n
    ranked = sorted(candidates, key=lambda c: c.rerank_score or 0.0, reverse=True)
    top = ranked[:top_n]

    logger.info(
        "Reranked %d → %d candidates. Top score: %.3f",
        len(candidates),
        len(top),
        top[0].rerank_score if top else 0.0,
    )
    return top
