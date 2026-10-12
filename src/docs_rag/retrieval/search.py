"""
Hybrid search against Qdrant — dense + BM25 fused with RRF.

Design decisions:
- Uses Qdrant's prefetch + FusionQuery to run both searches and fuse results
  in a single round-trip. This is much faster than two separate queries + manual fusion.
- RRF (Reciprocal Rank Fusion) uses only ranks, not scores, so it works
  correctly even though dense cosine similarity and BM25 scores are on
  completely different scales.
- Config A (dense only) simply skips the sparse prefetch.
"""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient, models

from docs_rag.config import EnvSettings, YAMLConfig
from docs_rag.models import Chunk, RetrievedChunk
from docs_rag.retrieval.encoders import encode_query_dense, encode_query_sparse

logger = logging.getLogger(__name__)

# Module-level Qdrant client singleton
_client: QdrantClient | None = None


def get_client(settings: EnvSettings | None = None) -> QdrantClient:
    """Return a singleton Qdrant client."""
    global _client
    if _client is None:
        if settings is None:
            from docs_rag.config import get_env_settings
            settings = get_env_settings()
        kwargs = {"url": settings.qdrant_url}
        if settings.qdrant_api_key:
            kwargs["api_key"] = settings.qdrant_api_key
        _client = QdrantClient(**kwargs)
    return _client


def hybrid_search(
    question: str,
    config: YAMLConfig,
    settings: EnvSettings | None = None,
) -> list[RetrievedChunk]:
    """Run hybrid (dense + BM25) search with RRF fusion.

    Args:
        question: The user's question.
        config: YAML config controlling retrieval behavior.
        settings: Environment settings for Qdrant connection.

    Returns:
        List of RetrievedChunk objects, ranked by fused score.
    """
    if settings is None:
        from docs_rag.config import get_env_settings
        settings = get_env_settings()

    client = get_client(settings)
    collection = settings.qdrant_collection
    limit = config.retrieval.prefetch_limit

    # Encode the question
    dense_vec = encode_query_dense(question)

    if config.retrieval.use_sparse:
        # Hybrid: dense + BM25 with RRF fusion
        sparse_indices, sparse_values = encode_query_sparse(question)

        result = client.query_points(
            collection_name=collection,
            prefetch=[
                models.Prefetch(
                    query=dense_vec,
                    using="dense",
                    limit=limit,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_indices,
                        values=sparse_values,
                    ),
                    using="bm25",
                    limit=limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
    else:
        # Dense only (config A)
        result = client.query_points(
            collection_name=collection,
            query=dense_vec,
            using="dense",
            limit=limit,
            with_payload=True,
        )

    # Convert Qdrant points to RetrievedChunk objects
    retrieved: list[RetrievedChunk] = []
    for rank, point in enumerate(result.points, start=1):
        payload = point.payload or {}
        chunk = Chunk(
            id=str(point.id),
            text=payload.get("text", ""),
            breadcrumb=payload.get("breadcrumb", ""),
            title=payload.get("title", ""),
            doc_path=payload.get("doc_path", ""),
            url=payload.get("url", ""),
            chunk_index=payload.get("chunk_index", 0),
            has_code=payload.get("has_code", False),
            n_tokens=payload.get("n_tokens", 0),
            file_hash=payload.get("file_hash", ""),
            source_commit=payload.get("source_commit", ""),
        )
        retrieved.append(RetrievedChunk(
            chunk=chunk,
            fused_rank=rank,
            fused_score=point.score if point.score else 0.0,
        ))

    logger.info(
        "Retrieved %d chunks (hybrid=%s) for: %.50s...",
        len(retrieved),
        config.retrieval.use_sparse,
        question,
    )
    return retrieved
