"""
Embed chunks and upsert them into Qdrant.

Handles both full rebuilds and incremental indexing:
- Full: delete the collection, recreate it, upsert everything.
- Incremental: compare file hashes against a manifest, only re-index changed files.

Incremental indexing is critical for a docs repo that changes frequently:
  1. Hash each file → compare against the saved manifest.
  2. Changed files: delete their old chunks by doc_path filter, then upsert new ones.
  3. Deleted files: delete their old chunks.
  4. Unchanged files: skip entirely.

This delete-then-upsert strategy avoids stale chunks when a file gets shorter
(removing sections), which a pure upsert would miss.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from qdrant_client import QdrantClient, models

from docs_rag.config import EnvSettings
from docs_rag.models import Chunk

logger = logging.getLogger(__name__)

MANIFEST_PATH = Path("data/index_manifest.json")
BATCH_SIZE = 64


def _get_client(settings: EnvSettings) -> QdrantClient:
    """Create a Qdrant client from environment settings."""
    kwargs = {"url": settings.qdrant_url}
    if settings.qdrant_api_key:
        kwargs["api_key"] = settings.qdrant_api_key
    return QdrantClient(**kwargs)


def ensure_collection(client: QdrantClient, collection_name: str) -> None:
    """Create the collection if it doesn't exist.

    Uses named vectors so we can have both 'dense' and 'bm25' in one collection,
    queried independently and fused with RRF.
    """
    collections = [c.name for c in client.get_collections().collections]
    if collection_name in collections:
        logger.info("Collection '%s' already exists", collection_name)
        return

    logger.info("Creating collection '%s'", collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=384,  # bge-small-en-v1.5 output dimension
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            ),
        },
    )

    # Payload index on doc_path for fast filtered deletes during incremental indexing
    client.create_payload_index(
        collection_name=collection_name,
        field_name="doc_path",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    logger.info("Collection '%s' created with dense + bm25 vectors", collection_name)


def recreate_collection(client: QdrantClient, collection_name: str) -> None:
    """Drop and recreate the collection (for full rebuilds)."""
    collections = [c.name for c in client.get_collections().collections]
    if collection_name in collections:
        logger.info("Deleting collection '%s' for full rebuild", collection_name)
        client.delete_collection(collection_name)
    ensure_collection(client, collection_name)


def upsert_chunks(
    client: QdrantClient,
    collection_name: str,
    chunks: list[Chunk],
    dense_vectors: list[list[float]],
    sparse_vectors: list[tuple[list[int], list[float]]] | None = None,
) -> None:
    """Upsert chunks with their vectors in batches.

    Args:
        client: Qdrant client.
        collection_name: Target collection.
        chunks: Chunk objects with metadata.
        dense_vectors: Dense embedding vectors aligned with chunks.
        sparse_vectors: Optional BM25 sparse vectors as (indices, values) tuples.
    """
    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[i : i + BATCH_SIZE]
        batch_dense = dense_vectors[i : i + BATCH_SIZE]
        batch_sparse = sparse_vectors[i : i + BATCH_SIZE] if sparse_vectors else None

        points = []
        for j, chunk in enumerate(batch_chunks):
            vectors: dict = {"dense": batch_dense[j]}
            if batch_sparse:
                indices, values = batch_sparse[j]
                vectors["bm25"] = models.SparseVector(
                    indices=indices,
                    values=values,
                )

            point = models.PointStruct(
                id=chunk.id,
                vector=vectors,
                payload=chunk.model_dump(exclude={"id"}),
            )
            points.append(point)

        client.upsert(collection_name=collection_name, points=points)
        logger.debug("Upserted batch of %d points", len(points))

    logger.info("Upserted %d chunks total", len(chunks))


def delete_by_doc_path(client: QdrantClient, collection_name: str, doc_path: str) -> None:
    """Delete all chunks belonging to a specific doc file."""
    client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_path",
                        match=models.MatchValue(value=doc_path),
                    )
                ]
            )
        ),
    )
    logger.debug("Deleted chunks for doc_path: %s", doc_path)


# ---------------------------------------------------------------------------
# Manifest for incremental indexing
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: Path | None = None) -> dict[str, str]:
    """Load the index manifest (doc_path → file_hash mapping)."""
    path = manifest_path or MANIFEST_PATH
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save_manifest(manifest: dict[str, str], manifest_path: Path | None = None) -> None:
    """Save the index manifest to disk."""
    path = manifest_path or MANIFEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved manifest with %d entries", len(manifest))
