"""
CLI entry point for the ingestion pipeline.

Usage:
    uv run python -m docs_rag.ingest --config configs/base.yaml --full
    uv run python -m docs_rag.ingest --config configs/base.yaml --dry-run
    uv run python -m docs_rag.ingest --config configs/base.yaml --limit 100

Stages (per batch of files):
    1. Fetch (clone/pull) the docs repo
    2. Parse markdown into sections
    3. Chunk sections respecting code-block boundaries
    4. Embed (dense + sparse) and upsert into Qdrant
    5. Save manifest checkpoint

The --dry-run flag prints chunks without embedding or upserting.
The --limit flag caps the number of files to process.

Batched processing:
    Files are processed in batches of FILES_PER_BATCH. After each batch,
    chunks are embedded, upserted into Qdrant, and the manifest is saved.
    This means progress is checkpointed after every batch — if the process
    is interrupted, the next run (without --full) will skip already-indexed
    files and continue from where it left off.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time

from docs_rag.config import get_config, get_env_settings
from docs_rag.ingest.chunk import chunk_sections, merge_small_sections
from docs_rag.ingest.fetch import fetch_docs, list_doc_files
from docs_rag.ingest.index import (
    _get_client,
    delete_by_doc_path,
    ensure_collection,
    load_manifest,
    recreate_collection,
    save_manifest,
    upsert_chunks,
)
from docs_rag.ingest.parse import parse_file
from docs_rag.models import Chunk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Number of files to process per batch before checkpointing
FILES_PER_BATCH = 50


# ---------------------------------------------------------------------------
# Embedding helpers (lazy imports to avoid loading ONNX unless needed)
# ---------------------------------------------------------------------------

_dense_model = None
_sparse_model = None


def _get_dense_model(config):
    """Lazy-load and cache the dense embedding model."""
    global _dense_model
    if _dense_model is None:
        from fastembed import TextEmbedding
        settings = get_env_settings()
        logger.info("Loading dense model: %s", config.retrieval.dense_model)
        _dense_model = TextEmbedding(
            model_name=config.retrieval.dense_model,
            cache_dir=settings.model_cache_dir,
        )
    return _dense_model


def _get_sparse_model(config):
    """Lazy-load and cache the sparse embedding model."""
    global _sparse_model
    if _sparse_model is None:
        from fastembed import SparseTextEmbedding
        settings = get_env_settings()
        logger.info("Loading sparse model: %s", config.retrieval.sparse_model)
        _sparse_model = SparseTextEmbedding(
            model_name=config.retrieval.sparse_model,
            cache_dir=settings.model_cache_dir,
        )
    return _sparse_model


def _embed_chunks(
    chunks: list[Chunk], config
) -> tuple[list[list[float]], list[tuple[list[int], list[float]]] | None]:
    """Embed chunks with dense and optionally sparse models.

    Models are loaded once and reused across batches.
    """
    embed_texts = [c.embed_text for c in chunks]

    dense_model = _get_dense_model(config)
    logger.info("Embedding %d chunks with dense model...", len(chunks))
    dense_vectors = list(dense_model.passage_embed(embed_texts))
    dense_vectors = [v.tolist() for v in dense_vectors]

    sparse_vectors = None
    if config.retrieval.use_sparse:
        sparse_model = _get_sparse_model(config)
        logger.info("Embedding %d chunks with BM25 model...", len(chunks))
        sparse_embeddings = list(sparse_model.embed(embed_texts))
        sparse_vectors = [
            (emb.indices.tolist(), emb.values.tolist())
            for emb in sparse_embeddings
        ]

    return dense_vectors, sparse_vectors


def main():
    parser = argparse.ArgumentParser(description="Ingest documentation into Qdrant")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--full", action="store_true", help="Full rebuild (delete and recreate collection)")
    parser.add_argument("--dry-run", action="store_true", help="Parse and chunk only, don't embed or upsert")
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max number of files to process (0 = all). Useful for quick testing.",
    )
    args = parser.parse_args()

    settings = get_env_settings()
    config = get_config(args.config)

    # 1. Fetch docs
    logger.info("=== Stage 1: Fetching docs ===")
    repo_path, commit_sha = fetch_docs(config)

    # 2. List files
    doc_files = list_doc_files(repo_path, config)
    if not doc_files:
        logger.error("No doc files found! Check your config globs.")
        sys.exit(1)

    # 3. Determine docs root
    docs_root = repo_path / config.corpus.docs_subdir
    if not docs_root.exists():
        docs_root = repo_path

    # 4. Build hashes for all files and decide which need processing
    logger.info("=== Stage 2: Scanning files for changes ===")
    old_manifest = {} if args.full else load_manifest()
    file_hashes: dict[str, str] = {}
    files_to_process: list[tuple[str, "Path"]] = []  # (doc_path, file_path)

    for file_path in doc_files:
        doc_path = file_path.relative_to(docs_root).as_posix()
        current_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
        file_hashes[doc_path] = current_hash

        # Skip unchanged files in incremental mode
        if not args.full and old_manifest.get(doc_path) == current_hash:
            continue

        files_to_process.append((doc_path, file_path))

    # Apply --limit
    if args.limit > 0:
        files_to_process = files_to_process[:args.limit]
        logger.info("Limiting to %d files (--limit)", args.limit)

    logger.info(
        "Total files: %d | To process: %d | Already indexed: %d",
        len(doc_files),
        len(files_to_process),
        len(doc_files) - len(files_to_process),
    )

    if args.dry_run:
        # Dry-run: parse and chunk all files, print summary
        all_chunks: list[Chunk] = []
        for doc_path, file_path in files_to_process:
            sections = parse_file(file_path, doc_path)
            sections = merge_small_sections(sections, config.chunking.min_tokens)
            chunks = chunk_sections(
                sections, config, file_path, commit_sha, config.corpus.base_url,
            )
            all_chunks.extend(chunks)

        for chunk in all_chunks[:20]:
            print(f"\n{'='*60}")
            print(f"  Breadcrumb: {chunk.breadcrumb}")
            print(f"  Doc path:   {chunk.doc_path}")
            print(f"  URL:        {chunk.url}")
            print(f"  Tokens:     {chunk.n_tokens}")
            print(f"  Has code:   {chunk.has_code}")
            print(f"  Text preview: {chunk.text[:200]}...")
        if len(all_chunks) > 20:
            print(f"\n... and {len(all_chunks) - 20} more chunks")
        print(f"\nTotal: {len(all_chunks)} chunks from {len(files_to_process)} files")
        return

    if not files_to_process and not args.full:
        logger.info("No changes detected. Nothing to index.")
        save_manifest(file_hashes)
        return

    # 5. Connect to Qdrant and prepare the collection
    client = _get_client(settings)

    if args.full:
        recreate_collection(client, settings.qdrant_collection)
    else:
        ensure_collection(client, settings.qdrant_collection)

    # 6. Process files in batches — embed, upsert, and checkpoint after each
    logger.info("=== Stage 3+4: Batched embedding & upserting ===")
    total_chunks_upserted = 0
    total_batches = (len(files_to_process) + FILES_PER_BATCH - 1) // FILES_PER_BATCH

    for batch_idx in range(0, len(files_to_process), FILES_PER_BATCH):
        batch_num = batch_idx // FILES_PER_BATCH + 1
        batch_files = files_to_process[batch_idx : batch_idx + FILES_PER_BATCH]
        batch_start = time.perf_counter()

        logger.info(
            "── Batch %d/%d: processing %d files ──",
            batch_num, total_batches, len(batch_files),
        )

        # Parse and chunk this batch
        batch_chunks: list[Chunk] = []
        for doc_path, file_path in batch_files:
            sections = parse_file(file_path, doc_path)
            sections = merge_small_sections(sections, config.chunking.min_tokens)
            chunks = chunk_sections(
                sections, config, file_path, commit_sha, config.corpus.base_url,
            )
            batch_chunks.extend(chunks)

        if not batch_chunks:
            logger.info("  Batch %d: no chunks produced, skipping", batch_num)
            continue

        # Embed this batch
        dense_vectors, sparse_vectors = _embed_chunks(batch_chunks, config)

        # Delete old chunks for changed files (incremental mode)
        if not args.full:
            changed_paths = {c.doc_path for c in batch_chunks}
            for doc_path in changed_paths:
                delete_by_doc_path(client, settings.qdrant_collection, doc_path)

        # Upsert this batch
        upsert_chunks(
            client, settings.qdrant_collection,
            batch_chunks, dense_vectors, sparse_vectors,
        )
        total_chunks_upserted += len(batch_chunks)

        # Checkpoint: save manifest with all hashes we've processed so far
        # Include both previously-indexed files and newly-processed ones
        checkpoint_manifest = dict(old_manifest)  # start with old entries
        # Update with all file hashes for files processed so far
        for dp, _ in files_to_process[:batch_idx + FILES_PER_BATCH]:
            checkpoint_manifest[dp] = file_hashes[dp]
        # Also include unchanged files
        for dp, h in file_hashes.items():
            if dp not in checkpoint_manifest:
                checkpoint_manifest[dp] = h
        save_manifest(checkpoint_manifest)

        batch_elapsed = time.perf_counter() - batch_start
        logger.info(
            "  Batch %d done: %d chunks in %.1fs | Total upserted: %d",
            batch_num, len(batch_chunks), batch_elapsed, total_chunks_upserted,
        )

    # Handle deleted files (files in old manifest but not in current file list)
    if not args.full:
        removed_paths = set(old_manifest.keys()) - set(file_hashes.keys())
        for doc_path in removed_paths:
            delete_by_doc_path(client, settings.qdrant_collection, doc_path)
        if removed_paths:
            logger.info("Deleted chunks for %d removed files", len(removed_paths))

    # Final manifest save with all file hashes
    save_manifest(file_hashes)

    # Summary
    collection_info = client.get_collection(settings.qdrant_collection)
    logger.info(
        "Done! Collection '%s' has %d points. Commit: %s",
        settings.qdrant_collection,
        collection_info.points_count,
        commit_sha,
    )


if __name__ == "__main__":
    main()
