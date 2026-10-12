"""
Tests for the search module.

Note: These tests require a running Qdrant instance (either local Docker
or the CI service container). They test the actual hybrid search behavior,
not mocked interactions.
"""

from __future__ import annotations

from unittest.mock import patch

from docs_rag.config import YAMLConfig
from docs_rag.models import Chunk, RetrievedChunk
from docs_rag.retrieval.rerank import NoopReranker, rerank_chunks


class TestNoopReranker:
    """Tests for the pass-through reranker."""

    def test_preserves_order(self):
        reranker = NoopReranker()
        scores = reranker.score("test query", ["doc1", "doc2", "doc3"])
        assert len(scores) == 3
        # Scores should be descending (preserving original order)
        assert scores[0] > scores[1] > scores[2]

    def test_empty_docs(self):
        reranker = NoopReranker()
        scores = reranker.score("test query", [])
        assert scores == []


class TestRerankChunks:
    """Tests for the rerank_chunks function."""

    def _make_retrieved(self, n: int) -> list[RetrievedChunk]:
        """Create n mock RetrievedChunk objects."""
        chunks = []
        for i in range(n):
            chunk = Chunk(
                id=f"chunk_{i}",
                text=f"Text for chunk {i}",
                breadcrumb=f"Section {i}",
                title="Test",
                doc_path="test.md",
                url=f"https://example.com#s{i}",
                chunk_index=i,
                n_tokens=50,
                file_hash="abc",
            )
            chunks.append(RetrievedChunk(
                chunk=chunk,
                fused_rank=i + 1,
                fused_score=1.0 / (60 + i),
            ))
        return chunks

    def test_returns_top_n(self):
        """Rerank should return at most top_n chunks."""
        candidates = self._make_retrieved(10)
        config_data = {
            "rerank": {"provider": "none", "model": "", "top_n": 5, "refusal_threshold": None}
        }
        config = YAMLConfig(config_data)

        # Force the NoopReranker
        with patch("docs_rag.retrieval.rerank._reranker", NoopReranker()):
            result = rerank_chunks("test query", candidates, config)

        assert len(result) == 5

    def test_empty_candidates(self):
        """Empty input should return empty output."""
        config_data = {
            "rerank": {"provider": "none", "model": "", "top_n": 5, "refusal_threshold": None}
        }
        config = YAMLConfig(config_data)

        result = rerank_chunks("test query", [], config)
        assert result == []

    def test_annotates_rerank_scores(self):
        """Each returned chunk should have a rerank_score set."""
        candidates = self._make_retrieved(3)
        config_data = {
            "rerank": {"provider": "none", "model": "", "top_n": 5, "refusal_threshold": None}
        }
        config = YAMLConfig(config_data)

        with patch("docs_rag.retrieval.rerank._reranker", NoopReranker()):
            result = rerank_chunks("test query", candidates, config)

        for chunk in result:
            assert chunk.rerank_score is not None
