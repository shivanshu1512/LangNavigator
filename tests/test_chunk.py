"""
Tests for the chunker.

Critical invariants tested:
1. No chunk ever contains an odd number of code fences (split code block)
2. Chunk sizes respect the configured limits
3. Small sections are merged
4. Every chunk has a URL
5. Breadcrumbs are preserved from the original sections
"""

from pathlib import Path

import pytest

from docs_rag.config import load_yaml_config
from docs_rag.ingest.chunk import (
    _approx_tokens,
    _has_balanced_fences,
    _split_into_blocks,
    chunk_sections,
    merge_small_sections,
)
from docs_rag.ingest.parse import parse_file
from docs_rag.models import Section

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def config():
    """Load the base config for chunk size parameters."""
    return load_yaml_config("configs/base.yaml")


@pytest.fixture
def sample_sections():
    """Parse the sample doc into sections."""
    return parse_file(FIXTURES / "sample_doc.md", "test/sample_doc.md")


class TestBalancedFences:
    """The most critical invariant: code blocks must never be split."""

    def test_balanced_empty(self):
        assert _has_balanced_fences("")

    def test_balanced_no_fences(self):
        assert _has_balanced_fences("Just some text\nMore text")

    def test_balanced_one_block(self):
        text = "before\n```python\ncode\n```\nafter"
        assert _has_balanced_fences(text)

    def test_unbalanced_one_fence(self):
        text = "before\n```python\ncode without closing"
        assert not _has_balanced_fences(text)

    def test_balanced_two_blocks(self):
        text = "```\nblock1\n```\ntext\n```\nblock2\n```"
        assert _has_balanced_fences(text)


class TestSplitIntoBlocks:
    """Blocks should keep code blocks as single units."""

    def test_separates_paragraphs(self):
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        blocks = _split_into_blocks(text)
        assert len(blocks) == 3

    def test_code_block_is_one_unit(self):
        text = "Text before.\n\n```python\nline1\nline2\nline3\n```\n\nText after."
        blocks = _split_into_blocks(text)
        # Code block should be one block
        code_blocks = [b for b in blocks if "```" in b]
        assert len(code_blocks) == 1
        assert "line1" in code_blocks[0]
        assert "line3" in code_blocks[0]

    def test_multiple_code_blocks(self):
        text = "```\na\n```\n\ntext\n\n```\nb\n```"
        blocks = _split_into_blocks(text)
        code_blocks = [b for b in blocks if "```" in b]
        assert len(code_blocks) == 2


class TestMergeSmallSections:
    """Small sections should be merged to improve retrieval quality."""

    def test_tiny_section_merged(self):
        sections = [
            Section(
                title="T", breadcrumb="A", heading_slug="a",
                body="Tiny.", doc_path="test.md", level=2,
            ),
            Section(
                title="T", breadcrumb="B", heading_slug="b",
                body="This is a much longer section with enough words " * 10,
                doc_path="test.md", level=2,
            ),
        ]
        merged = merge_small_sections(sections, min_tokens=80)
        # The tiny section should be merged into the second
        assert len(merged) == 1
        assert "Tiny." in merged[0].body

    def test_large_sections_not_merged(self):
        body = "This is a sufficiently long section body. " * 20
        sections = [
            Section(
                title="T", breadcrumb="A", heading_slug="a",
                body=body, doc_path="test.md", level=2,
            ),
            Section(
                title="T", breadcrumb="B", heading_slug="b",
                body=body, doc_path="test.md", level=2,
            ),
        ]
        merged = merge_small_sections(sections, min_tokens=80)
        assert len(merged) == 2


class TestChunkSections:
    """Integration tests for the full chunking pipeline."""

    def test_all_chunks_have_balanced_fences(self, sample_sections, config):
        """THE critical invariant: no chunk should split a code block."""
        chunks = chunk_sections(
            sample_sections, config,
            FIXTURES / "sample_doc.md", "abc123", "https://docs.example.com",
        )
        for chunk in chunks:
            assert _has_balanced_fences(chunk.text), (
                f"Chunk {chunk.chunk_index} has unbalanced fences:\n{chunk.text[:200]}"
            )

    def test_chunks_have_urls(self, sample_sections, config):
        """Every chunk must have a URL for citation linking."""
        chunks = chunk_sections(
            sample_sections, config,
            FIXTURES / "sample_doc.md", "abc123", "https://docs.example.com",
        )
        for chunk in chunks:
            assert chunk.url, f"Chunk {chunk.chunk_index} missing URL"
            assert chunk.url.startswith("https://")

    def test_chunks_have_breadcrumbs(self, sample_sections, config):
        """Every chunk must carry its section's breadcrumb."""
        chunks = chunk_sections(
            sample_sections, config,
            FIXTURES / "sample_doc.md", "abc123", "https://docs.example.com",
        )
        for chunk in chunks:
            assert chunk.breadcrumb, f"Chunk {chunk.chunk_index} missing breadcrumb"

    def test_chunk_ids_are_deterministic(self, sample_sections, config):
        """Same input should always produce the same chunk IDs."""
        chunks1 = chunk_sections(
            sample_sections, config,
            FIXTURES / "sample_doc.md", "abc123", "https://docs.example.com",
        )
        chunks2 = chunk_sections(
            sample_sections, config,
            FIXTURES / "sample_doc.md", "abc123", "https://docs.example.com",
        )
        ids1 = [c.id for c in chunks1]
        ids2 = [c.id for c in chunks2]
        assert ids1 == ids2

    def test_chunk_sizes_within_limits(self, sample_sections, config):
        """Most chunks should respect max_tokens (hard_max_tokens is the absolute limit)."""
        chunks = chunk_sections(
            sample_sections, config,
            FIXTURES / "sample_doc.md", "abc123", "https://docs.example.com",
        )
        for chunk in chunks:
            assert chunk.n_tokens <= config.chunking.hard_max_tokens, (
                f"Chunk {chunk.chunk_index} exceeds hard max: {chunk.n_tokens} tokens"
            )

    def test_no_empty_chunks(self, sample_sections, config):
        """Chunks with no text should not be emitted."""
        chunks = chunk_sections(
            sample_sections, config,
            FIXTURES / "sample_doc.md", "abc123", "https://docs.example.com",
        )
        for chunk in chunks:
            assert chunk.text.strip(), f"Empty chunk at index {chunk.chunk_index}"


class TestApproxTokens:
    """Token approximation tests."""

    def test_empty_string(self):
        assert _approx_tokens("") == 0

    def test_single_word(self):
        assert _approx_tokens("hello") == 1  # 1 * 1.3 = 1.3, int = 1

    def test_sentence(self):
        tokens = _approx_tokens("This is a simple test sentence with eight words")
        assert 8 <= tokens <= 15  # Roughly 8 * 1.3 = 10.4
