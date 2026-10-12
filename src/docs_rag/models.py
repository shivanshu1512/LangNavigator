"""
Pydantic models for the entire pipeline.

These are the data contracts that flow between stages:
  parse → Section → chunk → Chunk → index (Qdrant)
  search → RetrievedChunk → rerank → RetrievedChunk → generate → Answer

Using strict pydantic models means every stage's output is validated,
which makes bugs surface early instead of silently propagating bad data.
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Ingestion models
# ---------------------------------------------------------------------------

class Section(BaseModel):
    """One heading-delimited section extracted from a markdown file."""

    title: str = Field(description="Page title from front matter or first H1")
    breadcrumb: str = Field(description="Heading path, e.g. 'Persistence > Checkpointers'")
    heading_slug: str = Field(description="Anchor slug for the URL, e.g. 'checkpointers'")
    body: str = Field(description="Text content under this heading")
    doc_path: str = Field(description="Relative path inside the docs repo")
    level: int = Field(description="Heading level (1-6)")
    has_code: bool = Field(default=False, description="Whether body contains fenced code blocks")


class Chunk(BaseModel):
    """One indexable unit ready for embedding and storage in Qdrant."""

    id: str = Field(description="Deterministic UUID based on doc_path and chunk_index")
    text: str = Field(description="Chunk text shown to the LLM")
    breadcrumb: str = Field(description="Section heading path")
    title: str = Field(description="Page title")
    doc_path: str = Field(description="Path inside the docs repo")
    url: str = Field(description="Public docs URL including heading anchor")
    chunk_index: int = Field(description="Position within the source file")
    has_code: bool = Field(default=False, description="Contains a fenced code block")
    n_tokens: int = Field(description="Approximate token count")
    file_hash: str = Field(description="SHA-256 of the source file for incremental indexing")
    source_commit: str = Field(default="", description="Git commit SHA at indexing time")

    @property
    def embed_text(self) -> str:
        """What gets embedded: breadcrumb + body for context-enriched retrieval.

        Adding the section path helps short chunks that don't mention their
        own topic (e.g. a code snippet under a heading).
        """
        return f"{self.breadcrumb}\n\n{self.text}"

    @staticmethod
    def make_id(doc_path: str, chunk_index: int) -> str:
        """Deterministic UUID from path and index — same input always gives same ID."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_path}#{chunk_index}"))


# ---------------------------------------------------------------------------
# Retrieval models
# ---------------------------------------------------------------------------

class RetrievedChunk(BaseModel):
    """A chunk returned from search, annotated with scores."""

    chunk: Chunk
    fused_rank: int = Field(description="Rank in the fused result list (1-based)")
    fused_score: float = Field(default=0.0, description="RRF fusion score")
    rerank_score: float | None = Field(default=None, description="Cross-encoder score, if reranked")


# ---------------------------------------------------------------------------
# API / generation models
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    """One citation in the answer, linking to a doc section."""

    n: int = Field(description="Citation number, e.g. 1 for [1]")
    breadcrumb: str
    url: str
    snippet: str = Field(description="First ~200 chars of the chunk")
    rerank_score: float | None = None
    fused_rank: int = 0


class Timings(BaseModel):
    """Per-stage latency in milliseconds."""

    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    generate_ms: float = 0.0
    total_ms: float = 0.0


class Answer(BaseModel):
    """The full response returned by the pipeline and the API."""

    answer: str
    refused: bool = False
    citations: list[Citation] = Field(default_factory=list)
    timings: Timings = Field(default_factory=Timings)
    config: str = ""
    model: str = ""
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
