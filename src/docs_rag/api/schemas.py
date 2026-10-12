"""
Request/response schemas for the FastAPI endpoints.

These are separate from the internal pipeline models (models.py) because
the API surface may evolve differently from the internal data contracts.
Using dedicated schemas also lets us control exactly what's exposed to clients
and add validation rules specific to the HTTP layer (e.g. question length limits).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """POST /query request body."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The question to answer from the documentation",
        examples=["How do I persist state between runs in LangGraph?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of source chunks to return",
    )


class CitationResponse(BaseModel):
    """One citation in the API response."""

    n: int = Field(description="Citation number, e.g. 1 for [1]")
    breadcrumb: str = Field(description="Section heading path")
    url: str = Field(description="Public docs URL with anchor")
    snippet: str = Field(description="First ~200 chars of the chunk text")
    rerank_score: float | None = Field(default=None, description="Cross-encoder relevance score")
    fused_rank: int = Field(default=0, description="Rank in the fused result list")


class TimingsResponse(BaseModel):
    """Per-stage latency in milliseconds."""

    retrieve: float = Field(default=0.0, description="Retrieval stage latency")
    rerank: float = Field(default=0.0, description="Reranking stage latency")
    generate: float = Field(default=0.0, description="LLM generation latency")
    total: float = Field(default=0.0, description="End-to-end latency")


class QueryResponse(BaseModel):
    """POST /query response body."""

    answer: str = Field(description="The grounded answer with [n] citations")
    refused: bool = Field(description="Whether the system refused to answer (not in docs)")
    citations: list[CitationResponse] = Field(default_factory=list)
    timings_ms: TimingsResponse = Field(default_factory=TimingsResponse)
    config: str = Field(default="", description="Which ablation config was used")
    model: str = Field(default="", description="LLM model ID")
    request_id: str = Field(default="", description="Unique request identifier")


class HealthResponse(BaseModel):
    """GET /health response body."""

    status: str = "ok"
    collection: str = ""
    points_count: int = 0
    source_commit: str = ""
    config: str = ""


class FeedbackRequest(BaseModel):
    """POST /feedback request body (stretch goal)."""

    request_id: str = Field(..., description="The request_id from the query response")
    helpful: bool = Field(..., description="True for thumbs up, False for thumbs down")
    comment: str = Field(default="", max_length=500)
