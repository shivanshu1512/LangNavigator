"""
FastAPI application — routes, middleware, and error handling.

Design decisions:
- Rate limiting (10 req/min/IP) via slowapi protects free-tier API keys from abuse.
- Structured JSON logging per request for observability.
- CORS is NOT enabled because Streamlit calls the API server-side (not from browser).
- The /health endpoint returns collection stats and commit SHA, useful for
  verifying deployment and debugging.
- All upstream failures (Qdrant, Groq) return 503 with a readable message,
  not a raw stack trace.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from docs_rag.api.schemas import (
    CitationResponse,
    FeedbackRequest,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    TimingsResponse,
)
from docs_rag.config import get_config, get_config_name, get_env_settings
from docs_rag.pipeline import answer as pipeline_answer

logger = logging.getLogger(__name__)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)


# ---------------------------------------------------------------------------
# App lifespan — pre-load models on startup so the first request is fast
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load embedding and reranker models at startup.

    This runs once when the server starts. Loading models here means
    the first user request doesn't pay the model-loading penalty.
    """
    logger.info("Starting up — pre-loading models...")
    try:
        from docs_rag.retrieval.encoders import get_dense_model, get_sparse_model
        from docs_rag.retrieval.rerank import get_reranker

        get_dense_model()
        config = get_config()
        if config.retrieval.use_sparse:
            get_sparse_model()
        get_reranker(config)
        logger.info("Models loaded ✓")
    except Exception as e:
        logger.error("Failed to pre-load models: %s", e)
        # Don't crash — let the server start and try loading on first request
    yield
    logger.info("Shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Docs RAG Assistant",
    description=(
        "Q&A over framework documentation with hybrid retrieval, "
        "cross-encoder reranking, and grounded citations."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Rate limiting — 10 requests per minute per IP
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please wait before retrying."},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness check with collection stats and deployment info."""
    try:
        settings = get_env_settings()
        from qdrant_client import QdrantClient

        kwargs = {"url": settings.qdrant_url}
        if settings.qdrant_api_key:
            kwargs["api_key"] = settings.qdrant_api_key
        client = QdrantClient(**kwargs)

        collection_info = client.get_collection(settings.qdrant_collection)

        # Get the source commit from any point's payload
        source_commit = ""
        try:
            sample = client.scroll(
                collection_name=settings.qdrant_collection,
                limit=1,
                with_payload=["source_commit"],
            )
            if sample[0]:
                source_commit = sample[0][0].payload.get("source_commit", "")
        except Exception:
            pass

        return HealthResponse(
            status="ok",
            collection=settings.qdrant_collection,
            points_count=collection_info.points_count or 0,
            source_commit=source_commit,
            config=get_config_name(),
        )
    except Exception as e:
        logger.error("Health check failed: %s", e)
        return HealthResponse(status="degraded")


@app.post("/query", response_model=QueryResponse, tags=["Query"])
@limiter.limit("10/minute")
async def query(request: Request, body: QueryRequest):
    """Answer a question using the documentation.

    The pipeline: hybrid search → rerank → refuse or generate with citations.
    """
    t_start = time.perf_counter()

    try:
        result = pipeline_answer(question=body.question)

        # Map internal Answer model to API response schema
        response = QueryResponse(
            answer=result.answer,
            refused=result.refused,
            citations=[
                CitationResponse(
                    n=c.n,
                    breadcrumb=c.breadcrumb,
                    url=c.url,
                    snippet=c.snippet,
                    rerank_score=c.rerank_score,
                    fused_rank=c.fused_rank,
                )
                for c in result.citations
            ],
            timings_ms=TimingsResponse(
                retrieve=result.timings.retrieve_ms,
                rerank=result.timings.rerank_ms,
                generate=result.timings.generate_ms,
                total=result.timings.total_ms,
            ),
            config=result.config,
            model=result.model,
            request_id=result.request_id,
        )

        # Structured log line — one per request for observability
        log_entry = {
            "request_id": result.request_id,
            "question_len": len(body.question),
            "refused": result.refused,
            "n_citations": len(result.citations),
            "timings_ms": result.timings.model_dump(),
            "top_rerank_score": (
                result.citations[0].rerank_score if result.citations else None
            ),
        }
        logger.info("Query completed: %s", json.dumps(log_entry))

        return response

    except Exception as e:
        elapsed = (time.perf_counter() - t_start) * 1000
        logger.error("Query failed after %.0fms: %s", elapsed, e, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"Service temporarily unavailable: {type(e).__name__}",
        )


@app.post("/feedback", tags=["Feedback"])
@limiter.limit("20/minute")
async def feedback(request: Request, body: FeedbackRequest):
    """Record user feedback (thumbs up/down) for an answer.

    Stretch goal — logs feedback for later analysis.
    """
    logger.info(
        "Feedback: request_id=%s, helpful=%s, comment=%.100s",
        body.request_id,
        body.helpful,
        body.comment,
    )
    return {"status": "recorded"}
