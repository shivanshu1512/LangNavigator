"""
Main pipeline — orchestrates search → rerank → refuse-or-generate.

This is the single function that the API calls. It:
1. Encodes and searches (hybrid or dense-only based on config)
2. Reranks the top candidates with a cross-encoder
3. Checks the refusal threshold — if the best score is too low,
   it returns a refusal immediately WITHOUT calling the LLM (saves quota)
4. Otherwise, formats sources and calls the LLM with strict citation rules
5. Parses citations, maps them to URLs, and returns the full Answer

Every stage is timed independently so the API can report per-stage latency,
which is critical for identifying bottlenecks and demonstrating understanding
in interviews.
"""

from __future__ import annotations

import logging
import time
import uuid

from docs_rag.config import YAMLConfig, get_config, get_config_name, get_env_settings
from docs_rag.generation.llm import generate_answer
from docs_rag.generation.prompts import (
    format_system_prompt,
    format_user_prompt,
    get_refusal_sentence,
)
from docs_rag.models import Answer, Citation, RetrievedChunk, Timings
from docs_rag.retrieval.rerank import rerank_chunks
from docs_rag.retrieval.search import hybrid_search

logger = logging.getLogger(__name__)


def answer(
    question: str,
    config: YAMLConfig | None = None,
    config_path: str | None = None,
) -> Answer:
    """End-to-end: question in → grounded answer out.

    Args:
        question: The user's natural-language question.
        config: Pre-loaded YAML config. If None, loads from config_path or env.
        config_path: Path to YAML config file. Used to derive config name.

    Returns:
        An Answer object with the response, citations, timings, and metadata.
    """
    if config is None:
        config = get_config(config_path)

    settings = get_env_settings()
    framework = config.corpus.name
    request_id = uuid.uuid4().hex[:8]

    logger.info("[%s] Processing question: %.80s...", request_id, question)

    # ── Stage 1: Retrieve ──────────────────────────────────────────────
    t0 = time.perf_counter()
    candidates = hybrid_search(question, config, settings)
    retrieve_ms = (time.perf_counter() - t0) * 1000

    if not candidates:
        logger.warning("[%s] No candidates retrieved", request_id)
        return Answer(
            answer=get_refusal_sentence(framework),
            refused=True,
            timings=Timings(retrieve_ms=retrieve_ms, total_ms=retrieve_ms),
            config=get_config_name(config_path),
            model=settings.llm_model,
            request_id=request_id,
        )

    # ── Stage 2: Rerank ────────────────────────────────────────────────
    t1 = time.perf_counter()
    top_chunks = rerank_chunks(question, candidates, config)
    rerank_ms = (time.perf_counter() - t1) * 1000

    # ── Stage 3: Refusal check ─────────────────────────────────────────
    # If the best rerank score is below the threshold, refuse immediately
    # without calling the LLM. This saves quota and avoids hallucination.
    threshold = config.rerank.refusal_threshold
    if threshold is not None and top_chunks:
        best_score = top_chunks[0].rerank_score
        if best_score is not None and best_score < threshold:
            total_ms = retrieve_ms + rerank_ms
            logger.info(
                "[%s] Refusing: best score %.3f < threshold %.3f",
                request_id, best_score, threshold,
            )
            return Answer(
                answer=get_refusal_sentence(framework),
                refused=True,
                citations=_build_citations(top_chunks),
                timings=Timings(
                    retrieve_ms=retrieve_ms,
                    rerank_ms=rerank_ms,
                    total_ms=total_ms,
                ),
                config=get_config_name(config_path),
                model=settings.llm_model,
                request_id=request_id,
            )

    # ── Stage 4: Generate ──────────────────────────────────────────────
    system_prompt = format_system_prompt(framework)
    user_prompt = format_user_prompt(question, top_chunks)

    t2 = time.perf_counter()
    answer_text, refused, citation_nums, prompt_tokens, completion_tokens = generate_answer(
        question=question,
        sources=top_chunks,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        gen_config=config.generation,
        framework=framework,
    )
    generate_ms = (time.perf_counter() - t2) * 1000

    total_ms = retrieve_ms + rerank_ms + generate_ms

    # Build citations from the valid citation numbers
    citations = _build_citations(top_chunks, citation_nums)

    logger.info(
        "[%s] Done in %.0fms (retrieve=%.0f, rerank=%.0f, generate=%.0f). "
        "Refused=%s, citations=%d, tokens=%d+%d",
        request_id, total_ms, retrieve_ms, rerank_ms, generate_ms,
        refused, len(citations), prompt_tokens, completion_tokens,
    )

    return Answer(
        answer=answer_text,
        refused=refused,
        citations=citations,
        timings=Timings(
            retrieve_ms=round(retrieve_ms, 1),
            rerank_ms=round(rerank_ms, 1),
            generate_ms=round(generate_ms, 1),
            total_ms=round(total_ms, 1),
        ),
        config=get_config_name(config_path),
        model=settings.llm_model,
        request_id=request_id,
    )


def _build_citations(
    chunks: list[RetrievedChunk],
    citation_nums: list[int] | None = None,
) -> list[Citation]:
    """Convert retrieved chunks into Citation objects.

    If citation_nums is provided, only include chunks that were actually
    cited by the LLM. Otherwise, include all chunks (for refusals where
    we still want to show what was retrieved).
    """
    citations = []
    for i, rc in enumerate(chunks, start=1):
        if citation_nums is not None and i not in citation_nums:
            continue
        chunk = rc.chunk
        citations.append(Citation(
            n=i,
            breadcrumb=chunk.breadcrumb,
            url=chunk.url,
            snippet=chunk.text[:200],
            rerank_score=rc.rerank_score,
            fused_rank=rc.fused_rank,
        ))
    return citations
