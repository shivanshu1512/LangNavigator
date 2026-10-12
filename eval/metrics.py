"""
Evaluation metrics — retrieval accuracy, faithfulness, and refusal quality.

Metrics implemented:
- Recall@K (hit rate): did at least one relevant doc appear in the top K?
- MRR@K (Mean Reciprocal Rank): 1/rank of the first relevant doc (0 if missed)
- Refusal precision/recall: on unanswerable questions did it refuse?
  On answerable questions did it wrongly refuse?
- Latency statistics: p50, p95, mean per stage and total

These are the metrics that fill the ablation table — the centerpiece
of the project that proves each design choice actually helps.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field


@dataclass
class QuestionResult:
    """Result for a single evaluation question."""

    question_id: str
    question: str
    answerable: bool
    relevant_paths: list[str]  # Expected doc_path#anchor values

    # Retrieval results
    retrieved_paths: list[str] = field(default_factory=list)  # Ordered by rank
    hit_at_5: bool = False
    first_relevant_rank: int | None = None  # 1-based, None if not found

    # Generation results
    answer_text: str = ""
    refused: bool = False
    faithful: bool | None = None  # LLM judge result
    correct: bool | None = None  # Answer matches reference

    # Timings (ms)
    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    generate_ms: float = 0.0
    total_ms: float = 0.0

    # Token usage
    prompt_tokens: int = 0
    completion_tokens: int = 0


def recall_at_k(results: list[QuestionResult], k: int = 5) -> float:
    """Share of answerable questions where at least one relevant doc is in top K.

    This is the most important retrieval metric: if the right document
    never makes it into the top K, the LLM can't possibly answer correctly.
    """
    answerable = [r for r in results if r.answerable]
    if not answerable:
        return 0.0

    hits = 0
    for r in answerable:
        # Check if any relevant path appears in the top K retrieved paths
        top_k_paths = r.retrieved_paths[:k]
        for relevant in r.relevant_paths:
            # Match on doc_path prefix (anchor may vary)
            doc_path = relevant.split("#")[0]
            if any(doc_path in p for p in top_k_paths):
                hits += 1
                break

    return hits / len(answerable)


def mrr_at_k(results: list[QuestionResult], k: int = 10) -> float:
    """Mean Reciprocal Rank — average of 1/rank for the first relevant doc.

    Higher MRR means relevant docs tend to appear earlier in the ranking,
    which matters because the LLM pays more attention to top sources.
    """
    answerable = [r for r in results if r.answerable]
    if not answerable:
        return 0.0

    rr_sum = 0.0
    for r in answerable:
        if r.first_relevant_rank is not None and r.first_relevant_rank <= k:
            rr_sum += 1.0 / r.first_relevant_rank

    return rr_sum / len(answerable)


def faithfulness_rate(results: list[QuestionResult]) -> float:
    """Share of non-refused answers where the LLM judge said every claim is supported."""
    judged = [r for r in results if not r.refused and r.faithful is not None]
    if not judged:
        return 0.0
    return sum(1 for r in judged if r.faithful) / len(judged)


def answer_correctness_rate(results: list[QuestionResult]) -> float:
    """Share of non-refused answers that match the reference answer's meaning."""
    judged = [r for r in results if not r.refused and r.correct is not None]
    if not judged:
        return 0.0
    return sum(1 for r in judged if r.correct) / len(judged)


def refusal_precision(results: list[QuestionResult]) -> float:
    """Among questions that were refused, what fraction were truly unanswerable?

    High precision = it rarely refuses answerable questions (few false refusals).
    """
    refused = [r for r in results if r.refused]
    if not refused:
        return 0.0
    return sum(1 for r in refused if not r.answerable) / len(refused)


def refusal_recall(results: list[QuestionResult]) -> float:
    """Among truly unanswerable questions, what fraction did it refuse?

    High recall = it catches most unanswerable questions instead of hallucinating.
    """
    unanswerable = [r for r in results if not r.answerable]
    if not unanswerable:
        return 0.0
    return sum(1 for r in unanswerable if r.refused) / len(unanswerable)


def false_refusal_rate(results: list[QuestionResult]) -> float:
    """Share of answerable questions that were wrongly refused."""
    answerable = [r for r in results if r.answerable]
    if not answerable:
        return 0.0
    return sum(1 for r in answerable if r.refused) / len(answerable)


def latency_stats(results: list[QuestionResult], field: str = "total_ms") -> dict:
    """Compute p50, p95, mean for a latency field."""
    values = [getattr(r, field) for r in results if getattr(r, field, 0) > 0]
    if not values:
        return {"p50": 0, "p95": 0, "mean": 0}

    values.sort()
    n = len(values)
    return {
        "p50": values[n // 2],
        "p95": values[int(n * 0.95)] if n > 1 else values[0],
        "mean": statistics.mean(values),
    }


def token_stats(results: list[QuestionResult]) -> dict:
    """Average prompt and completion tokens per query."""
    if not results:
        return {"avg_prompt": 0, "avg_completion": 0, "avg_total": 0}
    return {
        "avg_prompt": statistics.mean(r.prompt_tokens for r in results),
        "avg_completion": statistics.mean(r.completion_tokens for r in results),
        "avg_total": statistics.mean(r.prompt_tokens + r.completion_tokens for r in results),
    }


def compute_all_metrics(results: list[QuestionResult]) -> dict:
    """Compute all metrics and return as a flat dictionary."""
    return {
        "n_questions": len(results),
        "n_answerable": sum(1 for r in results if r.answerable),
        "n_unanswerable": sum(1 for r in results if not r.answerable),
        "recall_at_5": round(recall_at_k(results, k=5), 4),
        "mrr_at_10": round(mrr_at_k(results, k=10), 4),
        "faithfulness": round(faithfulness_rate(results), 4),
        "answer_correctness": round(answer_correctness_rate(results), 4),
        "refusal_precision": round(refusal_precision(results), 4),
        "refusal_recall": round(refusal_recall(results), 4),
        "false_refusal_rate": round(false_refusal_rate(results), 4),
        "latency_total_ms": latency_stats(results, "total_ms"),
        "latency_retrieve_ms": latency_stats(results, "retrieve_ms"),
        "latency_rerank_ms": latency_stats(results, "rerank_ms"),
        "latency_generate_ms": latency_stats(results, "generate_ms"),
        "tokens": token_stats(results),
    }
