"""
Evaluation runner — runs a config over a split and saves results.

Usage:
    uv run python -m eval.run --config configs/ablations/a_dense.yaml --split dev
    uv run python -m eval.run --config configs/ablations/c_hybrid_rerank.yaml --split test

For each question in the dataset:
1. Run the full pipeline (search → rerank → generate)
2. Check if the relevant doc appeared in the top K
3. Judge faithfulness (using a different LLM)
4. Record all metrics and timings

Output: JSON results file + markdown summary table in eval/results/.

Throttling: sleeps between LLM calls to stay within free-tier rate limits.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

from docs_rag.config import get_config, get_config_name
from docs_rag.pipeline import answer as pipeline_answer
from eval.judge import judge_correctness, judge_faithfulness
from eval.metrics import QuestionResult, compute_all_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

DATASET_PATH = Path("eval/dataset/questions.jsonl")
RESULTS_DIR = Path("eval/results")

# Throttle between LLM calls to respect free-tier rate limits
THROTTLE_SECONDS = 3


def load_dataset(split: str) -> list[dict]:
    """Load evaluation questions for the given split."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    questions = []
    with open(DATASET_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            if q.get("split") == split:
                questions.append(q)

    logger.info("Loaded %d questions for split '%s'", len(questions), split)
    return questions


def evaluate_question(q: dict, config_path: str) -> QuestionResult:
    """Run the pipeline on one question and compute per-question metrics."""
    config = get_config(config_path)

    result = QuestionResult(
        question_id=q["id"],
        question=q["question"],
        answerable=q.get("answerable", True),
        relevant_paths=q.get("relevant", []),
    )

    try:
        # Run the full pipeline
        answer = pipeline_answer(question=q["question"], config_path=config_path)

        result.answer_text = answer.answer
        result.refused = answer.refused
        result.retrieve_ms = answer.timings.retrieve_ms
        result.rerank_ms = answer.timings.rerank_ms
        result.generate_ms = answer.timings.generate_ms
        result.total_ms = answer.timings.total_ms

        # Extract retrieved doc paths for recall computation
        result.retrieved_paths = [c.breadcrumb for c in answer.citations]

        # Also check against doc_path from the retrieved chunks
        # We need to re-retrieve to get the full chunk data
        # For now, use breadcrumbs for approximate matching
        for i, citation in enumerate(answer.citations, 1):
            for relevant in result.relevant_paths:
                doc_path = relevant.split("#")[0]
                if doc_path in citation.breadcrumb or doc_path in citation.url:
                    if result.first_relevant_rank is None:
                        result.first_relevant_rank = i
                    if i <= 5:
                        result.hit_at_5 = True
                    break

        # Judge faithfulness (only for non-refused answers)
        if not answer.refused and answer.citations:
            time.sleep(THROTTLE_SECONDS)  # Respect rate limits
            source_texts = [c.snippet for c in answer.citations]
            source_bcs = [c.breadcrumb for c in answer.citations]
            supported, unsupported = judge_faithfulness(
                answer.answer, source_texts, source_bcs
            )
            result.faithful = supported

        # Judge correctness (if reference answer exists)
        ref_answer = q.get("reference_answer")
        if ref_answer and not answer.refused:
            time.sleep(THROTTLE_SECONDS)
            correct, reason = judge_correctness(answer.answer, ref_answer)
            result.correct = correct

    except Exception as e:
        logger.error("Failed to evaluate question %s: %s", q["id"], e)
        result.answer_text = f"ERROR: {e}"

    return result


def save_results(results: list[QuestionResult], config_name: str, split: str) -> Path:
    """Save detailed results and a summary table."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{config_name}_{split}"

    # Detailed JSON results
    json_path = RESULTS_DIR / f"{filename}.json"
    data = {
        "config": config_name,
        "split": split,
        "timestamp": timestamp,
        "metrics": compute_all_metrics(results),
        "questions": [
            {
                "id": r.question_id,
                "question": r.question,
                "answerable": r.answerable,
                "hit_at_5": r.hit_at_5,
                "first_relevant_rank": r.first_relevant_rank,
                "refused": r.refused,
                "faithful": r.faithful,
                "correct": r.correct,
                "total_ms": r.total_ms,
                "answer_preview": r.answer_text[:200],
            }
            for r in results
        ],
    }
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    # Markdown summary
    metrics = data["metrics"]
    md_path = RESULTS_DIR / f"{filename}.md"
    md_content = f"""# Evaluation Results: {config_name} ({split} split)

**Run:** {timestamp}
**Questions:** {metrics['n_questions']} ({metrics['n_answerable']} answerable, {metrics['n_unanswerable']} unanswerable)

## Retrieval Metrics

| Metric | Value |
|--------|-------|
| Recall@5 | {metrics['recall_at_5']:.4f} |
| MRR@10 | {metrics['mrr_at_10']:.4f} |

## Generation Metrics

| Metric | Value |
|--------|-------|
| Faithfulness | {metrics['faithfulness']:.4f} |
| Answer correctness | {metrics['answer_correctness']:.4f} |

## Refusal Metrics

| Metric | Value |
|--------|-------|
| Refusal precision | {metrics['refusal_precision']:.4f} |
| Refusal recall | {metrics['refusal_recall']:.4f} |
| False refusal rate | {metrics['false_refusal_rate']:.4f} |

## Latency (ms)

| Stage | p50 | p95 | Mean |
|-------|-----|-----|------|
| Retrieve | {metrics['latency_retrieve_ms']['p50']:.0f} | {metrics['latency_retrieve_ms']['p95']:.0f} | {metrics['latency_retrieve_ms']['mean']:.0f} |
| Rerank | {metrics['latency_rerank_ms']['p50']:.0f} | {metrics['latency_rerank_ms']['p95']:.0f} | {metrics['latency_rerank_ms']['mean']:.0f} |
| Generate | {metrics['latency_generate_ms']['p50']:.0f} | {metrics['latency_generate_ms']['p95']:.0f} | {metrics['latency_generate_ms']['mean']:.0f} |
| Total | {metrics['latency_total_ms']['p50']:.0f} | {metrics['latency_total_ms']['p95']:.0f} | {metrics['latency_total_ms']['mean']:.0f} |

## Token Usage

| Metric | Value |
|--------|-------|
| Avg prompt tokens | {metrics['tokens']['avg_prompt']:.0f} |
| Avg completion tokens | {metrics['tokens']['avg_completion']:.0f} |
| Avg total tokens | {metrics['tokens']['avg_total']:.0f} |
"""
    with open(md_path, "w") as f:
        f.write(md_content)

    logger.info("Results saved: %s", json_path)
    logger.info("Summary saved: %s", md_path)
    return json_path


def main():
    parser = argparse.ArgumentParser(description="Run evaluation suite")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--split", required=True, choices=["dev", "test"], help="Dataset split")
    parser.add_argument("--limit", type=int, default=None, help="Max questions to evaluate (for quick testing)")
    args = parser.parse_args()

    config_name = get_config_name(args.config)
    logger.info("Running eval: config=%s, split=%s", config_name, args.split)

    # Load questions
    questions = load_dataset(args.split)
    if args.limit:
        questions = questions[:args.limit]

    # Evaluate each question
    results: list[QuestionResult] = []
    for i, q in enumerate(questions, 1):
        logger.info("[%d/%d] Evaluating: %.60s...", i, len(questions), q["question"])
        result = evaluate_question(q, args.config)
        results.append(result)

        # Throttle between questions
        if i < len(questions):
            time.sleep(THROTTLE_SECONDS)

    # Save results and print summary
    save_results(results, config_name, args.split)

    metrics = compute_all_metrics(results)
    print(f"\n{'='*50}")
    print(f" Eval Results: {config_name} ({args.split})")
    print(f"{'='*50}")
    print(f"  Recall@5:         {metrics['recall_at_5']:.4f}")
    print(f"  MRR@10:           {metrics['mrr_at_10']:.4f}")
    print(f"  Faithfulness:     {metrics['faithfulness']:.4f}")
    print(f"  Refusal recall:   {metrics['refusal_recall']:.4f}")
    print(f"  False refusals:   {metrics['false_refusal_rate']:.4f}")
    print(f"  Latency p50:      {metrics['latency_total_ms']['p50']:.0f}ms")
    print(f"  Latency p95:      {metrics['latency_total_ms']['p95']:.0f}ms")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
