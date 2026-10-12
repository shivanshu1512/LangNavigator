"""
LLM-as-judge for faithfulness evaluation.

Uses a DIFFERENT (preferably stronger) model than the generator to avoid
self-evaluation bias — models tend to rate their own outputs favorably.

The judge answers one binary question per answer:
"Is EVERY factual claim supported by the given sources?"

Results are cached on disk (keyed by hash of model + prompt) so re-running
the eval after changing only retrieval doesn't re-spend judge LLM quota.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

from groq import Groq, RateLimitError

from docs_rag.config import get_env_settings

logger = logging.getLogger(__name__)

CACHE_DIR = Path("eval/cache")

JUDGE_PROMPT = """You are grading whether an answer is fully supported by the given sources.

Sources:
{numbered_sources}

Answer:
{answer}

Is EVERY factual claim in the answer supported by the sources?
Reply with JSON only: {{"supported": true or false, "unsupported_claims": ["..."]}}"""


CORRECTNESS_PROMPT = """You are comparing a generated answer against a reference answer.

Reference answer:
{reference_answer}

Generated answer:
{generated_answer}

Does the generated answer convey the same core meaning as the reference?
Minor wording differences are OK. Missing key information is NOT OK.
Reply with JSON only: {{"correct": true or false, "reason": "..."}}"""


def _cache_key(model: str, prompt: str) -> str:
    """Deterministic hash for caching judge responses."""
    content = f"{model}::{prompt}"
    return hashlib.sha256(content.encode()).hexdigest()


def _get_cached(key: str) -> dict | None:
    """Look up a cached judge response."""
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    return None


def _save_cache(key: str, result: dict) -> None:
    """Cache a judge response to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{key}.json"
    with open(cache_file, "w") as f:
        json.dump(result, f)


def _call_judge(prompt: str, model: str | None = None, api_key: str | None = None) -> dict:
    """Call the judge LLM and parse its JSON response."""
    settings = get_env_settings()
    model = model or settings.judge_model
    api_key = api_key or settings.judge_api_key or settings.groq_api_key

    client = Groq(api_key=api_key)

    # Check cache first
    cache_key = _cache_key(model, prompt)
    cached = _get_cached(cache_key)
    if cached is not None:
        logger.debug("Judge cache hit: %s", cache_key[:12])
        return cached

    # Call with retries for rate limits
    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
                timeout=30,
            )
            text = response.choices[0].message.content.strip()

            # Parse JSON from the response (handle markdown code blocks)
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            result = json.loads(text)
            _save_cache(cache_key, result)
            return result

        except RateLimitError:
            wait = 2 ** attempt
            logger.warning("Judge rate limited, waiting %ds...", wait)
            time.sleep(wait)
        except json.JSONDecodeError as e:
            logger.warning("Judge returned invalid JSON: %s", e)
            result = {"supported": None, "error": str(e)}
            _save_cache(cache_key, result)
            return result
        except Exception as e:
            logger.error("Judge call failed: %s", e)
            return {"supported": None, "error": str(e)}

    return {"supported": None, "error": "Rate limit exceeded after retries"}


def judge_faithfulness(
    answer_text: str,
    source_texts: list[str],
    source_breadcrumbs: list[str],
) -> tuple[bool | None, list[str]]:
    """Judge whether every claim in the answer is supported by the sources.

    Args:
        answer_text: The generated answer to evaluate.
        source_texts: The text of each source chunk.
        source_breadcrumbs: The breadcrumb of each source chunk.

    Returns:
        Tuple of (is_supported, list_of_unsupported_claims).
        is_supported is None if the judge failed.
    """
    # Format sources
    numbered = []
    for i, (text, bc) in enumerate(zip(source_texts, source_breadcrumbs), 1):
        numbered.append(f"[{i}] {bc}\n{text}")
    sources_str = "\n\n".join(numbered)

    prompt = JUDGE_PROMPT.format(numbered_sources=sources_str, answer=answer_text)
    result = _call_judge(prompt)

    supported = result.get("supported")
    claims = result.get("unsupported_claims", [])
    return supported, claims


def judge_correctness(
    generated_answer: str,
    reference_answer: str,
) -> tuple[bool | None, str]:
    """Judge whether the generated answer matches the reference answer's meaning.

    Args:
        generated_answer: The system's answer.
        reference_answer: The expected correct answer.

    Returns:
        Tuple of (is_correct, reason). is_correct is None if judge failed.
    """
    prompt = CORRECTNESS_PROMPT.format(
        reference_answer=reference_answer,
        generated_answer=generated_answer,
    )
    result = _call_judge(prompt)

    correct = result.get("correct")
    reason = result.get("reason", "")
    return correct, reason
