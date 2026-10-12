"""
LLM client — calls Groq (or compatible providers) with retries and timeouts.

Design decisions:
- Uses the Groq Python SDK directly (no LangChain wrapper) so every step
  is visible and explainable.
- Retries on HTTP 429 (rate limit) with exponential backoff, up to max_retries.
  This is critical because free-tier Groq has strict per-minute limits.
- 30-second timeout prevents the API from hanging on slow requests.
- Model ID comes from config/env, never hard-coded, because Groq's model
  list changes frequently and hard-coding would break without warning.
"""

from __future__ import annotations

import logging
import re
import time

from groq import Groq, RateLimitError

from docs_rag.config import GenerationConfig, get_env_settings
from docs_rag.generation.prompts import get_refusal_sentence
from docs_rag.models import RetrievedChunk

logger = logging.getLogger(__name__)

# Module-level client singleton
_groq_client: Groq | None = None

# Regex to extract citation markers like [1], [2][3], etc.
CITATION_RE = re.compile(r"\[(\d+)\]")


def _get_groq_client() -> Groq:
    """Return a singleton Groq client."""
    global _groq_client
    if _groq_client is None:
        settings = get_env_settings()
        _groq_client = Groq(api_key=settings.groq_api_key)
    return _groq_client


def generate_answer(
    question: str,
    sources: list[RetrievedChunk],
    system_prompt: str,
    user_prompt: str,
    gen_config: GenerationConfig,
    framework: str,
    model: str | None = None,
) -> tuple[str, bool, list[int], int, int]:
    """Call the LLM and return (answer_text, refused, citation_numbers, prompt_tokens, completion_tokens).

    Handles:
    - Exponential backoff on rate limits (429)
    - Timeout enforcement
    - Refusal detection via exact string match
    - Citation number extraction

    Args:
        question: User's original question.
        sources: Retrieved chunks formatted as numbered sources.
        system_prompt: System instruction with citation rules.
        user_prompt: Sources + question formatted for the LLM.
        gen_config: Generation parameters (temperature, max_tokens, etc.).
        framework: Framework name for refusal sentence matching.
        model: LLM model ID override. Defaults to env setting.

    Returns:
        Tuple of (answer_text, is_refused, citation_numbers, prompt_tokens, completion_tokens).
    """
    if model is None:
        model = get_env_settings().llm_model

    client = _get_groq_client()
    last_error = None

    for attempt in range(gen_config.max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=gen_config.temperature,
                max_tokens=gen_config.max_tokens,
                timeout=gen_config.timeout_s,
            )

            answer_text = response.choices[0].message.content.strip()
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0

            # Check for refusal
            refusal_sentence = get_refusal_sentence(framework)
            refused = refusal_sentence.lower() in answer_text.lower()

            # Extract citation numbers
            citation_nums = [int(m) for m in CITATION_RE.findall(answer_text)]
            # Remove duplicates while preserving order
            seen = set()
            unique_citations = []
            for n in citation_nums:
                if n not in seen:
                    seen.add(n)
                    unique_citations.append(n)

            # Filter out citations pointing to non-existent sources
            max_source = len(sources)
            valid_citations = [n for n in unique_citations if 1 <= n <= max_source]

            logger.info(
                "LLM response: %d tokens, refused=%s, citations=%s",
                completion_tokens,
                refused,
                valid_citations,
            )
            return answer_text, refused, valid_citations, prompt_tokens, completion_tokens

        except RateLimitError as e:
            last_error = e
            if attempt < gen_config.max_retries:
                wait = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(
                    "Rate limited (attempt %d/%d). Waiting %ds...",
                    attempt + 1,
                    gen_config.max_retries,
                    wait,
                )
                time.sleep(wait)
            else:
                logger.error("Rate limit exceeded after %d retries", gen_config.max_retries)
                raise

        except Exception as e:
            last_error = e
            logger.error("LLM call failed: %s", e)
            raise

    # Should not reach here, but just in case
    raise last_error or RuntimeError("LLM call failed")
