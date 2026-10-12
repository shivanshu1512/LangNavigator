"""
Prompt templates for the LLM.

Design decisions:
- System prompt enforces strict citation rules: every claim must cite [n].
- The exact refusal sentence is prescribed so we can reliably detect it
  in the response (instead of fuzzy matching).
- Sources are numbered and include breadcrumbs + URLs so the LLM can
  reference specific sections, and we can map citations to clickable links.
- Temperature is kept low (0.1) to minimize creative deviations from sources.
- Word limit (~200 words) keeps answers concise and reduces hallucination risk
  by forcing the model to stay close to the source material.
"""

from __future__ import annotations

from docs_rag.models import RetrievedChunk

# The refusal sentence — must match exactly for detection
REFUSAL_TEMPLATE = "I couldn't find this in the {framework} docs."

SYSTEM_PROMPT = """You are a documentation assistant for {framework}.
Answer ONLY using the numbered sources provided by the user.

Rules:
1. End every factual sentence with a citation such as [1] or [2][3].
2. If the sources do not contain the answer, reply exactly:
   "I couldn't find this in the {framework} docs."
   and nothing else.
3. Use code from the sources when helpful. Never invent function names,
   parameters, imports or version numbers.
4. Be concise: at most about 200 words unless the user asks for more."""


def format_system_prompt(framework: str) -> str:
    """Format the system prompt with the framework name."""
    return SYSTEM_PROMPT.format(framework=framework)


def format_user_prompt(question: str, sources: list[RetrievedChunk]) -> str:
    """Format the user message with numbered sources and the question.

    The sources include breadcrumb and URL so the LLM can produce
    meaningful citations, and we can map [n] markers to clickable links.
    """
    source_blocks = []
    for i, src in enumerate(sources, start=1):
        chunk = src.chunk
        block = f"[{i}] {chunk.breadcrumb} ({chunk.url})\n{chunk.text}"
        source_blocks.append(block)

    sources_text = "\n\n".join(source_blocks)
    return f"Sources:\n{sources_text}\n\nQuestion: {question}"


def get_refusal_sentence(framework: str) -> str:
    """Return the exact refusal sentence for detection."""
    return REFUSAL_TEMPLATE.format(framework=framework)
