"""
Markdown/MDX parser — converts raw files into structured Sections.

Key design decisions:
1. Split on H1–H3 headings only. Deeper headings stay within their parent
   section, which keeps chunks big enough to be useful.
2. Never treat a heading inside a fenced code block as a real heading.
   A line like `# comment` in Python code would otherwise corrupt the parse.
3. Strip YAML front matter but keep its `title` field for metadata.
4. For MDX files: remove import/export lines and JSX-only tags (e.g. <Tabs>)
   but keep text content inside components.
5. Build a breadcrumb (heading path) for each section, e.g.
   "Persistence > Checkpointers > Using a database".
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from docs_rag.models import Section

logger = logging.getLogger(__name__)

# Regex patterns
FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
FENCE_RE = re.compile(r"^(`{3,}|~{3,})")
MDX_IMPORT_RE = re.compile(r"^(import|export)\s+")
MDX_JSX_ONLY_RE = re.compile(r"^</?[A-Z]\w*[^>]*>?\s*$")
TITLE_RE = re.compile(r'title:\s*["\']?(.+?)["\']?\s*$', re.MULTILINE)


def _extract_front_matter_title(text: str) -> tuple[str, str]:
    """Strip YAML front matter, return (title_from_frontmatter, remaining_text).

    We keep the title because it's often more descriptive than the filename.
    """
    match = FRONT_MATTER_RE.match(text)
    if not match:
        return "", text

    fm_block = match.group(1)
    remaining = text[match.end():]

    title_match = TITLE_RE.search(fm_block)
    title = title_match.group(1) if title_match else ""
    return title, remaining


def _slugify(text: str) -> str:
    """Convert heading text to a URL anchor slug.

    Follows the same convention most markdown renderers use:
    lowercase, spaces→dashes, strip non-alphanumeric except dashes.
    """
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = slug.strip("-")
    return slug


def _clean_mdx(line: str) -> str | None:
    """Return the cleaned line, or None if it should be dropped entirely.

    MDX files contain import/export statements and JSX-only tags that
    carry no text content. We remove those but keep text inside components.
    """
    if MDX_IMPORT_RE.match(line.strip()):
        return None
    if MDX_JSX_ONLY_RE.match(line.strip()):
        return None
    return line


def parse_file(file_path: Path, doc_path: str) -> list[Section]:
    """Parse a single markdown/MDX file into a list of Sections.

    Args:
        file_path: Absolute path to the file on disk.
        doc_path: Relative path inside the docs repo (for metadata).

    Returns:
        List of Section objects, one per H1–H3 heading found.
    """
    text = file_path.read_text(encoding="utf-8", errors="replace")
    is_mdx = file_path.suffix.lower() in (".mdx",)

    # 1. Extract and strip front matter
    fm_title, text = _extract_front_matter_title(text)

    lines = text.split("\n")
    sections: list[Section] = []

    # Track heading hierarchy for breadcrumbs
    heading_stack: list[tuple[int, str]] = []  # [(level, text), ...]
    current_heading_text = fm_title or file_path.stem.replace("-", " ").replace("_", " ").title()
    current_heading_slug = _slugify(current_heading_text)
    current_heading_level = 1
    body_lines: list[str] = []

    in_code_fence = False
    fence_marker = ""

    for line in lines:
        # Track fenced code blocks — headings inside code are NOT headings
        fence_match = FENCE_RE.match(line.strip())
        if fence_match:
            marker = fence_match.group(1)[0]  # ` or ~
            marker_len = len(fence_match.group(1))
            if not in_code_fence:
                in_code_fence = True
                fence_marker = marker * marker_len
            elif line.strip().startswith(marker) and len(line.strip()) >= len(fence_marker):
                in_code_fence = False
                fence_marker = ""
            body_lines.append(line)
            continue

        if in_code_fence:
            body_lines.append(line)
            continue

        # MDX cleanup
        if is_mdx:
            cleaned = _clean_mdx(line)
            if cleaned is None:
                continue
            line = cleaned

        # Check for a heading
        heading_match = HEADING_RE.match(line.strip())
        if heading_match:
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()

            # Only split on H1–H3 to keep sections big enough
            if level <= 3:
                # Save the current section
                body = "\n".join(body_lines).strip()
                if body or sections:  # skip empty preamble before first heading
                    has_code = "```" in body or "~~~" in body
                    breadcrumb = _build_breadcrumb(heading_stack, current_heading_text)
                    sections.append(Section(
                        title=fm_title or current_heading_text,
                        breadcrumb=breadcrumb,
                        heading_slug=current_heading_slug,
                        body=body,
                        doc_path=doc_path,
                        level=current_heading_level,
                        has_code=has_code,
                    ))

                # Update heading stack
                # Pop headings at same or deeper level (they're siblings/children)
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, heading_text))

                current_heading_text = heading_text
                current_heading_slug = _slugify(heading_text)
                current_heading_level = level
                body_lines = []
                continue

        body_lines.append(line)

    # Don't forget the last section
    body = "\n".join(body_lines).strip()
    if body:
        has_code = "```" in body or "~~~" in body
        breadcrumb = _build_breadcrumb(heading_stack, current_heading_text)
        sections.append(Section(
            title=fm_title or current_heading_text,
            breadcrumb=breadcrumb,
            heading_slug=current_heading_slug,
            body=body,
            doc_path=doc_path,
            level=current_heading_level,
            has_code=has_code,
        ))

    # Filter out empty sections
    sections = [s for s in sections if s.body.strip()]
    logger.debug("Parsed %d sections from %s", len(sections), doc_path)
    return sections


def _build_breadcrumb(heading_stack: list[tuple[int, str]], current: str) -> str:
    """Build a breadcrumb string from the heading hierarchy.

    Example: if the stack is [(1, "Concepts"), (2, "Persistence")]
    and current is "Checkpointers", returns "Concepts > Persistence > Checkpointers".
    """
    parts = [text for _, text in heading_stack]
    if current and (not parts or parts[-1] != current):
        parts.append(current)
    return " > ".join(parts) if parts else current
