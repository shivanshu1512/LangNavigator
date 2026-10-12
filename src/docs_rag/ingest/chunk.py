"""
Chunker — converts Sections into Chunks ready for embedding.

Design decisions:
1. The unit of chunking is a Section (one heading-delimited block).
2. Tiny sections (< min_tokens) are merged into the next section with the
   same parent heading, because very small chunks retrieve poorly.
3. Long sections are split into multiple chunks by paragraphs and code blocks:
   - A fenced code block is treated as one indivisible block (never split mid-code).
   - Blocks are greedily packed into chunks up to max_tokens.
   - Each new chunk starts with the last paragraph of the previous one (overlap).
4. A single block larger than hard_max_tokens is split by lines as a last resort,
   but this should be rare.
5. Token counts are approximated as word_count * 1.3, which is close enough for
   English text and much faster than running a real tokenizer on every chunk.

Critical invariant: no chunk ever contains an odd number of code fences.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from docs_rag.config import YAMLConfig
from docs_rag.models import Chunk, Section

logger = logging.getLogger(__name__)

FENCE_RE = re.compile(r"^(`{3,}|~{3,})")


def _approx_tokens(text: str) -> int:
    """Approximate token count. Words × 1.3 is a fast heuristic that's
    close enough for English and avoids loading a tokenizer."""
    return int(len(text.split()) * 1.3)


def _file_hash(file_path: Path) -> str:
    """SHA-256 of file contents — used for incremental indexing to detect changes."""
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def _split_into_blocks(text: str) -> list[str]:
    """Split text into paragraphs and whole code blocks.

    A "block" is either:
    - A fenced code block (``` ... ```) treated as one unit
    - A paragraph (text separated by blank lines)

    This ensures code blocks are never split in the middle.
    """
    lines = text.split("\n")
    blocks: list[str] = []
    current_block: list[str] = []
    in_code = False

    for line in lines:
        fence_match = FENCE_RE.match(line.strip())

        if fence_match and not in_code:
            # Start of code block — flush any paragraph first
            if current_block:
                para = "\n".join(current_block).strip()
                if para:
                    blocks.append(para)
                current_block = []
            in_code = True
            current_block.append(line)
        elif fence_match and in_code:
            # End of code block
            current_block.append(line)
            blocks.append("\n".join(current_block))
            current_block = []
            in_code = False
        elif in_code:
            current_block.append(line)
        elif line.strip() == "":
            # Blank line = paragraph boundary
            if current_block:
                para = "\n".join(current_block).strip()
                if para:
                    blocks.append(para)
                current_block = []
        else:
            current_block.append(line)

    # Flush remaining
    if current_block:
        remaining = "\n".join(current_block).strip()
        if remaining:
            blocks.append(remaining)

    return blocks


def _split_large_block(block: str, hard_max: int) -> list[str]:
    """Split a single oversized block by lines as a last resort.

    This only triggers for extremely long code blocks. Each piece is
    marked with "(continued)" in practice by the caller.
    """
    lines = block.split("\n")
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for line in lines:
        line_tokens = _approx_tokens(line)
        if current_tokens + line_tokens > hard_max and current:
            pieces.append("\n".join(current))
            current = [line]
            current_tokens = line_tokens
        else:
            current.append(line)
            current_tokens += line_tokens

    if current:
        pieces.append("\n".join(current))

    return pieces


def _has_balanced_fences(text: str) -> bool:
    """Check that text has an even number of code fences (no split code blocks)."""
    count = 0
    for line in text.split("\n"):
        if FENCE_RE.match(line.strip()):
            count += 1
    return count % 2 == 0


def merge_small_sections(sections: list[Section], min_tokens: int) -> list[Section]:
    """Merge sections smaller than min_tokens into their next sibling.

    Very small sections (e.g. a heading with one line of text) retrieve
    poorly because there's not enough content for the embedding to match on.
    Merging them into the next section at the same parent level keeps the
    semantic grouping while making each chunk more substantial.
    """
    if not sections:
        return []

    merged: list[Section] = []
    pending: Section | None = None

    for section in sections:
        if pending is not None:
            # Merge pending into this section
            combined_body = pending.body + "\n\n" + section.body
            section = Section(
                title=section.title,
                breadcrumb=section.breadcrumb,
                heading_slug=section.heading_slug,
                body=combined_body,
                doc_path=section.doc_path,
                level=section.level,
                has_code=pending.has_code or section.has_code,
            )
            pending = None

        if _approx_tokens(section.body) < min_tokens:
            pending = section
        else:
            merged.append(section)

    # If the last section was too small, append it anyway
    if pending is not None:
        if merged:
            # Merge into the last section
            last = merged[-1]
            combined_body = last.body + "\n\n" + pending.body
            merged[-1] = Section(
                title=last.title,
                breadcrumb=last.breadcrumb,
                heading_slug=last.heading_slug,
                body=combined_body,
                doc_path=last.doc_path,
                level=last.level,
                has_code=last.has_code or pending.has_code,
            )
        else:
            merged.append(pending)

    return merged


def chunk_sections(
    sections: list[Section],
    config: YAMLConfig,
    file_path: Path,
    commit_sha: str,
    base_url: str,
) -> list[Chunk]:
    """Convert sections into chunks respecting size limits.

    Args:
        sections: Parsed and optionally merged sections from one file.
        config: YAML config with chunking parameters.
        file_path: Path to the source file (for hashing).
        commit_sha: Current git commit SHA.
        base_url: Docs site base URL for constructing public links.

    Returns:
        List of Chunk objects ready for embedding.
    """
    cc = config.chunking
    fhash = _file_hash(file_path)

    chunks: list[Chunk] = []
    chunk_idx = 0

    for section in sections:
        tokens = _approx_tokens(section.body)

        # Build the public URL for this section
        doc_path_clean = section.doc_path
        # Remove file extension for URL
        url_path = re.sub(r"\.(md|mdx)$", "", doc_path_clean)
        url = f"{base_url.rstrip('/')}/{url_path}#{section.heading_slug}" if base_url else ""

        if tokens <= cc.max_tokens:
            # Section fits in one chunk
            chunk = Chunk(
                id=Chunk.make_id(section.doc_path, chunk_idx),
                text=section.body,
                breadcrumb=section.breadcrumb,
                title=section.title,
                doc_path=section.doc_path,
                url=url,
                chunk_index=chunk_idx,
                has_code=section.has_code,
                n_tokens=tokens,
                file_hash=fhash,
                source_commit=commit_sha,
            )
            chunks.append(chunk)
            chunk_idx += 1
        else:
            # Section is too large — split by blocks
            blocks = _split_into_blocks(section.body)
            current_text_parts: list[str] = []
            current_tokens = 0
            last_paragraph: str | None = None

            for block in blocks:
                block_tokens = _approx_tokens(block)

                # Handle oversized single blocks
                if block_tokens > cc.hard_max_tokens:
                    # Flush current
                    if current_text_parts:
                        text = "\n\n".join(current_text_parts)
                        chunks.append(Chunk(
                            id=Chunk.make_id(section.doc_path, chunk_idx),
                            text=text,
                            breadcrumb=section.breadcrumb,
                            title=section.title,
                            doc_path=section.doc_path,
                            url=url,
                            chunk_index=chunk_idx,
                            has_code="```" in text or "~~~" in text,
                            n_tokens=_approx_tokens(text),
                            file_hash=fhash,
                            source_commit=commit_sha,
                        ))
                        chunk_idx += 1
                        current_text_parts = []
                        current_tokens = 0

                    # Split the large block itself
                    sub_blocks = _split_large_block(block, cc.hard_max_tokens)
                    for sb in sub_blocks:
                        chunks.append(Chunk(
                            id=Chunk.make_id(section.doc_path, chunk_idx),
                            text=sb,
                            breadcrumb=section.breadcrumb + " (continued)",
                            title=section.title,
                            doc_path=section.doc_path,
                            url=url,
                            chunk_index=chunk_idx,
                            has_code="```" in sb or "~~~" in sb,
                            n_tokens=_approx_tokens(sb),
                            file_hash=fhash,
                            source_commit=commit_sha,
                        ))
                        chunk_idx += 1
                    continue

                # Would adding this block exceed the limit?
                if current_tokens + block_tokens > cc.max_tokens and current_text_parts:
                    # Flush current chunk
                    text = "\n\n".join(current_text_parts)
                    chunks.append(Chunk(
                        id=Chunk.make_id(section.doc_path, chunk_idx),
                        text=text,
                        breadcrumb=section.breadcrumb,
                        title=section.title,
                        doc_path=section.doc_path,
                        url=url,
                        chunk_index=chunk_idx,
                        has_code="```" in text or "~~~" in text,
                        n_tokens=_approx_tokens(text),
                        file_hash=fhash,
                        source_commit=commit_sha,
                    ))
                    chunk_idx += 1

                    # Start new chunk with overlap (last paragraph of previous)
                    if last_paragraph and not FENCE_RE.match(last_paragraph.strip()):
                        current_text_parts = [last_paragraph]
                        current_tokens = _approx_tokens(last_paragraph)
                    else:
                        current_text_parts = []
                        current_tokens = 0

                current_text_parts.append(block)
                current_tokens += block_tokens

                # Track last paragraph for overlap
                if not FENCE_RE.match(block.strip()):
                    last_paragraph = block

            # Flush remaining
            if current_text_parts:
                text = "\n\n".join(current_text_parts)
                chunks.append(Chunk(
                    id=Chunk.make_id(section.doc_path, chunk_idx),
                    text=text,
                    breadcrumb=section.breadcrumb,
                    title=section.title,
                    doc_path=section.doc_path,
                    url=url,
                    chunk_index=chunk_idx,
                    has_code="```" in text or "~~~" in text,
                    n_tokens=_approx_tokens(text),
                    file_hash=fhash,
                    source_commit=commit_sha,
                ))
                chunk_idx += 1

    # Validate invariant: no chunk has unbalanced code fences
    for chunk in chunks:
        if not _has_balanced_fences(chunk.text):
            logger.warning(
                "Chunk %s has unbalanced code fences (doc_path=%s, index=%d)",
                chunk.id, chunk.doc_path, chunk.chunk_index,
            )

    logger.debug("Created %d chunks from %s", len(chunks), sections[0].doc_path if sections else "?")
    return chunks
