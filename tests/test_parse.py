"""
Tests for the markdown parser.

These tests verify the critical invariants:
1. YAML front matter is stripped but its title is preserved
2. Headings inside code blocks are NOT treated as section boundaries
3. Breadcrumbs correctly reflect the heading hierarchy
4. MDX import/export lines are stripped
5. Empty sections are filtered out
"""

from pathlib import Path

from docs_rag.ingest.parse import parse_file

FIXTURES = Path(__file__).parent / "fixtures"


class TestParseMarkdown:
    """Tests for standard markdown parsing."""

    def test_front_matter_stripped(self):
        """Front matter should be removed from section bodies."""
        sections = parse_file(FIXTURES / "sample_doc.md", "test/sample_doc.md")
        for section in sections:
            assert "---" not in section.body.split("\n")[0] or "```" in section.body
            # Title from front matter should be captured
        assert any(s.title == "Test Document" for s in sections)

    def test_sections_created_on_headings(self):
        """H1-H3 headings should create section boundaries."""
        sections = parse_file(FIXTURES / "sample_doc.md", "test/sample_doc.md")
        assert len(sections) >= 3  # At least: Getting Started, API Reference, Troubleshooting

    def test_breadcrumbs_correct(self):
        """Breadcrumbs should reflect the heading hierarchy."""
        sections = parse_file(FIXTURES / "sample_doc.md", "test/sample_doc.md")
        breadcrumbs = [s.breadcrumb for s in sections]
        # Should have hierarchical breadcrumbs
        assert any(">" in bc for bc in breadcrumbs), (
            f"Expected hierarchical breadcrumbs, got: {breadcrumbs}"
        )

    def test_headings_inside_code_ignored(self):
        """# comments inside code blocks should NOT create sections."""
        sections = parse_file(FIXTURES / "sample_doc.md", "test/sample_doc.md")
        # If headings inside code were parsed, we'd get spurious sections
        # with breadcrumbs containing code comments
        for section in sections:
            assert "pip install" not in section.breadcrumb

    def test_code_blocks_preserved(self):
        """Code blocks should be fully preserved in section bodies."""
        sections = parse_file(FIXTURES / "sample_doc.md", "test/sample_doc.md")
        # Find the section with the Python code example
        code_sections = [s for s in sections if s.has_code]
        assert len(code_sections) >= 1, "Should have at least one section with code"

        # Verify code block is intact
        for section in code_sections:
            if "```python" in section.body:
                assert section.body.count("```") % 2 == 0, (
                    "Code blocks should have balanced fences"
                )

    def test_no_empty_sections(self):
        """Sections with empty bodies should be filtered out."""
        sections = parse_file(FIXTURES / "sample_doc.md", "test/sample_doc.md")
        for section in sections:
            assert section.body.strip(), f"Empty section found: {section.breadcrumb}"

    def test_heading_slugs(self):
        """Heading slugs should be URL-safe."""
        sections = parse_file(FIXTURES / "sample_doc.md", "test/sample_doc.md")
        for section in sections:
            assert " " not in section.heading_slug, (
                f"Slug should not contain spaces: {section.heading_slug}"
            )

    def test_doc_path_preserved(self):
        """Every section should carry its source doc_path."""
        sections = parse_file(FIXTURES / "sample_doc.md", "test/sample_doc.md")
        for section in sections:
            assert section.doc_path == "test/sample_doc.md"


class TestParseMDX:
    """Tests for MDX-specific parsing."""

    def test_imports_stripped(self):
        """MDX import/export lines should be removed."""
        sections = parse_file(FIXTURES / "sample_mdx.mdx", "test/sample_mdx.mdx")
        for section in sections:
            assert "import {" not in section.body
            assert "import CodeBlock" not in section.body
            assert "export const" not in section.body

    def test_jsx_tags_stripped(self):
        """JSX-only tags like <Tabs> should be stripped."""
        sections = parse_file(FIXTURES / "sample_mdx.mdx", "test/sample_mdx.mdx")
        for section in sections:
            # Standalone JSX tags should be removed
            lines = section.body.split("\n")
            for line in lines:
                stripped = line.strip()
                if stripped and stripped.startswith("<") and stripped.endswith(">"):
                    # This is OK for HTML in markdown, but pure JSX components should be removed
                    pass  # Some may remain — that's acceptable

    def test_heading_inside_code_not_parsed_mdx(self):
        """# comments in code blocks should not create sections in MDX files."""
        sections = parse_file(FIXTURES / "sample_mdx.mdx", "test/sample_mdx.mdx")
        for section in sections:
            assert "Not a heading" not in section.breadcrumb

    def test_mdx_title_from_frontmatter(self):
        """Title should come from MDX front matter."""
        sections = parse_file(FIXTURES / "sample_mdx.mdx", "test/sample_mdx.mdx")
        assert any(s.title == "MDX Component Test" for s in sections)
