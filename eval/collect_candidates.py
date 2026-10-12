"""
Collect candidate evaluation questions from GitHub issues and discussions.

Searches for closed issues/discussions with accepted answers in the
target framework's repo. These represent real user questions with
verified answers — the most credible source for evaluation data.

Usage:
    uv run python -m eval.collect_candidates --repo langchain-ai/langgraph --limit 50

Output: prints candidate questions to stdout in JSONL format.
The builder must review and label each one (set relevant doc paths,
mark as answerable/unanswerable, assign to dev/test split).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import httpx

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"


def search_issues(
    repo: str,
    token: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Search for closed issues with answers in the given repo.

    Looks for issues labeled as questions, bugs with resolution, or
    discussions with accepted answers.
    """
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Search for closed issues that likely contain Q&A
    queries = [
        f"repo:{repo} is:issue is:closed label:question",
        f"repo:{repo} is:issue is:closed label:bug comments:>1",
        f"repo:{repo} is:issue is:closed comments:>2",
    ]

    candidates = []
    seen_urls = set()

    for query in queries:
        if len(candidates) >= limit:
            break

        params = {
            "q": query,
            "sort": "comments",
            "order": "desc",
            "per_page": min(30, limit - len(candidates)),
        }

        try:
            resp = httpx.get(
                f"{GITHUB_API}/search/issues",
                params=params,
                headers=headers,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("items", []):
                url = item.get("html_url", "")
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                title = item.get("title", "")
                body = item.get("body", "")

                # Skip very short issues
                if len(title) < 10:
                    continue

                candidates.append({
                    "title": title,
                    "url": url,
                    "body_preview": (body or "")[:300],
                    "comments": item.get("comments", 0),
                    "labels": [l["name"] for l in item.get("labels", [])],
                })

            # Respect rate limits
            time.sleep(2)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                logger.warning("GitHub rate limit hit. Set GITHUB_TOKEN for higher limits.")
                break
            logger.error("GitHub API error: %s", e)
        except Exception as e:
            logger.error("Failed to search: %s", e)

    return candidates[:limit]


def format_candidate(candidate: dict, index: int) -> dict:
    """Format a candidate into a JSONL-ready evaluation question template."""
    return {
        "id": f"q{index:03d}",
        "question": candidate["title"],
        "answerable": True,  # Must be verified by hand
        "relevant": [],  # Must be labeled: ["docs/path.md#section"]
        "reference_answer": None,  # Fill in the correct answer
        "source": "github_issue",
        "source_url": candidate["url"],
        "split": "dev" if index % 3 != 0 else "test",  # ~70/30 split
        "_review_note": "REVIEW: verify answerable, add relevant paths, write reference answer",
    }


def main():
    parser = argparse.ArgumentParser(description="Collect candidate eval questions from GitHub")
    parser.add_argument("--repo", default="langchain-ai/langgraph", help="GitHub repo (owner/name)")
    parser.add_argument("--limit", type=int, default=50, help="Max candidates to collect")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        logger.warning("No GITHUB_TOKEN set — rate limits will be strict (60 req/hr)")

    logger.info("Searching %s for candidate questions...", args.repo)
    candidates = search_issues(args.repo, token, args.limit)

    logger.info("Found %d candidates", len(candidates))

    # Output as JSONL
    for i, c in enumerate(candidates, 1):
        formatted = format_candidate(c, i)
        print(json.dumps(formatted))

    print(f"\n# Found {len(candidates)} candidates.", file=sys.stderr)
    print("# Review each one: verify answerable, add relevant doc paths,", file=sys.stderr)
    print("# write reference answers, and assign to dev/test split.", file=sys.stderr)
    print("# Save reviewed questions to eval/dataset/questions.jsonl", file=sys.stderr)


if __name__ == "__main__":
    main()
