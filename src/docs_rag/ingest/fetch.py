"""
Fetch documentation source files from a Git repository.

This module handles:
1. Shallow-cloning the docs repo (or pulling if already cloned)
2. Recording the commit SHA for provenance tracking
3. Listing files matching include/exclude globs from config

Why shallow clone: we only need the latest version of each file,
and a full clone of a large repo wastes time and disk space.
"""

from __future__ import annotations

import logging
import subprocess
from fnmatch import fnmatch
from pathlib import Path

from docs_rag.config import YAMLConfig

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/raw")


def fetch_docs(config: YAMLConfig, data_dir: Path | None = None) -> tuple[Path, str]:
    """Clone or pull the docs repo and return (repo_path, commit_sha).

    Args:
        config: YAML config with corpus settings.
        data_dir: Override the default data directory (for testing).

    Returns:
        Tuple of (path to the cloned repo, current commit SHA).
    """
    data_dir = data_dir or DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    corpus = config.corpus
    # Derive a folder name from the repo URL
    repo_name = corpus.repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    repo_path = data_dir / repo_name

    if repo_path.exists() and (repo_path / ".git").exists():
        # Pull latest changes
        logger.info("Pulling latest changes in %s", repo_path)
        subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )
    else:
        # Shallow clone — only the latest commit, saves bandwidth and time
        logger.info("Cloning %s into %s", corpus.repo_url, repo_path)
        subprocess.run(
            ["git", "clone", "--depth", "1", corpus.repo_url, str(repo_path)],
            check=True,
            capture_output=True,
        )

    # Record the commit SHA — stored with every chunk for traceability
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    commit_sha = result.stdout.strip()
    logger.info("Docs at commit: %s", commit_sha)

    return repo_path, commit_sha


def list_doc_files(repo_path: Path, config: YAMLConfig) -> list[Path]:
    """List all doc files matching the include/exclude globs.

    Args:
        repo_path: Root of the cloned docs repo.
        config: YAML config with corpus.include and corpus.exclude.

    Returns:
        Sorted list of absolute Paths to matching files.
    """
    corpus = config.corpus
    docs_root = repo_path / corpus.docs_subdir

    if not docs_root.exists():
        logger.warning("Docs subdir not found: %s — searching from repo root", docs_root)
        docs_root = repo_path

    matched: list[Path] = []

    for pattern in corpus.include:
        for file_path in docs_root.rglob("*"):
            if not file_path.is_file():
                continue
            rel = file_path.relative_to(docs_root).as_posix()
            if not fnmatch(rel, pattern):
                continue
            # Check exclusions
            excluded = any(fnmatch(rel, exc) for exc in corpus.exclude)
            if excluded:
                continue
            if file_path not in matched:
                matched.append(file_path)

    matched.sort(key=lambda p: p.as_posix())
    logger.info("Found %d doc files", len(matched))
    return matched
