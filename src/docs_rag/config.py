"""
Configuration loading: merges environment variables (.env) with YAML config files.

Design rationale:
- Secrets (API keys, URLs) live in .env → loaded by pydantic-settings.
- Tunable behavior (chunk sizes, retrieval params) lives in YAML → each ablation
  is just a different YAML file, no code changes needed.
- The YAML path itself is an env var (CONFIG_PATH), so the same Docker image can
  run different configs by changing one variable.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Environment-based settings (secrets + deployment specifics)
# ---------------------------------------------------------------------------

class EnvSettings(BaseSettings):
    """Loaded from .env or actual environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "docs"

    # LLM (Groq)
    groq_api_key: str = ""
    llm_model: str = "llama-3.1-8b-instant"

    # Judge LLM (evaluation only)
    judge_provider: str = "groq"
    judge_model: str = "llama-3.3-70b-versatile"
    judge_api_key: str = ""

    # Reranker
    reranker_provider: str = "local"  # local | hosted | none
    reranker_api_key: str = ""

    # Paths
    config_path: str = "configs/base.yaml"
    model_cache_dir: str = "./models"

    # Misc
    github_token: str = ""
    log_level: str = "INFO"


# ---------------------------------------------------------------------------
# YAML-based config (tunable behavior)
# ---------------------------------------------------------------------------

class CorpusConfig:
    """Settings about which docs to index."""

    def __init__(self, data: dict[str, Any]):
        self.name: str = data.get("name", "LangGraph")
        self.repo_url: str = data.get("repo_url", "")
        self.docs_subdir: str = data.get("docs_subdir", "docs")
        self.include: list[str] = data.get("include", ["**/*.md", "**/*.mdx"])
        self.exclude: list[str] = data.get("exclude", [])
        self.base_url: str = data.get("base_url", "")


class ChunkingConfig:
    """Parameters controlling how sections become chunks."""

    def __init__(self, data: dict[str, Any]):
        self.min_tokens: int = data.get("min_tokens", 80)
        self.max_tokens: int = data.get("max_tokens", 450)
        self.hard_max_tokens: int = data.get("hard_max_tokens", 1000)


class RetrievalConfig:
    """Which models to use for encoding and how many candidates to fetch."""

    def __init__(self, data: dict[str, Any]):
        self.dense_model: str = data.get("dense_model", "BAAI/bge-small-en-v1.5")
        self.sparse_model: str = data.get("sparse_model", "Qdrant/bm25")
        self.use_sparse: bool = data.get("use_sparse", True)
        self.prefetch_limit: int = data.get("prefetch_limit", 20)
        self.fusion: str = data.get("fusion", "rrf")


class RerankConfig:
    """Reranker selection and refusal threshold."""

    def __init__(self, data: dict[str, Any]):
        self.provider: str = data.get("provider", "local")
        self.model: str = data.get("model", "Xenova/ms-marco-MiniLM-L-6-v2")
        self.top_n: int = data.get("top_n", 5)
        self.refusal_threshold: float | None = data.get("refusal_threshold")


class GenerationConfig:
    """LLM generation parameters."""

    def __init__(self, data: dict[str, Any]):
        self.temperature: float = data.get("temperature", 0.1)
        self.max_tokens: int = data.get("max_tokens", 600)
        self.timeout_s: int = data.get("timeout_s", 30)
        self.max_retries: int = data.get("max_retries", 3)


class YAMLConfig:
    """Full YAML configuration parsed into typed sub-configs."""

    def __init__(self, data: dict[str, Any]):
        self.corpus = CorpusConfig(data.get("corpus", {}))
        self.chunking = ChunkingConfig(data.get("chunking", {}))
        self.retrieval = RetrievalConfig(data.get("retrieval", {}))
        self.rerank = RerankConfig(data.get("rerank", {}))
        self.generation = GenerationConfig(data.get("generation", {}))


def load_yaml_config(path: str | Path) -> YAMLConfig:
    """Load and parse a YAML config file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    return YAMLConfig(data or {})


# ---------------------------------------------------------------------------
# Unified settings accessor — loads once, cached
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_env_settings() -> EnvSettings:
    """Return the singleton environment settings."""
    return EnvSettings()


def get_config(config_path: str | Path | None = None) -> YAMLConfig:
    """Load YAML config, defaulting to the path in the environment."""
    if config_path is None:
        config_path = get_env_settings().config_path
    return load_yaml_config(config_path)


def get_config_name(config_path: str | Path | None = None) -> str:
    """Extract the config name from its filename (e.g. 'c_hybrid_rerank')."""
    if config_path is None:
        config_path = get_env_settings().config_path
    return Path(config_path).stem
