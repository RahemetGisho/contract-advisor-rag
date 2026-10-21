"""Load the project's central YAML config. Every module reads settings
through this function instead of hardcoding paths/models/thresholds."""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import ValidationError

from src.utils.settings import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


ENV_PATH = PROJECT_ROOT / ".env"


def load_env() -> None:
    """Load PROJECT_ROOT/.env into the process environment.

    Idempotent and non-overriding: a variable already set in the real
    environment (Docker, CI, your shell) always wins over the file, so the
    file only fills gaps. An explicit path is used rather than searching
    upward from the caller, which could pick up an unrelated .env.
    """
    load_dotenv(ENV_PATH, override=False)


# Environment variables that override a single config value. Lets the Docker
# image, CI, and offline demos switch providers without editing YAML.
ENV_OVERRIDES = {
    "RAG_EMBEDDING_PROVIDER": ("embeddings", "provider"),
    "RAG_LLM_PROVIDER": ("llm", "provider"),
    "RAG_VECTOR_STORE_DIR": ("vector_store", "persist_dir"),
    "RAG_API_REQUIRE_KEY": ("api", "require_api_key"),
}


class ConfigError(ValueError):
    """The config file (after environment overrides) does not match the schema."""


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    for var, (section, key) in ENV_OVERRIDES.items():
        value = os.environ.get(var)
        if value:
            config[section][key] = value
    return config


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")
    with open(path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config at {path} did not parse into a mapping")
    load_env()  # before overrides, so RAG_* values placed in .env take effect too
    config = _apply_env_overrides(config)
    try:
        return Settings.model_validate(config).model_dump()
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
        )
        raise ConfigError(f"Invalid configuration in {path}: {problems}") from exc


def resolve_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path
