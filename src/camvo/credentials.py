"""Explicit, non-logging API credential loader."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

KNOWN_API_KEY_NAMES = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "CUSTOM_LLM_BASE_URL",
    "CUSTOM_LLM_API_KEY",
    "INTERNAL_LLM_API_KEY",
    "DIFY_WORKFLOW_API_KEY",
    "ADAMS_PLATFORM_USER",
    "ADAMS_USER_TOKEN",
)


def load_api_keys(
    path: str | Path = "config/api_keys.env",
    *,
    override: bool = False,
) -> dict[str, str]:
    """Load the dedicated local key file and return only configured values.

    This function never prints values and checkpoints never include them.
    """

    key_file = Path(path)
    if key_file.exists():
        load_dotenv(key_file, override=override)
    return {name: os.environ[name] for name in KNOWN_API_KEY_NAMES if os.environ.get(name)}
