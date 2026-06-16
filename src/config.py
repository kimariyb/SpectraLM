"""Configuration loading helpers for YAML-driven workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    path
        Path to a YAML configuration file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration values.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    TypeError
        If the YAML root is not a mapping.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping config from {config_path}, got {type(payload).__name__}")
    return payload

