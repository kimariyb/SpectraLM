"""Configuration loading helpers for command-line workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path | None) -> dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    path
        Path to a YAML configuration file. If ``None``, an empty dictionary is
        returned.

    Returns
    -------
    dict[str, Any]
        Parsed configuration values.
    """
    if path is None:
        return {}
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping config from {config_path}, got {type(payload).__name__}")
    return payload


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    """Add a shared ``--config`` option to an argument parser.

    Parameters
    ----------
    parser
        Parser that should receive the shared option.
    """
    parser.add_argument("--config", default=None, help="Path to a YAML configuration file.")

