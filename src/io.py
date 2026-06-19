"""File IO helpers used across SpectraLM workflows."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_json(path: str | Path, payload: Any) -> None:
    """Write a JSON file with UTF-8 encoding.

    Parameters
    ----------
    path
        Output JSON path.
    payload
        JSON-serializable object.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_rows_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write row dictionaries to a CSV file.

    Parameters
    ----------
    path
        Output CSV path.
    rows
        Row dictionaries to write.
    fieldnames
        Ordered CSV header names.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
