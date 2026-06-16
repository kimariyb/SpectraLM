"""File IO helpers used across SpectraLM workflows."""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Any


def load_pickle_list(path: str | Path) -> list[dict[str, Any]]:
    """Load a pickle file and require it to contain a list of dictionaries.

    Parameters
    ----------
    path
        Pickle file path.

    Returns
    -------
    list[dict[str, Any]]
        Loaded samples.

    Raises
    ------
    TypeError
        If the pickle payload is not a list.
    """
    pickle_path = Path(path)
    with pickle_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, list):
        raise TypeError(f"Expected a list from {pickle_path}, got {type(payload).__name__}")
    return payload


def write_pickle(path: str | Path, payload: Any) -> None:
    """Write a Python object to a pickle file.

    Parameters
    ----------
    path
        Output pickle path.
    payload
        Object to serialize.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle)


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

