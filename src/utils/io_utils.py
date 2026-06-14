import csv
import json
import pickle
from pathlib import Path
from typing import Any


def load_pickle_list(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, list):
        raise TypeError(f"Expected a list from {path}, got {type(payload).__name__}")
    return payload


def write_pickle(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_rows_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
