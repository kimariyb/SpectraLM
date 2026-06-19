"""Build a paired 1H/13C NMR JSONL dataset from a raw CSV file.

This is the scalable preprocessing path for million-sample experiments.  It
streams the raw CSV into a small SQLite candidate index, selects one paired
1H/13C record per canonical molecule, and writes:

- ``samples.jsonl``: one normalized sample per line
- ``manifest.csv``: QC metadata and scaffold-disjoint split assignment
- ``manifest_summary.json`` and split id lists
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

# Allow running from project root without PYTHONPATH.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from script.build_manifest import (
    MANIFEST_FIELDS,
    assign_scaffold_splits,
    manifest_summary,
    sample_manifest_row,
    write_split_files,
)
from src.data.molecules import canonicalize_smiles, smiles_to_selfies
from src.data.utils import process_13c_peaks, process_1h_peaks, safe_literal_eval
from src.io import write_json, write_rows_csv


SOURCE_COLUMNS = [
    "Filename",
    "SMILES",
    "Page_in_file_mol",
    "Page_in_file_para",
    "Location_in_page_mol",
    "Location_in_page_para",
    "NMR_type",
    "NMR_frequency",
    "NMR_solvent",
    "NMR_processed",
    "Atom_number",
    "Atom_number_diff_env",
    "Atom_number_abstract",
]


def sample_id(canonical_smiles: str) -> str:
    """Return a stable sample id for one canonical molecule."""
    import hashlib

    digest = hashlib.sha256(canonical_smiles.encode("utf-8")).hexdigest()[:16]
    return f"nmr-{digest}"


def nucleus_label(nmr_type: str) -> str | None:
    """Map raw NMR type strings to internal nucleus labels."""
    if nmr_type == "1H NMR":
        return "1h"
    if nmr_type == "13C NMR":
        return "13c"
    return None


def candidate_score(row: pd.Series) -> int:
    """Score one raw candidate for deterministic best-record selection."""
    solvent = str(row.get("NMR_solvent", "")).strip()
    processed = str(row.get("NMR_processed", "")).strip()
    score = 0
    if processed:
        score += 100
    if solvent and solvent not in {"not_known", "nan", "None"}:
        score += 10
    if solvent == "CDCl3":
        score += 5
    return score


def connect_db(path: str | Path) -> sqlite3.Connection:
    """Open the candidate SQLite database."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS candidates (
            canonical_smiles TEXT NOT NULL,
            nucleus TEXT NOT NULL,
            score INTEGER NOT NULL,
            row_index INTEGER NOT NULL,
            raw_smiles TEXT NOT NULL,
            filename TEXT,
            page_mol TEXT,
            page_para TEXT,
            location_mol TEXT,
            location_para TEXT,
            frequency TEXT,
            solvent TEXT,
            processed TEXT,
            atom_number TEXT,
            atom_number_diff_env TEXT,
            atom_number_abstract TEXT
        )
        """
    )
    return conn


def reset_db(conn: sqlite3.Connection) -> None:
    """Clear existing candidate rows."""
    conn.execute("DELETE FROM candidates")
    conn.commit()


def index_raw_csv(
    csv_path: str | Path,
    db_path: str | Path,
    *,
    chunksize: int = 100_000,
    max_rows: int | None = None,
    reset: bool = True,
) -> dict[str, Any]:
    """Stream raw CSV rows into a SQLite candidate index."""
    conn = connect_db(db_path)
    if reset:
        reset_db(conn)

    rows_seen = 0
    candidates = 0
    invalid_smiles = 0
    unsupported_nmr = 0

    insert_sql = """
        INSERT INTO candidates VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
    """

    reader = pd.read_csv(csv_path, chunksize=chunksize, usecols=SOURCE_COLUMNS)
    for chunk in reader:
        if max_rows is not None:
            remaining = max_rows - rows_seen
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)

        batch = []
        for row_index, row in chunk.iterrows():
            rows_seen += 1
            nucleus = nucleus_label(str(row["NMR_type"]))
            if nucleus is None:
                unsupported_nmr += 1
                continue

            canonical = canonicalize_smiles(row["SMILES"])
            if canonical is None:
                invalid_smiles += 1
                continue

            batch.append(
                (
                    canonical,
                    nucleus,
                    candidate_score(row),
                    int(row_index),
                    str(row["SMILES"]),
                    str(row.get("Filename", "")),
                    str(row.get("Page_in_file_mol", "")),
                    str(row.get("Page_in_file_para", "")),
                    str(row.get("Location_in_page_mol", "")),
                    str(row.get("Location_in_page_para", "")),
                    str(row.get("NMR_frequency", "")),
                    str(row.get("NMR_solvent", "")),
                    str(row.get("NMR_processed", "")),
                    str(row.get("Atom_number", "")),
                    str(row.get("Atom_number_diff_env", "")),
                    str(row.get("Atom_number_abstract", "")),
                )
            )

        if batch:
            conn.executemany(insert_sql, batch)
            conn.commit()
            candidates += len(batch)

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_candidates_pair "
        "ON candidates(canonical_smiles, nucleus, score DESC, row_index ASC)"
    )
    conn.commit()
    conn.close()

    return {
        "rows_seen": rows_seen,
        "candidate_rows": candidates,
        "invalid_smiles": invalid_smiles,
        "unsupported_nmr_rows": unsupported_nmr,
    }


def candidate_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a sqlite row into a plain dictionary."""
    return {key: row[key] for key in row.keys()}


def select_best_pair(
    h_candidates: list[dict[str, Any]],
    c_candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Choose the best 1H/13C pair from limited candidate lists."""
    best: tuple[int, dict[str, Any], dict[str, Any]] | None = None
    for h_row in h_candidates:
        for c_row in c_candidates:
            h_solvent = str(h_row.get("solvent", ""))
            c_solvent = str(c_row.get("solvent", ""))
            score = int(h_row["score"]) + int(c_row["score"])
            if h_solvent == c_solvent and h_solvent not in {"", "not_known"}:
                score += 1000
            if h_solvent == "CDCl3" and c_solvent == "CDCl3":
                score += 100
            score -= abs(int(h_row["row_index"]) - int(c_row["row_index"])) // 100000
            if best is None or score > best[0]:
                best = (score, h_row, c_row)
    if best is None:
        raise ValueError("Cannot select pair from empty candidates.")
    return best[1], best[2]


def raw_pair_to_sample(
    canonical: str,
    h_row: dict[str, Any],
    c_row: dict[str, Any],
) -> dict[str, Any] | None:
    """Build one normalized sample from paired raw candidate rows."""
    selfies = smiles_to_selfies(canonical)
    if selfies is None:
        return None

    try:
        h_raw = safe_literal_eval(h_row["processed"])
        c_raw = safe_literal_eval(c_row["processed"])
    except Exception:
        return None

    sample = {
        "id": sample_id(canonical),
        "smiles": canonical,
        "canonical_smiles": canonical,
        "selfies": selfies,
        "meta": {
            "source": "experimental",
            "source_13c": {
                "filename": c_row.get("filename"),
                "page_mol": c_row.get("page_mol"),
                "page_para": c_row.get("page_para"),
                "location_mol": c_row.get("location_mol"),
                "location_para": c_row.get("location_para"),
            },
            "source_1h": {
                "filename": h_row.get("filename"),
                "page_mol": h_row.get("page_mol"),
                "page_para": h_row.get("page_para"),
                "location_mol": h_row.get("location_mol"),
                "location_para": h_row.get("location_para"),
            },
        },
        "13C_NMR": {
            "frequency": c_row.get("frequency"),
            "solvent": c_row.get("solvent"),
            "peaks": process_13c_peaks(c_raw),
        },
        "1H_NMR": {
            "frequency": h_row.get("frequency"),
            "solvent": h_row.get("solvent"),
            "peaks": process_1h_peaks(h_raw),
        },
        "spectrum": {"1H_image": None, "13C_image": None},
    }
    return sample


def paired_canonicals(conn: sqlite3.Connection) -> Iterable[str]:
    """Yield canonical SMILES with both 1H and 13C candidates."""
    query = """
        SELECT canonical_smiles
        FROM candidates
        GROUP BY canonical_smiles
        HAVING SUM(CASE WHEN nucleus = '1h' THEN 1 ELSE 0 END) > 0
           AND SUM(CASE WHEN nucleus = '13c' THEN 1 ELSE 0 END) > 0
        ORDER BY canonical_smiles
    """
    for (canonical,) in conn.execute(query):
        yield str(canonical)


def fetch_candidates(
    conn: sqlite3.Connection,
    canonical: str,
    nucleus: str,
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    """Fetch top candidates for a canonical molecule and nucleus."""
    rows = conn.execute(
        """
        SELECT * FROM candidates
        WHERE canonical_smiles = ? AND nucleus = ?
        ORDER BY score DESC, row_index ASC
        LIMIT ?
        """,
        (canonical, nucleus, top_k),
    ).fetchall()
    return [candidate_dict(row) for row in rows]


def build_jsonl_from_index(
    db_path: str | Path,
    out_dir: str | Path,
    *,
    top_k: int = 3,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 3407,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Build JSONL dataset and manifest from an indexed candidate DB."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_path / "samples.jsonl"

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = []
    written = 0
    skipped = 0
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for canonical in paired_canonicals(conn):
            if max_samples is not None and written >= max_samples:
                break
            h_candidates = fetch_candidates(conn, canonical, "1h", top_k=top_k)
            c_candidates = fetch_candidates(conn, canonical, "13c", top_k=top_k)
            if not h_candidates or not c_candidates:
                skipped += 1
                continue
            h_row, c_row = select_best_pair(h_candidates, c_candidates)
            sample = raw_pair_to_sample(canonical, h_row, c_row)
            if sample is None:
                skipped += 1
                continue
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            rows.append(sample_manifest_row(sample))
            written += 1

    conn.close()

    rows = assign_scaffold_splits(
        rows,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    summary = manifest_summary(rows)
    summary["jsonl_samples_written"] = written
    summary["jsonl_samples_skipped"] = skipped
    summary["top_k_candidates"] = top_k

    write_rows_csv(out_path / "manifest.csv", rows, MANIFEST_FIELDS)
    write_json(out_path / "manifest_summary.json", summary)
    write_split_files(rows, out_path)

    return summary


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="Raw NMR CSV file.")
    parser.add_argument("--out-dir", default="dataset/paired_jsonl")
    parser.add_argument("--db", default="dataset/paired_jsonl/candidates.sqlite")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--reuse-index",
        action="store_true",
        help="Skip CSV indexing and reuse an existing SQLite DB.",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.reuse_index:
        index_report = index_raw_csv(
            args.csv,
            db_path,
            chunksize=args.chunksize,
            max_rows=args.max_rows,
            reset=True,
        )
        write_json(Path(args.out_dir) / "index_report.json", index_report)
        print(json.dumps(index_report, ensure_ascii=False, indent=2))

    summary = build_jsonl_from_index(
        db_path,
        args.out_dir,
        top_k=args.top_k,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote paired JSONL dataset to {args.out_dir}")


if __name__ == "__main__":
    main()
