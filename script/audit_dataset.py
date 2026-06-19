"""Audit raw NMR CSV files before SpectraLM preprocessing.

The audit is intentionally read-only.  It streams large CSV files in chunks and
emits a compact JSON report with coverage, pairing, and missingness statistics.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

# Allow running from project root without PYTHONPATH.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.molecules import canonicalize_smiles
from src.io import write_json


CRITICAL_COLUMNS = [
    "Filename",
    "SMILES",
    "NMR_type",
    "NMR_frequency",
    "NMR_solvent",
    "NMR_processed",
    "Atom_number",
    "Atom_number_diff_env",
    "Atom_number_abstract",
]


def _counter_top(counter: Counter[str], n: int = 20) -> list[dict[str, Any]]:
    """Return a JSON-friendly top-n view of a Counter."""
    return [{"value": key, "count": count} for key, count in counter.most_common(n)]


def _nonnull_count(series: pd.Series) -> int:
    """Count values that are neither null nor empty strings."""
    return int(series.notna().sum() - (series.astype(str).str.strip() == "").sum())


def audit_csv(
    path: str | Path,
    *,
    chunksize: int = 100_000,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Audit a raw NMR CSV file.

    Parameters
    ----------
    path
        Input CSV path.
    chunksize
        Number of rows per pandas chunk.
    max_rows
        Optional cap for quick pilot audits.

    Returns
    -------
    dict[str, Any]
        JSON-serializable audit report.
    """
    csv_path = Path(path)
    total_rows = 0
    nmr_type_counts: Counter[str] = Counter()
    solvent_counts: Counter[str] = Counter()
    frequency_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    nonnull_counts: Counter[str] = Counter()
    raw_smiles: set[str] = set()
    canonical_by_type: dict[str, set[str]] = defaultdict(set)
    invalid_smiles = 0

    reader = pd.read_csv(csv_path, chunksize=chunksize)
    for chunk in reader:
        if max_rows is not None:
            remaining = max_rows - total_rows
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)

        total_rows += len(chunk)

        for column in CRITICAL_COLUMNS:
            if column not in chunk.columns:
                missing_counts[column] += len(chunk)
                continue
            nonnull = _nonnull_count(chunk[column])
            nonnull_counts[column] += nonnull
            missing_counts[column] += len(chunk) - nonnull

        nmr_types = chunk.get("NMR_type", pd.Series(dtype=str)).fillna("missing").astype(str)
        nmr_type_counts.update(nmr_types)

        solvents = chunk.get("NMR_solvent", pd.Series(dtype=str)).fillna("missing").astype(str)
        solvent_counts.update(solvents)

        frequencies = chunk.get("NMR_frequency", pd.Series(dtype=str)).fillna("missing").astype(str)
        frequency_counts.update(frequencies)

        smiles_series = chunk.get("SMILES", pd.Series(dtype=str)).dropna().astype(str)
        raw_smiles.update(smiles_series)

        type_series = chunk.get("NMR_type", pd.Series(index=chunk.index, dtype=str)).astype(str)
        for smiles, nmr_type in zip(smiles_series, type_series.loc[smiles_series.index]):
            canonical = canonicalize_smiles(smiles)
            if canonical is None:
                invalid_smiles += 1
                continue
            canonical_by_type[str(nmr_type)].add(canonical)

    h_set = canonical_by_type.get("1H NMR", set())
    c_set = canonical_by_type.get("13C NMR", set())
    paired = h_set & c_set

    return {
        "input": str(csv_path),
        "file_size_bytes": csv_path.stat().st_size,
        "rows_audited": total_rows,
        "columns": CRITICAL_COLUMNS,
        "nmr_type_counts": dict(nmr_type_counts),
        "unique_raw_smiles": len(raw_smiles),
        "invalid_smiles_rows": invalid_smiles,
        "unique_canonical_by_type": {
            key: len(value) for key, value in sorted(canonical_by_type.items())
        },
        "paired_1h_13c_unique_canonical": len(paired),
        "only_1h_unique_canonical": len(h_set - c_set),
        "only_13c_unique_canonical": len(c_set - h_set),
        "missing_counts": dict(missing_counts),
        "nonnull_counts": dict(nonnull_counts),
        "top_solvents": _counter_top(solvent_counts),
        "top_frequencies": _counter_top(frequency_counts),
    }


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="Raw NMR CSV file to audit.")
    parser.add_argument(
        "--output",
        default="dataset/audit/raw_nmr_audit.json",
        help="Output JSON path.",
    )
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap for quick pilot audits.",
    )
    args = parser.parse_args()

    report = audit_csv(args.csv, chunksize=args.chunksize, max_rows=args.max_rows)
    write_json(args.output, report)
    print(f"Wrote {args.output}")
    print(
        "paired_1h_13c_unique_canonical="
        f"{report['paired_1h_13c_unique_canonical']:,}"
    )


if __name__ == "__main__":
    main()
