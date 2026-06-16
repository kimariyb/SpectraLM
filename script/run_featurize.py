"""Structure-only ECFP fingerprint features for SpectraLM sampling.

Uses ``skfp.fingerprints.ECFPFingerprint`` for batch Morgan/ECFP fingerprint
computation with scikit-learn compatible API.
"""

from __future__ import annotations

import os
import sys

# Allow running from project root without PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from typing import Any
from tqdm import tqdm
from skfp.fingerprints import ECFPFingerprint
from src.data.molecules import functional_group_labels, murcko_scaffold
from src.io import load_pickle_list, write_rows_csv
import pandas as pd


FEATURE_INDEX_FIELDS = [
    "row_index",
    "id",
    "canonical_smiles",
    "murcko_scaffold",
    "functional_groups",
    "feature_status",
]


def feature_csv_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert feature metadata rows to CSV-ready dictionaries.

    Parameters
    ----------
    rows
        Feature metadata rows.

    Returns
    -------
    list[dict[str, Any]]
        CSV-ready metadata rows.
    """
    output = []
    for row in rows:
        item = dict(row)
        item["functional_groups"] = ";".join(row.get("functional_groups", []))
        output.append({field: item.get(field, "") for field in FEATURE_INDEX_FIELDS})
    return output


def save_feature_outputs(
    matrix: np.ndarray,
    rows: list[dict[str, Any]],
    fingerprint_path: str | Path,
    index_path: str | Path,
) -> None:
    """Write ECFP fingerprints and feature metadata.

    Parameters
    ----------
    matrix
        ECFP fingerprint matrix.
    rows
        Feature metadata rows.
    fingerprint_path
        Output ``.npz`` path.
    index_path
        Output CSV index path.
    """
    output_path = Path(fingerprint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, fingerprints=matrix)
    write_rows_csv(index_path, feature_csv_rows(rows), FEATURE_INDEX_FIELDS)


def load_feature_outputs(
    fingerprint_path: str | Path,
    index_path: str | Path,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Load cached ECFP fingerprints and feature metadata from disk.

    Parameters
    ----------
    fingerprint_path
        ``.npz`` path containing a ``fingerprints`` array.
    index_path
        CSV index path with columns matching ``FEATURE_INDEX_FIELDS``.

    Returns
    -------
    tuple[numpy.ndarray, list[dict[str, Any]]]
        Fingerprint matrix (float32) and metadata rows with
        ``functional_groups`` restored to a list.
    """
    matrix = np.load(fingerprint_path)["fingerprints"].astype(np.float32)
    df = pd.read_csv(index_path)
    rows: list[dict[str, Any]] = []
    for _, series in df.iterrows():
        row = series.to_dict()
        fg_raw = row.get("functional_groups", "")
        if isinstance(fg_raw, str) and fg_raw.strip():
            row["functional_groups"] = [g for g in fg_raw.split(";") if g]
        else:
            row["functional_groups"] = []
        rows.append(row)
    return matrix, rows



def build_sample_features(
    samples: list[dict[str, Any]],
    bits: int = 1024,
    radius: int = 2,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Build ECFP fingerprint matrix and metadata rows using skfp.

    Valid SMILES are collected in a single pass, then fingerprints are
    computed in one batch via ``ECFPFingerprint.transform``.

    Parameters
    ----------
    samples
        SpectraLM sample dictionaries.
    bits
        Fingerprint vector length (``fp_size``).
    radius
        ECFP fingerprint radius.

    Returns
    -------
    tuple[numpy.ndarray, list[dict[str, Any]]]
        Fingerprint matrix (float32) and metadata rows for valid structures.
    """
    valid_smiles: list[str] = []
    rows: list[dict[str, Any]] = []

    for sample in tqdm(samples, total=len(samples), desc="Processing samples"):
        smiles = sample.get("canonical_smiles")
        row = {
            "id": sample.get("id", ""),
            "canonical_smiles": smiles or "",
            "murcko_scaffold": murcko_scaffold(smiles) if smiles else "",
            "functional_groups": [],
            "feature_status": "invalid_smiles",
        }

        row.update(
            {
                "functional_groups": functional_group_labels(smiles),
                "feature_status": "ok",
            }
        )
        valid_smiles.append(smiles)
        rows.append(row)

    if not valid_smiles:
        return np.empty((0, bits), dtype=np.float32), []

    # Batch compute fingerprints
    ecfp = ECFPFingerprint(fp_size=bits, radius=radius, n_jobs=-1, verbose=1)
    fingerprints = ecfp.transform(valid_smiles).astype(np.float32)

    # Assign sequential row indices
    for idx, row in enumerate(rows):
        row["row_index"] = idx

    return fingerprints, rows


if __name__ == "__main__":
    dataset = "dataset/NMRexp_spectra_dataset.pkl"
    fingerprint_path = "dataset/features/morgan_fingerprints.npz"
    index_path = "dataset/features/morgan_feature_index.csv"

    samples = load_pickle_list(dataset)
    matrix, rows = build_sample_features(samples, bits=1024, radius=2)

    save_feature_outputs(matrix, rows, fingerprint_path, index_path)
    print(
        f"Wrote {fingerprint_path}; rows={len(rows)}; "
        f"bits={matrix.shape[1] if matrix.ndim == 2 else 0}"
    )
