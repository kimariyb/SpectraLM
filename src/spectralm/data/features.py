"""Structure-only Morgan fingerprint features for SpectraLM sampling."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from spectralm.config import add_config_argument, load_config
from spectralm.data.molecules import canonicalize_smiles, functional_group_labels, murcko_scaffold, sample_smiles
from spectralm.io import load_pickle_list, write_rows_csv


FEATURE_INDEX_FIELDS = [
    "row_index",
    "id",
    "canonical_smiles",
    "murcko_scaffold",
    "functional_groups",
    "feature_status",
]


def molecule_fingerprint(smiles: str, bits: int = 1024, radius: int = 2) -> np.ndarray | None:
    """Build a Morgan fingerprint bit vector.

    Parameters
    ----------
    smiles
        Canonical SMILES string.
    bits
        Fingerprint vector length.
    radius
        Morgan fingerprint radius.

    Returns
    -------
    numpy.ndarray | None
        Float32 Morgan fingerprint, or ``None`` when parsing fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bits)
    array = np.zeros((bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bitvect, array)
    return array


def sample_feature_vector(
    sample: dict[str, Any],
    config: dict[str, Any],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Build a structure-only Morgan fingerprint and metadata row.

    Parameters
    ----------
    sample
        SpectraLM sample dictionary.
    config
        Feature extraction configuration.

    Returns
    -------
    tuple[numpy.ndarray | None, dict[str, Any]]
        Fingerprint vector and metadata. The vector is ``None`` for invalid SMILES.
    """
    bits = int(config.get("fingerprint_bits", 1024))
    radius = int(config.get("fingerprint_radius", 2))
    smiles = canonicalize_smiles(sample_smiles(sample))
    row = {
        "id": sample.get("id", ""),
        "canonical_smiles": smiles or "",
        "murcko_scaffold": murcko_scaffold(smiles) if smiles else "",
        "functional_groups": [],
        "feature_status": "invalid_smiles",
    }
    if smiles is None:
        return None, row
    fingerprint = molecule_fingerprint(smiles, bits=bits, radius=radius)
    if fingerprint is None:
        return None, row
    row.update(
        {
            "functional_groups": functional_group_labels(smiles),
            "feature_status": "ok",
        }
    )
    return fingerprint.astype(np.float32), row


def build_sample_features(
    samples: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Build Morgan fingerprint matrix and metadata rows.

    Parameters
    ----------
    samples
        SpectraLM sample dictionaries.
    config
        Feature extraction configuration.

    Returns
    -------
    tuple[numpy.ndarray, list[dict[str, Any]]]
        Fingerprint matrix and metadata rows for valid structures.
    """
    cfg = config or {}
    fingerprints = []
    rows = []
    for sample in samples:
        fingerprint, row = sample_feature_vector(sample, cfg)
        if fingerprint is None:
            continue
        row["row_index"] = len(rows)
        fingerprints.append(fingerprint)
        rows.append(row)
    if not fingerprints:
        width = int(cfg.get("fingerprint_bits", 1024))
        return np.empty((0, width), dtype=np.float32), []
    return np.vstack(fingerprints).astype(np.float32), rows


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
    """Write Morgan fingerprints and feature metadata.

    Parameters
    ----------
    matrix
        Morgan fingerprint matrix.
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


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the Morgan fingerprint CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(description="Build structure-only Morgan fingerprints.")
    add_config_argument(parser)
    parser.add_argument("--dataset", default=None, help="Input sample pickle path.")
    parser.add_argument("--fingerprints", default=None, help="Output fingerprint NPZ path.")
    parser.add_argument("--index", default=None, help="Output feature index CSV path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run Morgan fingerprint extraction from the command line.

    Parameters
    ----------
    argv
        Optional command-line argument list.
    """
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)
    dataset = args.dataset or config.get("dataset", "dataset/NMRexp_spectra_dataset.pkl")
    fingerprint_path = args.fingerprints or config.get("fingerprints", "dataset/features/morgan_fingerprints.npz")
    index_path = args.index or config.get("index", "dataset/features/morgan_feature_index.csv")
    samples = load_pickle_list(dataset)
    matrix, rows = build_sample_features(samples, config)
    save_feature_outputs(matrix, rows, fingerprint_path, index_path)
    print(f"Wrote {fingerprint_path}; rows={len(rows)}; bits={matrix.shape[1] if matrix.ndim == 2 else 0}")


if __name__ == "__main__":
    main()
