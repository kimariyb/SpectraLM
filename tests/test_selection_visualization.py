"""Tests for PCA-style selected molecule visualization."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from spectralm.analysis.visualization import (
    build_selection_projection,
    pca_projection,
    plot_selection_projection,
)


def write_index_csv(path: Path) -> None:
    """Write a tiny fingerprint index CSV fixture.

    Parameters
    ----------
    path
        Output CSV path.
    """
    rows = [
        {"row_index": "0", "id": "a", "canonical_smiles": "CCO"},
        {"row_index": "1", "id": "b", "canonical_smiles": "CCN"},
        {"row_index": "2", "id": "c", "canonical_smiles": "c1ccccc1"},
        {"row_index": "3", "id": "d", "canonical_smiles": "CCCl"},
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_selected_csv(path: Path) -> None:
    """Write a tiny selected-samples CSV fixture.

    Parameters
    ----------
    path
        Output CSV path.
    """
    rows = [
        {"split": "train", "id": "a", "canonical_smiles": "CCO"},
        {"split": "test", "id": "c", "canonical_smiles": "c1ccccc1"},
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_build_selection_projection_marks_selected_points(tmp_path: Path) -> None:
    """Projection data should include all molecules and mark selected samples."""
    fingerprint_path = tmp_path / "fingerprints.npz"
    index_path = tmp_path / "index.csv"
    selected_path = tmp_path / "selected.csv"
    matrix = np.array(
        [
            [1, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 1, 1, 0],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(fingerprint_path, fingerprints=matrix)
    write_index_csv(index_path)
    write_selected_csv(selected_path)
    frame = build_selection_projection(fingerprint_path, index_path, selected_path)
    assert len(frame) == 4
    assert int(frame["selected"].sum()) == 2
    assert set(frame.loc[frame["selected"], "split"]) == {"train", "test"}


def test_pca_projection_uses_covariance_for_wide_binary_matrix() -> None:
    """PCA projection should handle matrices with many rows without full SVD."""
    matrix = np.tile(np.eye(4, dtype=np.float32), (10, 1))
    coords = pca_projection(matrix)
    assert coords.shape == (40, 2)
    assert np.isfinite(coords).all()


def test_plot_selection_projection_writes_image(tmp_path: Path) -> None:
    """Projection plotting should write a non-empty image file."""
    fingerprint_path = tmp_path / "fingerprints.npz"
    index_path = tmp_path / "index.csv"
    selected_path = tmp_path / "selected.csv"
    output = tmp_path / "selection_pca.png"
    np.savez_compressed(
        fingerprint_path,
        fingerprints=np.eye(5, dtype=np.float32),
    )
    rows = [
        {"row_index": str(idx), "id": chr(ord("a") + idx), "canonical_smiles": f"C{idx}"}
        for idx in range(5)
    ]
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_selected_csv(selected_path)
    frame = build_selection_projection(fingerprint_path, index_path, selected_path)
    plot_selection_projection(frame, output)
    assert output.exists()
    assert output.stat().st_size > 0
