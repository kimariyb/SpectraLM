"""Tests for manifest-driven JSONL subset curation."""

from __future__ import annotations

from pathlib import Path

from script import curate_jsonl_subsets as curation


build_subsets = curation.build_subsets
filter_rows = curation.filter_rows


def test_default_subset_sizes_match_50k_scaling_design() -> None:
    """Default curation should materialize the approved nested scaling sets."""
    assert getattr(curation, "DEFAULT_SUBSET_SIZES", None) == [
        5_000,
        10_000,
        25_000,
        50_000,
    ]


def _manifest_row(
    sample_id: str,
    split: str,
    scaffold: str,
    *,
    heavy_atoms: int = 8,
    h_peaks: int = 3,
    c_peaks: int = 4,
    solvent: str = "CDCl3",
    qc_status: str = "pass",
    canonical_smiles: str = "CCO",
    isotope_label_count: int = 0,
) -> dict[str, str]:
    return {
        "id": sample_id,
        "split": split,
        "qc_status": qc_status,
        "qc_reason": "",
        "canonical_smiles": canonical_smiles,
        "isotope_label_count": str(isotope_label_count),
        "molecular_formula": "C2H6O",
        "murcko_scaffold": scaffold,
        "heavy_atom_count": str(heavy_atoms),
        "h_peak_count": str(h_peaks),
        "c_peak_count": str(c_peaks),
        "h_solvent": solvent,
        "c_solvent": solvent,
        "h_frequency": "400 MHz",
        "c_frequency": "101 MHz",
    }


def test_filter_rows_rejects_out_of_range_manifest_rows() -> None:
    """Manifest QC filters should reject obvious low-quality records."""
    rows = [
        _manifest_row("ok", "train", "s1"),
        _manifest_row("too-big", "train", "s2", heavy_atoms=99),
        _manifest_row("no-peaks", "train", "s3", h_peaks=0),
    ]

    kept, rejected = filter_rows(rows, max_heavy_atoms=60)

    assert [row["id"] for row in kept] == ["ok"]
    assert rejected["too_many_heavy_atoms"] == 1
    assert rejected["too_few_1h_peaks"] == 1


def test_filter_rows_rechecks_legacy_manifest_molecule_policy() -> None:
    """Old pass-status rows must not bypass current structure requirements."""
    rows = [
        _manifest_row("ok", "train", "s1"),
        _manifest_row(
            "salt",
            "train",
            "s2",
            canonical_smiles="C[NH3+].[Cl-]",
        ),
        _manifest_row(
            "charged",
            "train",
            "s3",
            canonical_smiles="[NH4+]",
        ),
        _manifest_row(
            "radical",
            "train",
            "s4",
            canonical_smiles="[CH3]",
        ),
        _manifest_row(
            "isotope",
            "train",
            "s5",
            canonical_smiles="CCO",
            isotope_label_count=1,
        ),
    ]

    kept, rejected = filter_rows(rows)

    assert [row["id"] for row in kept] == ["ok"]
    assert rejected == {
        "multiple_components": 1,
        "nonzero_formal_charge": 1,
        "radical": 1,
        "isotope_labeled_structure": 1,
    }


def test_build_subsets_writes_nested_scaling_id_files(tmp_path: Path) -> None:
    """Curation should write train/val/test ids for named nested subsets."""
    rows = []
    for idx in range(6):
        rows.append(_manifest_row(f"train-{idx}", "train", f"s{idx % 3}"))
    for idx in range(3):
        rows.append(_manifest_row(f"val-{idx}", "val", f"v{idx}"))
        rows.append(_manifest_row(f"test-{idx}", "test", f"t{idx}"))

    summary = build_subsets(
        rows,
        tmp_path,
        subset_sizes=[2, 4],
        val_size=2,
        test_size=2,
        prefix="clean",
        seed=1,
    )

    train_2 = (tmp_path / "clean_2_train_ids.txt").read_text(encoding="utf-8").splitlines()
    train_4 = (tmp_path / "clean_4_train_ids.txt").read_text(encoding="utf-8").splitlines()
    val_2 = (tmp_path / "clean_2_val_ids.txt").read_text(encoding="utf-8").splitlines()

    assert len(train_2) == 2
    assert len(train_4) == 4
    assert train_4[:2] == train_2
    assert len(val_2) == 2
    assert summary["subsets"]["clean_2"]["train"]["samples"] == 2
