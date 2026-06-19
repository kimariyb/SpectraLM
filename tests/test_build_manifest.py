"""Tests for QC and split manifest construction."""

from __future__ import annotations

from script.build_manifest import build_manifest, sample_manifest_row


def test_sample_manifest_row_marks_missing_peaks_failed(ethanol_sample) -> None:
    """Rows with missing paired spectra should be excluded by QC."""
    sample = dict(ethanol_sample)
    sample["13C_NMR"] = {"peaks": []}

    row = sample_manifest_row(sample)

    assert row["qc_status"] == "fail"
    assert "missing_13c_peaks" in row["qc_reason"]


def test_build_manifest_assigns_scaffold_disjoint_splits(ethanol_sample) -> None:
    """A scaffold must not appear in more than one split."""
    samples = []
    smiles_by_scaffold = ["CCO", "CCO", "CCN", "CCN", "c1ccccc1", "c1ccccc1"]
    for idx, smiles in enumerate(smiles_by_scaffold):
        sample = dict(ethanol_sample)
        sample["id"] = f"sample-{idx}"
        sample["smiles"] = smiles
        sample["canonical_smiles"] = smiles
        samples.append(sample)

    rows, summary = build_manifest(
        samples,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=1,
    )

    scaffold_to_split = {}
    for row in rows:
        scaffold = row["murcko_scaffold"]
        split = row["split"]
        assert scaffold_to_split.setdefault(scaffold, split) == split

    assert summary["scaffold_overlap_counts"] == {
        "train_val": 0,
        "train_test": 0,
        "val_test": 0,
    }
    assert sum(summary["split_counts"].values()) == len(samples)
