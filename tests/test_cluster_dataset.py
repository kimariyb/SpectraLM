"""Tests for Morgan fingerprint and Butina dataset construction."""

from __future__ import annotations

import csv
import pickle
from pathlib import Path

from spectralm.data.sampling import select_cluster_representatives
from spectralm.data.clustering import cluster_samples
from spectralm.data.features import build_sample_features, feature_csv_rows
from spectralm.io import write_pickle


def make_sample(idx: int, smiles: str, h_shifts: list[float], c_shifts: list[float]) -> dict:
    """Build a compact paired-NMR sample for clustered sampling tests."""
    return {
        "id": f"sample-{idx}",
        "smiles": smiles,
        "canonical_smiles": smiles,
        "selfies": "",
        "molecular_formula": "",
        "murcko_scaffold": f"scaffold-{idx}",
        "1H_NMR": {
            "peaks": [
                {"shift": shift, "multiplicity": "s", "J": [], "integration": 1.0}
                for shift in h_shifts
            ]
        },
        "13C_NMR": {"peaks": [{"shift": shift} for shift in c_shifts]},
    }


def test_build_sample_features_uses_only_structure_and_has_stable_dimensions(ethanol_sample) -> None:
    """Feature extraction should skip invalid structures and use only structure fields."""
    invalid = dict(ethanol_sample)
    invalid["id"] = "invalid"
    invalid["canonical_smiles"] = "not-a-smiles"
    fingerprints, rows = build_sample_features(
        [ethanol_sample, invalid],
        {
            "fingerprint_bits": 128,
        },
    )
    assert fingerprints.shape == (1, 128)
    assert rows[0]["id"] == "ethanol"
    assert rows[0]["feature_status"] == "ok"
    assert "h_peak_count" not in rows[0]
    assert "c_peak_count" not in rows[0]


def test_cluster_samples_returns_one_label_per_feature_row() -> None:
    """Butina clustering should return one label for each fingerprint row."""
    samples = [
        make_sample(0, "CCO", [1.2, 3.6], [18.0, 58.0]),
        make_sample(1, "CCN", [1.1, 2.7], [15.0, 42.0]),
        make_sample(2, "c1ccccc1", [7.2], [128.0]),
    ]
    fingerprints, _ = build_sample_features(samples, {"fingerprint_bits": 128})
    result = cluster_samples(fingerprints, {"butina_cutoff": 0.7})
    assert len(result.labels) == len(fingerprints)
    assert result.method == "butina"
    assert result.cluster_count >= 1


def test_cluster_samples_can_run_scaffold_bucketed_butina() -> None:
    """Bucketed Butina should avoid a single global distance calculation."""
    samples = [
        make_sample(0, "CCO", [1.2, 3.6], [18.0, 58.0]),
        make_sample(1, "CCN", [1.1, 2.7], [15.0, 42.0]),
        make_sample(2, "CCCl", [1.5, 3.4], [20.0, 45.0]),
        make_sample(3, "c1ccccc1", [7.2], [128.0]),
        make_sample(4, "Cc1ccccc1", [2.3, 7.2], [21.0, 128.0]),
    ]
    fingerprints, rows = build_sample_features(samples, {"fingerprint_bits": 128})
    result = cluster_samples(
        fingerprints,
        {
            "butina_cutoff": 0.7,
            "bucketed": True,
            "max_bucket_size": 2,
        },
        rows,
    )
    assert len(result.labels) == len(fingerprints)
    assert result.method == "bucketed_butina"
    assert result.bucket_count >= 3
    assert max(result.bucket_sizes) <= 2


def test_cluster_representatives_respect_sizes_and_scaffold_disjoint() -> None:
    """Cluster representative selection should satisfy split sizes without scaffold overlap."""
    samples = [
        make_sample(0, "CCO", [1.2, 3.6], [18.0, 58.0]),
        make_sample(1, "CCN", [1.1, 2.7], [15.0, 42.0]),
        make_sample(2, "CCCl", [1.5, 3.4], [20.0, 45.0]),
        make_sample(3, "c1ccccc1", [7.2], [128.0]),
        make_sample(4, "C1CCCCC1", [1.4], [27.0]),
        make_sample(5, "CC(=O)O", [2.1], [20.0, 178.0]),
    ]
    fingerprints, rows = build_sample_features(samples, {"fingerprint_bits": 64})
    labels = cluster_samples(fingerprints, {"butina_cutoff": 0.85}).labels
    selected = select_cluster_representatives(
        samples,
        rows,
        labels,
        fingerprints,
        {
            "train_size": 3,
            "test_size": 2,
            "max_per_scaffold": 1,
            "seed": 11,
        },
    )
    assert len(selected.train) == 3
    assert len(selected.test) == 2
    assert {row["murcko_scaffold"] for row in selected.train}.isdisjoint(
        {row["murcko_scaffold"] for row in selected.test}
    )
    assert selected.report["scaffold_overlap_train_test"] == 0


def test_feature_csv_rows_are_csv_serializable(ethanol_sample, tmp_path: Path) -> None:
    """Feature metadata rows should be serializable to CSV."""
    _, rows = build_sample_features([ethanol_sample], {"fingerprint_bits": 64})
    csv_rows = feature_csv_rows(rows)
    output = tmp_path / "features.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    assert output.read_text(encoding="utf-8").startswith("row_index,id,canonical_smiles")


def test_cluster_sample_cli_writes_subset_outputs(tmp_path: Path, ethanol_sample) -> None:
    """The cluster sampling CLI should write train/test subsets and reports."""
    from spectralm.data.sampling import main

    samples = []
    for idx, smiles in enumerate(["CCO", "CCN", "CCCl", "c1ccccc1"]):
        sample = dict(ethanol_sample)
        sample["id"] = f"cli-{idx}"
        sample["canonical_smiles"] = smiles
        sample["smiles"] = smiles
        sample["murcko_scaffold"] = f"cli-scaffold-{idx}"
        samples.append(sample)
    dataset = tmp_path / "dataset.pkl"
    config = tmp_path / "cluster.yaml"
    out_dir = tmp_path / "subset"
    features = tmp_path / "features.npz"
    index = tmp_path / "feature_index.csv"
    write_pickle(dataset, samples)
    config.write_text(
        "\n".join(
            [
                f"dataset: {dataset}",
                f"fingerprints: {features}",
                f"index: {index}",
                f"out_dir: {out_dir}",
                "train_size: 2",
                "test_size: 1",
                "butina_cutoff: 0.8",
                "fingerprint_bits: 64",
                "seed: 5",
            ]
        ),
        encoding="utf-8",
    )
    main(["--config", str(config)])
    with (out_dir / "train.pkl").open("rb") as handle:
        train = pickle.load(handle)
    with (out_dir / "test.pkl").open("rb") as handle:
        test = pickle.load(handle)
    assert len(train) == 2
    assert len(test) == 1
    assert features.exists()
    assert index.exists()
    assert (out_dir / "selected_samples.csv").exists()
    assert (out_dir / "cluster_report.json").exists()
