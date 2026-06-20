"""Tests for formula-matched candidate-ranking sidecars."""

from __future__ import annotations

import json
from pathlib import Path

from script.build_candidate_sidecar import build_candidate_sidecar
from src.data.dataset import load_candidate_map
from src.data.molecules import molecule_formula


def _write_candidate_dataset(tmp_path: Path) -> Path:
    rows = [
        {"id": "propanol-1", "canonical_smiles": "CCCO"},
        {"id": "propanol-2", "canonical_smiles": "CC(C)O"},
        {"id": "methoxyethane", "canonical_smiles": "CCOC"},
        {"id": "ethylamine", "canonical_smiles": "CCN"},
    ]
    for row in rows:
        row["molecular_formula"] = molecule_formula(row["canonical_smiles"])
    (tmp_path / "samples.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "train_ids.txt").write_text(
        "\n".join(row["id"] for row in rows) + "\n",
        encoding="utf-8",
    )
    return tmp_path


def test_candidate_sidecar_uses_formula_matched_hard_negatives(
    tmp_path: Path,
) -> None:
    """Every candidate set should contain one target and only formula isomers."""
    dataset_dir = _write_candidate_dataset(tmp_path)
    output = tmp_path / "candidate_sets.jsonl"

    report = build_candidate_sidecar(
        dataset_dir,
        split="train",
        output_path=output,
        candidates_per_sample=3,
        seed=7,
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]

    assert report["input_samples"] == 4
    assert report["candidate_sets"] == 3
    assert report["omitted_without_negatives"] == 1
    for row in rows:
        assert row["target"] in row["candidates"]
        assert len(row["candidates"]) == 3
        assert len(set(row["candidates"])) == 3
        assert {
            molecule_formula(candidate) for candidate in row["candidates"]
        } == {row["molecular_formula"]}
        assert row["negative_tanimoto"] == sorted(
            row["negative_tanimoto"], reverse=True
        )


def test_candidate_sidecar_is_deterministic_and_loadable(tmp_path: Path) -> None:
    """A fixed seed should reproduce candidate ordering and map loading."""
    dataset_dir = _write_candidate_dataset(tmp_path)
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    build_candidate_sidecar(dataset_dir, "train", first, seed=13)
    build_candidate_sidecar(dataset_dir, "train", second, seed=13)

    assert first.read_text() == second.read_text()
    candidate_map = load_candidate_map(first)
    assert set(candidate_map) == {"propanol-1", "propanol-2", "methoxyethane"}
    assert all(len(candidates) == 3 for candidates in candidate_map.values())
