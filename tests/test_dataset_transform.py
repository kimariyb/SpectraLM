"""Tests for dataset splitting and message transforms."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from src.data.dataset import (
    NMRMessageTransform,
    load_lazy_nmr_dataset,
    _resolve_and_load_samples,
    _split_by_scaffold,
    load_nmr_dataset,
)


def test_split_by_scaffold_keeps_scaffolds_disjoint(ethanol_sample) -> None:
    """Train/test split should not place one scaffold in both splits."""
    samples = []
    for idx, scaffold in enumerate(["a", "a", "b", "b", "c", "c"]):
        sample = dict(ethanol_sample)
        sample["id"] = f"sample-{idx}"
        sample["murcko_scaffold"] = scaffold
        samples.append(sample)

    train = _split_by_scaffold(samples, train_size=0.5, split="train")
    test = _split_by_scaffold(samples, train_size=0.5, split="test")

    train_scaffolds = {sample["murcko_scaffold"] for sample in train}
    test_scaffolds = {sample["murcko_scaffold"] for sample in test}

    assert train_scaffolds.isdisjoint(test_scaffolds)
    assert len(train) + len(test) == len(samples)


def test_message_transform_can_emit_reasoning_target(ethanol_sample) -> None:
    """Reasoning target mode should train structured output, not only SMILES."""
    transform = NMRMessageTransform(
        task_probs={"structure": 1.0},
        seed=1,
        target_format="reasoning",
    )
    batch = {
        "h_image": [None],
        "c_image": [None],
        "sample_pickle": [pickle.dumps(ethanol_sample)],
    }

    output = transform(batch)
    target = output["messages"][0][1]["content"][0]["text"]

    assert "Spectral reasoning:" in target
    assert "Final SELFIES:" in target
    assert "Final canonical SMILES:" in target


def test_message_transform_can_omit_formula(ethanol_sample) -> None:
    """Formula-free training should not leak formula through prompts."""
    transform = NMRMessageTransform(
        task_probs={"structure": 1.0},
        seed=1,
        include_formula=False,
    )
    batch = {
        "h_image": [None],
        "c_image": [None],
        "sample_pickle": [pickle.dumps(ethanol_sample)],
    }

    output = transform(batch)
    prompt = output["messages"][0][0]["content"][2]["text"]

    assert "Molecular formula:" not in prompt


def test_resolve_jsonl_samples_with_split_ids(tmp_path: Path, ethanol_sample) -> None:
    """JSONL datasets should load only ids listed in the requested split."""
    samples = []
    for idx in range(3):
        sample = dict(ethanol_sample)
        sample["id"] = f"sample-{idx}"
        samples.append(sample)

    (tmp_path / "samples.jsonl").write_text(
        "\n".join(json.dumps(sample) for sample in samples) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "train_ids.txt").write_text("sample-0\nsample-2\n", encoding="utf-8")
    (tmp_path / "val_ids.txt").write_text("sample-1\n", encoding="utf-8")

    train = _resolve_and_load_samples(tmp_path, split="train", train_size=0.8)
    val = _resolve_and_load_samples(tmp_path, split="validation", train_size=0.8)

    assert [sample["id"] for sample in train] == ["sample-0", "sample-2"]
    assert [sample["id"] for sample in val] == ["sample-1"]


def test_resolve_jsonl_samples_with_nested_subset_ids(tmp_path: Path, ethanol_sample) -> None:
    """Named curation subsets under subsets/ should be valid split names."""
    samples = []
    for idx in range(3):
        sample = dict(ethanol_sample)
        sample["id"] = f"sample-{idx}"
        samples.append(sample)

    (tmp_path / "samples.jsonl").write_text(
        "\n".join(json.dumps(sample) for sample in samples) + "\n",
        encoding="utf-8",
    )
    subsets_dir = tmp_path / "subsets"
    subsets_dir.mkdir()
    (subsets_dir / "clean_50k_train_ids.txt").write_text(
        "sample-1\nsample-2\n",
        encoding="utf-8",
    )

    train = _resolve_and_load_samples(
        tmp_path,
        split="clean_50k_train",
        train_size=0.8,
    )

    assert [sample["id"] for sample in train] == ["sample-1", "sample-2"]


def test_load_nmr_dataset_from_jsonl_directory(tmp_path: Path, ethanol_sample) -> None:
    """HF builder should support samples.jsonl plus split id files."""
    sample = dict(ethanol_sample)
    sample["id"] = "sample-0"
    (tmp_path / "samples.jsonl").write_text(
        json.dumps(sample) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "train_ids.txt").write_text("sample-0\n", encoding="utf-8")

    ds = load_nmr_dataset(
        tmp_path,
        split="train",
        hf_cache_dir=str(tmp_path / "hf_cache"),
        render_cache_dir=str(tmp_path / "render_cache"),
        image_size=(64, 64),
        with_messages=False,
    )

    assert len(ds) == 1
    assert ds[0]["id"] == "sample-0"
    assert ds[0]["smiles"] == "CCO"


def test_load_lazy_nmr_dataset_from_jsonl_directory(tmp_path: Path, ethanol_sample) -> None:
    """Lazy JSONL dataset should render images only when indexed."""
    sample = dict(ethanol_sample)
    sample["id"] = "sample-0"
    (tmp_path / "samples.jsonl").write_text(
        json.dumps(sample) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "train_ids.txt").write_text("sample-0\n", encoding="utf-8")

    ds = load_lazy_nmr_dataset(
        tmp_path,
        split="train",
        task_probs={"structure": 1.0},
        target_format="reasoning",
        image_size=(64, 64),
    )

    row = ds[0]
    assert len(ds) == 1
    assert [message["role"] for message in row["messages"]] == ["user", "assistant"]
    assert row["messages"][0]["content"][0]["image"].size == (64, 64)
    assert "Final canonical SMILES:" in row["messages"][1]["content"][0]["text"]
