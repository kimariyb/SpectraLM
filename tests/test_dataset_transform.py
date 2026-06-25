"""Tests for text-only lazy JSONL datasets and message transforms."""

from __future__ import annotations

import json
from pathlib import Path

from src.data.dataset import (
    LazyNMRJsonlDataset,
    NMRMessageTransform,
    load_lazy_nmr_dataset,
    load_raw_nmr_samples,
)
from src.evaluation.prompts import SYSTEM_PROMPT


def _write_lazy_jsonl_fixture(
    tmp_path: Path,
    sample: dict,
    *,
    sample_id: str = "sample-0",
) -> None:
    """Write one JSONL sample and a matching train split."""
    row = dict(sample)
    row["id"] = sample_id
    (tmp_path / "samples.jsonl").write_text(
        json.dumps(row) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "train_ids.txt").write_text(
        sample_id + "\n",
        encoding="utf-8",
    )


def test_lazy_dataset_reuses_cached_offsets(
    tmp_path: Path,
    ethanol_sample,
    monkeypatch,
) -> None:
    """A valid offset cache should avoid rescanning the full JSONL file."""
    _write_lazy_jsonl_fixture(tmp_path, ethanol_sample)
    first = load_lazy_nmr_dataset(tmp_path, split="train")
    assert len(first) == 1
    assert (tmp_path / ".offset_cache" / "train.npy").exists()
    assert (tmp_path / ".offset_cache" / "train.json").exists()

    def fail_scan(self, split_ids):
        raise AssertionError("JSONL was rescanned instead of using its offset cache")

    monkeypatch.setattr(
        LazyNMRJsonlDataset,
        "_scan_offsets",
        fail_scan,
        raising=False,
    )

    second = load_lazy_nmr_dataset(tmp_path, split="train")
    assert len(second) == 1


def test_lazy_dataset_invalidates_offsets_when_split_changes(
    tmp_path: Path,
    ethanol_sample,
) -> None:
    """Changing split IDs should rebuild offsets instead of using stale rows."""
    rows = []
    for idx in range(2):
        row = dict(ethanol_sample)
        row["id"] = f"sample-{idx}"
        rows.append(row)
    (tmp_path / "samples.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    split_path = tmp_path / "train_ids.txt"
    split_path.write_text("sample-0\n", encoding="utf-8")

    assert len(load_lazy_nmr_dataset(tmp_path, split="train")) == 1

    split_path.write_text("sample-0\nsample-1\n", encoding="utf-8")

    assert len(load_lazy_nmr_dataset(tmp_path, split="train")) == 2


def test_lazy_dataset_reuses_jsonl_handle(tmp_path: Path, ethanol_sample) -> None:
    """Repeated item reads in one process should share one open JSONL handle."""
    _write_lazy_jsonl_fixture(tmp_path, ethanol_sample)
    dataset = load_lazy_nmr_dataset(tmp_path, split="train")

    dataset._load_sample_at(dataset.offsets[0])
    first_handle = dataset._jsonl_handle
    dataset._load_sample_at(dataset.offsets[0])

    assert first_handle is not None
    assert dataset._jsonl_handle is first_handle
    assert not first_handle.closed
    dataset.close()
    assert first_handle.closed


def test_lazy_dataset_pickle_state_drops_jsonl_handle(
    tmp_path: Path,
    ethanol_sample,
) -> None:
    """Spawned DataLoader workers must open their own JSONL handle."""
    _write_lazy_jsonl_fixture(tmp_path, ethanol_sample)
    dataset = load_lazy_nmr_dataset(tmp_path, split="train")
    dataset._load_sample_at(dataset.offsets[0])

    state = dataset.__getstate__()

    assert state["_jsonl_handle"] is None
    assert state["_jsonl_handle_pid"] is None
    dataset.close()


def test_message_transform_emits_system_user_assistant_text(ethanol_sample) -> None:
    """Training examples should be pure text chat messages."""
    transform = NMRMessageTransform(seed=1, prompt_template_index=0)
    output = transform({"sample": [ethanol_sample]})

    messages = output["messages"][0]
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
    ]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert "1H NMR:" in messages[1]["content"]
    assert "13C NMR:" in messages[1]["content"]
    assert "Molecular formula: C2H6O" in messages[1]["content"]
    assert json.loads(messages[2]["content"]) == {"smiles": "CCO"}


def test_message_transform_can_omit_formula(ethanol_sample) -> None:
    """Formula-free training should not leak formula through prompts."""
    transform = NMRMessageTransform(
        seed=1,
        include_formula=False,
        prompt_template_index=0,
    )

    output = transform({"sample": [ethanol_sample]})
    prompt = output["messages"][0][1]["content"]

    assert "Molecular formula:" not in prompt
    assert "1H NMR:" in prompt
    assert "13C NMR:" in prompt


def test_message_transform_can_add_formula_free_rule_context(ethanol_sample) -> None:
    """Training messages should propagate the opt-in rule-context setting."""
    transform = NMRMessageTransform(
        seed=1,
        include_formula=False,
        include_rule_context=True,
        max_rule_evidence=4,
        prompt_template_index=0,
    )

    output = transform({"sample": [ethanol_sample]})
    prompt = output["messages"][0][1]["content"]

    assert "## Derived 1D NMR Constraints" in prompt
    assert "ethyl fragment" in prompt
    assert "Molecular formula:" not in prompt
    assert "DBE" not in prompt


def test_message_transform_can_force_functional_group_task(ethanol_sample) -> None:
    """Task weights should select auxiliary supervision."""
    transform = NMRMessageTransform(
        seed=1,
        task_weights={"functional_group_recognition": 1.0},
        prompt_template_index=0,
    )
    output = transform({"sample": [ethanol_sample]})

    messages = output["messages"][0]
    assert "functional group" in messages[1]["content"].lower()
    assert messages[2]["content"] == '["alcohol"]'


def test_message_transform_falls_back_when_candidates_are_missing(
    ethanol_sample,
) -> None:
    """Unavailable candidate sets should fall back to the main task."""
    transform = NMRMessageTransform(
        seed=1,
        task_weights={"candidate_ranking": 1.0},
        candidate_map={},
        prompt_template_index=0,
    )
    output = transform({"sample": [ethanol_sample]})

    messages = output["messages"][0]
    assert '"smiles"' in messages[1]["content"]
    assert json.loads(messages[2]["content"]) == {"smiles": "CCO"}


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

    train = load_raw_nmr_samples(tmp_path, split="train")
    val = load_raw_nmr_samples(tmp_path, split="validation")

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
    (subsets_dir / "clean_10k_train_ids.txt").write_text(
        "sample-1\nsample-2\n",
        encoding="utf-8",
    )

    train = load_raw_nmr_samples(tmp_path, split="clean_10k_train")

    assert [sample["id"] for sample in train] == ["sample-1", "sample-2"]


def test_load_lazy_nmr_dataset_from_jsonl_directory(tmp_path: Path, ethanol_sample) -> None:
    """Lazy JSONL dataset should return text-only chat rows."""
    _write_lazy_jsonl_fixture(tmp_path, ethanol_sample)

    dataset = load_lazy_nmr_dataset(
        tmp_path,
        split="train",
        prompt_template_index=0,
    )

    row = dataset[0]
    assert len(dataset) == 1
    assert [message["role"] for message in row["messages"]] == [
        "system",
        "user",
        "assistant",
    ]
    assert json.loads(row["messages"][2]["content"]) == {"smiles": "CCO"}


def test_lazy_dataset_loads_candidate_sidecar_for_ranking(
    tmp_path: Path,
    ethanol_sample,
) -> None:
    """Lazy training should connect sidecar candidates to ranking prompts."""
    sample = dict(ethanol_sample)
    sample["id"] = "sample-0"
    (tmp_path / "samples.jsonl").write_text(
        json.dumps(sample) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "train_ids.txt").write_text("sample-0\n", encoding="utf-8")
    sidecar = tmp_path / "candidates.jsonl"
    sidecar.write_text(
        json.dumps(
            {
                "id": "sample-0",
                "target": "CCO",
                "candidates": ["COC", "CCO"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = load_lazy_nmr_dataset(
        tmp_path,
        split="train",
        task_weights={"candidate_ranking": 1.0},
        candidate_sidecar_path=sidecar,
        prompt_template_index=0,
    )
    row = dataset[0]
    prompt = row["messages"][1]["content"]

    assert "1. COC" in prompt
    assert "2. CCO" in prompt
    assert json.loads(row["messages"][2]["content"]) == {"smiles": "CCO"}
