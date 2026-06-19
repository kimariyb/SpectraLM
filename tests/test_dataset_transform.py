"""Tests for dataset splitting and message transforms."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image as PILImage

import src.data.dataset as dataset_module
from src.data.dataset import (
    LazyNMRJsonlDataset,
    NMRMessageTransform,
    _resize_image,
    load_lazy_nmr_dataset,
    load_raw_nmr_samples,
)


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


def test_resize_image_skips_resize_when_dimensions_match(monkeypatch) -> None:
    """Already-sized pre-rendered images should not be resampled."""
    image = PILImage.new("RGB", (32, 18), color=(255, 255, 255))
    original_resize = PILImage.Image.resize
    resize_calls: list[tuple[int, int]] = []

    def tracking_resize(self, size, *args, **kwargs):
        resize_calls.append(tuple(size))
        return original_resize(self, size, *args, **kwargs)

    monkeypatch.setattr(PILImage.Image, "resize", tracking_resize)

    result = _resize_image(image, (32, 18))

    assert result.size == (32, 18)
    assert resize_calls == []


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


def test_message_transform_can_omit_formula(ethanol_sample) -> None:
    """Formula-free training should not leak formula through prompts."""
    transform = NMRMessageTransform(
        seed=1,
        include_formula=False,
    )
    batch = {
        "h_image": [None],
        "c_image": [None],
        "sample": [ethanol_sample],
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
    (subsets_dir / "clean_50k_train_ids.txt").write_text(
        "sample-1\nsample-2\n",
        encoding="utf-8",
    )

    train = load_raw_nmr_samples(tmp_path, split="clean_50k_train")

    assert [sample["id"] for sample in train] == ["sample-1", "sample-2"]


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
        image_size=(64, 64),
    )

    row = ds[0]
    assert len(ds) == 1
    assert [message["role"] for message in row["messages"]] == ["user", "assistant"]
    assert row["messages"][0]["content"][0]["image"].size == (64, 64)
    assert row["messages"][1]["content"][0]["text"] == "CCO"


def test_load_lazy_nmr_dataset_with_pre_rendered_images(
    tmp_path: Path,
    ethanol_sample,
) -> None:
    """Pre-rendered image backend should read PNGs instead of drawing spectra."""
    sample = dict(ethanol_sample)
    sample["id"] = "sample-0"
    (tmp_path / "samples.jsonl").write_text(
        json.dumps(sample) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "train_ids.txt").write_text("sample-0\n", encoding="utf-8")

    rendered_dir = tmp_path / "rendered"
    rendered_dir.mkdir()
    PILImage.new("RGB", (32, 18), color=(255, 255, 255)).save(
        rendered_dir / "sample-0_1h.png"
    )
    PILImage.new("RGB", (32, 18), color=(240, 240, 240)).save(
        rendered_dir / "sample-0_13c.png"
    )

    ds = load_lazy_nmr_dataset(
        tmp_path,
        split="train",
        image_size=(128, 72),
        image_backend="pre_rendered",
        rendered_image_dir=rendered_dir,
    )

    row = ds[0]

    assert row["messages"][0]["content"][0]["image"].size == (128, 72)
    assert row["messages"][0]["content"][1]["image"].size == (128, 72)
    assert row["messages"][1]["content"][0]["text"] == "CCO"


def test_load_sample_images_supports_pre_rendered_inference(
    tmp_path: Path,
    ethanol_sample,
) -> None:
    """Inference should reuse the same pre-rendered images as training."""
    rendered_dir = tmp_path / "rendered"
    rendered_dir.mkdir()
    PILImage.new("RGB", (32, 18), color=(255, 255, 255)).save(
        rendered_dir / "ethanol_1h.png"
    )
    PILImage.new("RGB", (32, 18), color=(240, 240, 240)).save(
        rendered_dir / "ethanol_13c.png"
    )
    load_sample_images = getattr(dataset_module, "load_sample_images", None)

    assert callable(load_sample_images)
    images = load_sample_images(
        ethanol_sample,
        image_backend="pre_rendered",
        rendered_image_dir=rendered_dir,
        image_size=(32, 18),
    )

    assert [image.size for image in images] == [(32, 18), (32, 18)]


def test_load_lazy_nmr_dataset_missing_pre_rendered_images_raises(
    tmp_path: Path,
    ethanol_sample,
) -> None:
    """Strict pre-rendered mode should fail fast when image files are absent."""
    sample = dict(ethanol_sample)
    sample["id"] = "sample-0"
    (tmp_path / "samples.jsonl").write_text(
        json.dumps(sample) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "train_ids.txt").write_text("sample-0\n", encoding="utf-8")
    rendered_dir = tmp_path / "rendered"
    rendered_dir.mkdir()

    ds = load_lazy_nmr_dataset(
        tmp_path,
        split="train",
        image_backend="pre_rendered",
        rendered_image_dir=rendered_dir,
        missing_image_policy="error",
    )

    try:
        ds[0]
    except FileNotFoundError as exc:
        assert "Missing pre-rendered NMR image" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing PNGs")
