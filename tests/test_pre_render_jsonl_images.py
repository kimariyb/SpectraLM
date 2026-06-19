"""Tests for the JSONL pre-rendering utility."""

from __future__ import annotations

import json
from pathlib import Path

from script.pre_render_jsonl_images import pre_render_images


def test_pre_render_images_writes_pngs_and_manifest(
    tmp_path: Path,
    ethanol_sample,
) -> None:
    """Pre-rendering should write 1H/13C PNGs for ids selected by split files."""
    sample = dict(ethanol_sample)
    sample["id"] = "sample-0"
    (tmp_path / "samples.jsonl").write_text(
        json.dumps(sample) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "train_ids.txt").write_text("sample-0\n", encoding="utf-8")

    out_dir = tmp_path / "rendered"
    summary = pre_render_images(
        tmp_path,
        out_dir,
        splits=["train"],
        image_size=(128, 72),
        num_workers=1,
    )

    assert summary["rendered"] == 1
    assert (out_dir / "sample-0_1h.png").exists()
    assert (out_dir / "sample-0_13c.png").exists()
    assert (out_dir / "render_manifest.json").exists()
