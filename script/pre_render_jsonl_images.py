"""Pre-render NMR spectrum images for a paired JSONL dataset.

The script keeps ``samples.jsonl`` as the source of truth and writes only PNG
images.  Training can then use ``image_backend: pre_rendered`` to load images
directly instead of drawing spectra during ``__getitem__``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

# Allow running from project root without PYTHONPATH.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import (  # noqa: E402
    _jsonl_sample_iter,
    _safe_cache_key,
    _split_ids_path,
    render_sample_images,
)
from src.spectra.render import HEIGHT_PX, WIDTH_PX  # noqa: E402


def read_split_ids(dataset_dir: str | Path, splits: list[str]) -> set[str]:
    """Read the union of sample ids for requested split names."""
    base = Path(dataset_dir)
    ids: set[str] = set()
    for split in splits:
        path = _split_ids_path(base, split)
        if not path.exists():
            raise FileNotFoundError(f"Split id file not found: {path}")
        ids.update(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return ids


def iter_selected_samples(
    dataset_dir: str | Path,
    split_ids: set[str],
) -> Iterable[dict[str, Any]]:
    """Yield JSONL samples whose ids are in ``split_ids``."""
    jsonl_path = Path(dataset_dir) / "samples.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL dataset not found: {jsonl_path}")
    for sample in _jsonl_sample_iter(jsonl_path):
        if str(sample.get("id", "")) in split_ids:
            yield sample


def _render_one(task: dict[str, Any]) -> dict[str, Any]:
    """Render and save images for one sample."""
    sample = task["sample"]
    out_dir = Path(task["out_dir"])
    image_size = tuple(task["image_size"]) if task["image_size"] else None
    sample_id = str(sample.get("id", "unknown"))
    cache_key = _safe_cache_key(sample_id)
    h_path = out_dir / f"{cache_key}_1h.png"
    c_path = out_dir / f"{cache_key}_13c.png"

    if not task["overwrite"] and h_path.exists() and c_path.exists():
        return {"id": sample_id, "status": "skipped"}

    h_image, c_image = render_sample_images(
        sample,
        h_snr=float(task["h_snr"]),
        c_snr=float(task["c_snr"]),
        render_seed=task["render_seed"],
        image_size=image_size,
    )
    h_image.save(h_path)
    c_image.save(c_path)
    return {"id": sample_id, "status": "rendered"}


def pre_render_images(
    dataset_dir: str | Path,
    out_dir: str | Path,
    *,
    splits: list[str],
    image_size: tuple[int, int] | None = (768, 432),
    h_snr: float = 500.0,
    c_snr: float = 300.0,
    render_seed: int | None = 3407,
    num_workers: int = 1,
    overwrite: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Pre-render image files for selected JSONL samples."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    split_ids = read_split_ids(dataset_dir, splits)
    selected_samples = iter_selected_samples(dataset_dir, split_ids)
    if limit is not None:
        selected_samples = _take(selected_samples, limit)

    tasks = (
        {
            "sample": sample,
            "out_dir": str(out_path),
            "image_size": list(image_size) if image_size else None,
            "h_snr": h_snr,
            "c_snr": c_snr,
            "render_seed": render_seed,
            "overwrite": overwrite,
        }
        for sample in selected_samples
    )

    rendered = 0
    skipped = 0
    total = min(len(split_ids), limit) if limit is not None else len(split_ids)

    for result in tqdm(
        _iter_render_results(tasks, num_workers=num_workers),
        total=total,
        desc="Pre-render NMR images",
    ):
        if result["status"] == "rendered":
            rendered += 1
        elif result["status"] == "skipped":
            skipped += 1

    summary = {
        "dataset_dir": str(dataset_dir),
        "out_dir": str(out_path),
        "splits": splits,
        "requested_ids": len(split_ids),
        "rendered": rendered,
        "skipped": skipped,
        "image_size": list(image_size) if image_size else [WIDTH_PX, HEIGHT_PX],
        "h_snr": h_snr,
        "c_snr": c_snr,
        "render_seed": render_seed,
    }
    (out_path / "render_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _iter_render_results(
    tasks: Iterable[dict[str, Any]],
    *,
    num_workers: int,
) -> Iterable[dict[str, Any]]:
    """Yield render results without submitting all samples at once."""
    if num_workers <= 1:
        for task in tasks:
            yield _render_one(task)
        return

    task_iter = iter(tasks)
    max_pending = max(1, num_workers * 4)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        pending = set()
        for _ in range(max_pending):
            try:
                pending.add(executor.submit(_render_one, next(task_iter)))
            except StopIteration:
                break

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()
                try:
                    pending.add(executor.submit(_render_one, next(task_iter)))
                except StopIteration:
                    pass


def _take(items: Iterable[dict[str, Any]], limit: int) -> Iterable[dict[str, Any]]:
    """Yield at most ``limit`` items."""
    for idx, item in enumerate(items):
        if idx >= limit:
            break
        yield item


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--image-size", nargs=2, type=int, default=[768, 432])
    parser.add_argument("--h-snr", type=float, default=500.0)
    parser.add_argument("--c-snr", type=float, default=300.0)
    parser.add_argument("--render-seed", type=int, default=3407)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    image_size = tuple(args.image_size) if args.image_size else None
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path(args.dataset_dir) / "rendered" / f"{image_size[0]}x{image_size[1]}"
    )
    summary = pre_render_images(
        args.dataset_dir,
        out_dir,
        splits=args.splits,
        image_size=image_size,
        h_snr=args.h_snr,
        c_snr=args.c_snr,
        render_seed=args.render_seed,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote pre-rendered images to {out_dir}")


if __name__ == "__main__":
    main()
