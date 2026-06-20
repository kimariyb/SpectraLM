"""Lazy JSONL dataset and spectrum-image loading for NMR structure tuning."""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Mapping, Sequence

import numpy as np
from PIL import Image as PILImage
from PIL.Image import Resampling
from torch.utils.data import Dataset

from src.data.tasks import (
    CANDIDATE_RANKING,
    STRUCTURE_PREDICTION,
    build_task_example,
    normalize_task_weights,
)
from src.evaluation.prompts import STRUCTURE_PROMPTS
from src.spectra.render import carbon_to_spectra, hydrogen_to_spectra


_OFFSET_CACHE_DIRNAME = ".offset_cache"
_OFFSET_CACHE_VERSION = 1


def _safe_cache_key(sample_id: str) -> str:
    """Sanitize a sample ID into a filesystem-safe cache key."""
    safe = "".join(c for c in str(sample_id) if c.isalnum() or c in "_-.")
    safe = safe.strip(". ")
    if not safe or len(safe) > 128:
        return hashlib.sha256(str(sample_id).encode("utf-8")).hexdigest()[:16]
    return safe


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically through a process-specific temporary file."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _stable_render_seed(
    base_seed: int | None,
    sample_id: str,
    nucleus: str,
) -> int | None:
    """Derive a deterministic per-sample render seed."""
    if base_seed is None:
        return None
    payload = f"{base_seed}:{sample_id}:{nucleus}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _jsonl_sample_iter(path: Path) -> Iterator[dict[str, Any]]:
    """Yield non-empty sample dictionaries from a JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                yield json.loads(text)


def load_candidate_map(path: str | Path) -> dict[str, list[str]]:
    """Load a candidate-ranking JSONL sidecar keyed by sample ID.

    Parameters
    ----------
    path
        Candidate sidecar path.

    Returns
    -------
    dict[str, list[str]]
        Candidate SMILES for each sample with formula-matched alternatives.
    """
    candidate_map: dict[str, list[str]] = {}
    for row in _jsonl_sample_iter(Path(path)):
        sample_id = str(row.get("id", ""))
        candidates = [str(value) for value in row.get("candidates", [])]
        if sample_id and candidates:
            candidate_map[sample_id] = candidates
    return candidate_map


def _split_ids_path(base: Path, split: str) -> Path:
    """Return the direct or curated ID-list path for a split."""
    split_key = {"validation": "val"}.get(split, split)
    direct = base / f"{split_key}_ids.txt"
    if direct.exists():
        return direct
    return base / "subsets" / f"{split_key}_ids.txt"


def _load_jsonl_samples(base: Path, split: str | None) -> list[dict[str, Any]]:
    """Load raw JSONL samples, optionally filtering by a named split."""
    jsonl_path = base / "samples.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL dataset not found: {jsonl_path}")
    if split is None:
        return list(_jsonl_sample_iter(jsonl_path))

    ids_path = _split_ids_path(base, split)
    if not ids_path.exists():
        raise FileNotFoundError(
            f"Split id file not found for split={split!r}: {ids_path}"
        )
    split_ids = {
        line.strip()
        for line in ids_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    return [
        sample
        for sample in _jsonl_sample_iter(jsonl_path)
        if str(sample.get("id", "")) in split_ids
    ]


class NMRMessageTransform:
    """Convert raw NMR samples and images into configurable SFT tasks."""

    def __init__(
        self,
        *,
        seed: int | None = 3407,
        include_formula: bool = True,
        include_rule_context: bool = False,
        max_rule_evidence: int = 12,
        task_weights: Mapping[str, float] | None = None,
        candidate_map: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.include_formula = bool(include_formula)
        self.include_rule_context = bool(include_rule_context)
        self.max_rule_evidence = int(max_rule_evidence)
        self.task_weights = normalize_task_weights(task_weights)
        self.task_names = tuple(self.task_weights)
        self.task_probabilities = tuple(self.task_weights.values())
        self.candidate_map = dict(candidate_map or {})

    def __call__(self, batch: dict[str, Any]) -> dict[str, list[Any]]:
        """Build image-plus-table prompts and task-specific targets."""
        h_images = self._as_list(batch["h_image"])
        c_images = self._as_list(batch["c_image"])
        samples = self._as_list(batch["sample"])
        messages_batch: list[list[dict[str, Any]]] = []

        for h_image, c_image, sample in zip(h_images, c_images, samples):
            task = str(
                self.rng.choice(
                    self.task_names,
                    p=self.task_probabilities,
                )
            )
            sample_id = str(sample.get("id", ""))
            candidates = self.candidate_map.get(sample_id)
            if task == CANDIDATE_RANKING and not candidates:
                task = STRUCTURE_PREDICTION
            example = build_task_example(
                sample,
                task,
                candidates=candidates,
                structure_prompt=str(self.rng.choice(STRUCTURE_PROMPTS)),
                include_formula=self.include_formula,
                include_rule_context=self.include_rule_context,
                max_rule_evidence=self.max_rule_evidence,
            )
            messages_batch.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": h_image},
                            {"type": "image", "image": c_image},
                            {"type": "text", "text": example.prompt},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": example.target}
                        ],
                    },
                ]
            )
        return {"messages": messages_batch}

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        """Wrap scalar transform inputs in a one-element list."""
        return value if isinstance(value, list) else [value]


def _resize_image(
    image: PILImage.Image,
    image_size: tuple[int, int] | list[int] | None,
) -> PILImage.Image:
    """Convert an image to RGB and resize only when dimensions differ."""
    image = image.convert("RGB")
    target_size = tuple(image_size) if image_size is not None else None
    if target_size is None or image.size == target_size:
        return image
    return image.resize(target_size, Resampling.LANCZOS)


def render_sample_images(
    sample: dict[str, Any],
    *,
    h_snr: float = 500.0,
    c_snr: float = 300.0,
    render_seed: int | None = 3407,
    image_size: tuple[int, int] | list[int] | None = None,
) -> tuple[PILImage.Image, PILImage.Image]:
    """Render proton and carbon spectrum images for one sample."""
    sample_id = str(sample.get("id", "unknown"))
    h_image = hydrogen_to_spectra(
        sample,
        snr=h_snr,
        seed=_stable_render_seed(render_seed, sample_id, "1h"),
    )
    c_image = carbon_to_spectra(
        sample,
        snr=c_snr,
        seed=_stable_render_seed(render_seed, sample_id, "13c"),
    )
    return (
        _resize_image(h_image, image_size),
        _resize_image(c_image, image_size),
    )


def load_sample_images(
    sample: dict[str, Any],
    *,
    image_backend: str = "lazy_render",
    rendered_image_dir: str | Path | None = None,
    missing_image_policy: str = "error",
    h_snr: float = 500.0,
    c_snr: float = 300.0,
    render_seed: int | None = 3407,
    image_size: tuple[int, int] | list[int] | None = None,
) -> tuple[PILImage.Image, PILImage.Image]:
    """Load pre-rendered spectra or render them on demand."""
    if image_backend not in {"lazy_render", "pre_rendered"}:
        raise ValueError(
            "image_backend must be 'lazy_render' or 'pre_rendered', "
            f"got {image_backend!r}"
        )
    if missing_image_policy not in {"error", "lazy_render"}:
        raise ValueError(
            "missing_image_policy must be 'error' or 'lazy_render', "
            f"got {missing_image_policy!r}"
        )

    def render() -> tuple[PILImage.Image, PILImage.Image]:
        return render_sample_images(
            sample,
            h_snr=h_snr,
            c_snr=c_snr,
            render_seed=render_seed,
            image_size=image_size,
        )

    if image_backend == "lazy_render":
        return render()
    if rendered_image_dir is None:
        raise ValueError(
            "rendered_image_dir is required when image_backend='pre_rendered'."
        )

    cache_key = _safe_cache_key(str(sample.get("id", "unknown")))
    rendered_dir = Path(rendered_image_dir)
    h_path = rendered_dir / f"{cache_key}_1h.png"
    c_path = rendered_dir / f"{cache_key}_13c.png"
    missing = [str(path) for path in (h_path, c_path) if not path.exists()]
    if missing:
        if missing_image_policy == "lazy_render":
            return render()
        raise FileNotFoundError(
            "Missing pre-rendered NMR image(s): " + ", ".join(missing)
        )

    with PILImage.open(h_path) as h_image:
        h_loaded = _resize_image(h_image, image_size)
    with PILImage.open(c_path) as c_image:
        c_loaded = _resize_image(c_image, image_size)
    return h_loaded, c_loaded


class LazyNMRJsonlDataset(Dataset):
    """Offset-indexed JSONL dataset for million-scale VLM training."""

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        split: str = "train",
        include_formula: bool = True,
        include_rule_context: bool = False,
        max_rule_evidence: int = 12,
        task_weights: Mapping[str, float] | None = None,
        candidate_sidecar_path: str | Path | None = None,
        seed: int | None = 3407,
        h_snr: float = 500.0,
        c_snr: float = 300.0,
        render_seed: int | None = 3407,
        image_size: tuple[int, int] | list[int] | None = None,
        image_backend: str = "lazy_render",
        rendered_image_dir: str | Path | None = None,
        missing_image_policy: str = "error",
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.jsonl_path = self.dataset_dir / "samples.jsonl"
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"JSONL dataset not found: {self.jsonl_path}")
        if image_backend not in {"lazy_render", "pre_rendered"}:
            raise ValueError(f"Unsupported image_backend: {image_backend!r}")
        if missing_image_policy not in {"error", "lazy_render"}:
            raise ValueError(
                f"Unsupported missing_image_policy: {missing_image_policy!r}"
            )
        if image_backend == "pre_rendered" and rendered_image_dir is None:
            raise ValueError(
                "rendered_image_dir is required when image_backend='pre_rendered'."
            )

        self.split = split
        self.h_snr = float(h_snr)
        self.c_snr = float(c_snr)
        self.render_seed = render_seed
        self.image_size = tuple(image_size) if image_size is not None else None
        self.image_backend = image_backend
        self.rendered_image_dir = (
            Path(rendered_image_dir) if rendered_image_dir is not None else None
        )
        self.missing_image_policy = missing_image_policy
        self._jsonl_handle: BinaryIO | None = None
        self._jsonl_handle_pid: int | None = None
        self.transform = NMRMessageTransform(
            seed=seed,
            include_formula=include_formula,
            include_rule_context=include_rule_context,
            max_rule_evidence=max_rule_evidence,
            task_weights=task_weights,
            candidate_map=(
                load_candidate_map(candidate_sidecar_path)
                if candidate_sidecar_path is not None
                else None
            ),
        )
        self.offsets = self._index_offsets()

    def _index_offsets(self) -> np.ndarray:
        """Load valid cached offsets or scan and cache the JSONL file."""
        split_ids_path = _split_ids_path(self.dataset_dir, self.split)
        if not split_ids_path.exists():
            split_ids_path = None

        manifest = self._offset_cache_manifest(split_ids_path)
        cached = self._load_offset_cache(manifest)
        if cached is not None:
            return cached

        split_ids = self._read_split_ids(split_ids_path)
        offsets = self._scan_offsets(split_ids)
        self._write_offset_cache(offsets, manifest)
        return offsets

    @staticmethod
    def _read_split_ids(split_ids_path: Path | None) -> set[str] | None:
        """Read split IDs only after an offset-cache miss."""
        if split_ids_path is None:
            return None
        return {
            line.strip()
            for line in split_ids_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    def _scan_offsets(self, split_ids: set[str] | None) -> np.ndarray:
        """Scan JSONL byte positions for rows in the requested split."""
        offsets: list[int] = []
        with self.jsonl_path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if split_ids is None:
                    offsets.append(offset)
                    continue
                sample = json.loads(line)
                if str(sample.get("id", "")) in split_ids:
                    offsets.append(offset)
        return np.asarray(offsets, dtype=np.int64)

    @staticmethod
    def _file_signature(path: Path) -> dict[str, Any]:
        """Return metadata used to invalidate offset caches."""
        stat = path.stat()
        return {
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    def _offset_cache_manifest(
        self,
        split_ids_path: Path | None,
    ) -> dict[str, Any]:
        """Build the current source signature for this split."""
        return {
            "version": _OFFSET_CACHE_VERSION,
            "split": self.split,
            "jsonl": self._file_signature(self.jsonl_path),
            "split_ids": (
                self._file_signature(split_ids_path)
                if split_ids_path is not None
                else None
            ),
        }

    def _offset_cache_paths(self) -> tuple[Path, Path]:
        """Return NumPy offsets and JSON manifest cache paths."""
        cache_dir = self.dataset_dir / _OFFSET_CACHE_DIRNAME
        cache_key = _safe_cache_key(self.split)
        return cache_dir / f"{cache_key}.npy", cache_dir / f"{cache_key}.json"

    def _load_offset_cache(
        self,
        expected_manifest: dict[str, Any],
    ) -> np.ndarray | None:
        """Memory-map offsets when the source signature matches."""
        offsets_path, manifest_path = self._offset_cache_paths()
        if not offsets_path.exists() or not manifest_path.exists():
            return None
        try:
            stored_manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            if stored_manifest != expected_manifest:
                return None
            offsets = np.load(offsets_path, allow_pickle=False, mmap_mode="r")
            if offsets.ndim != 1 or offsets.dtype != np.int64:
                return None
            return offsets
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    def _write_offset_cache(
        self,
        offsets: np.ndarray,
        manifest: dict[str, Any],
    ) -> None:
        """Persist offsets atomically, falling back on read-only datasets."""
        offsets_path, manifest_path = self._offset_cache_paths()
        try:
            offsets_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = offsets_path.with_name(
                f".{offsets_path.name}.{os.getpid()}.tmp"
            )
            with tmp_path.open("wb") as handle:
                np.save(handle, offsets, allow_pickle=False)
            tmp_path.replace(offsets_path)
            _atomic_write_text(
                manifest_path,
                json.dumps(manifest, ensure_ascii=False, indent=2),
            )
        except OSError as exc:
            warnings.warn(
                f"Could not write JSONL offset cache under "
                f"{offsets_path.parent}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    def __len__(self) -> int:
        """Return the number of samples in this split."""
        return len(self.offsets)

    def _get_jsonl_handle(self) -> BinaryIO:
        """Return one reusable JSONL file handle per worker process."""
        process_id = os.getpid()
        if (
            self._jsonl_handle is None
            or self._jsonl_handle.closed
            or self._jsonl_handle_pid != process_id
        ):
            self.close()
            self._jsonl_handle = self.jsonl_path.open("rb")
            self._jsonl_handle_pid = process_id
        return self._jsonl_handle

    def _load_sample_at(self, offset: int) -> dict[str, Any]:
        """Read one JSONL sample by byte offset."""
        handle = self._get_jsonl_handle()
        handle.seek(int(offset))
        return json.loads(handle.readline())

    def close(self) -> None:
        """Close the current process's reusable JSONL handle."""
        if self._jsonl_handle is not None and not self._jsonl_handle.closed:
            self._jsonl_handle.close()
        self._jsonl_handle = None
        self._jsonl_handle_pid = None

    def __getstate__(self) -> dict[str, Any]:
        """Exclude open handles when spawning DataLoader workers."""
        state = self.__dict__.copy()
        state["_jsonl_handle"] = None
        state["_jsonl_handle_pid"] = None
        return state

    def __del__(self) -> None:
        """Release the JSONL handle when the dataset is collected."""
        try:
            self.close()
        except Exception:
            pass

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return one multimodal NMR supervision instruction."""
        sample = self._load_sample_at(self.offsets[idx])
        h_image, c_image = load_sample_images(
            sample,
            image_backend=self.image_backend,
            rendered_image_dir=self.rendered_image_dir,
            missing_image_policy=self.missing_image_policy,
            h_snr=self.h_snr,
            c_snr=self.c_snr,
            render_seed=self.render_seed,
            image_size=self.image_size,
        )
        transformed = self.transform(
            {
                "h_image": [h_image],
                "c_image": [c_image],
                "sample": [sample],
            }
        )
        return {"messages": transformed["messages"][0]}


def load_lazy_nmr_dataset(
    dataset_dir: str | Path,
    *,
    split: str = "train",
    include_formula: bool = True,
    include_rule_context: bool = False,
    max_rule_evidence: int = 12,
    task_weights: Mapping[str, float] | None = None,
    candidate_sidecar_path: str | Path | None = None,
    seed: int | None = 3407,
    h_snr: float = 500.0,
    c_snr: float = 300.0,
    render_seed: int | None = 3407,
    image_size: tuple[int, int] | list[int] | None = None,
    image_backend: str = "lazy_render",
    rendered_image_dir: str | Path | None = None,
    missing_image_policy: str = "error",
) -> LazyNMRJsonlDataset:
    """Create the active lazy JSONL training dataset."""
    return LazyNMRJsonlDataset(
        dataset_dir,
        split=split,
        include_formula=include_formula,
        include_rule_context=include_rule_context,
        max_rule_evidence=max_rule_evidence,
        task_weights=task_weights,
        candidate_sidecar_path=candidate_sidecar_path,
        seed=seed,
        h_snr=h_snr,
        c_snr=c_snr,
        render_seed=render_seed,
        image_size=image_size,
        image_backend=image_backend,
        rendered_image_dir=rendered_image_dir,
        missing_image_policy=missing_image_policy,
    )


def load_raw_nmr_samples(
    dataset_dir: str | Path,
    *,
    split: str | None = None,
) -> list[dict[str, Any]]:
    """Load raw JSONL samples for bounded validation or inference splits."""
    return _load_jsonl_samples(Path(dataset_dir), split)
