"""Text-only lazy JSONL dataset for NMR structure tuning."""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Mapping, Sequence

import numpy as np
from torch.utils.data import Dataset

from src.data.tasks import (
    CANDIDATE_RANKING,
    STRUCTURE_PREDICTION,
    build_task_example,
    normalize_task_weights,
    select_weighted_task,
)
from src.evaluation.prompts import (
    SYSTEM_PROMPT,
    select_structure_prompt,
    structure_prompts,
)


_OFFSET_CACHE_DIRNAME = ".offset_cache"
_OFFSET_CACHE_VERSION = 2


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


def _jsonl_sample_iter(path: Path) -> Iterator[dict[str, Any]]:
    """Yield non-empty sample dictionaries from a JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                yield json.loads(text)


def load_candidate_map(path: str | Path) -> dict[str, list[str]]:
    """Load a candidate-ranking JSONL sidecar keyed by sample ID."""
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
    """Convert raw NMR samples into text-only SFT chat messages."""

    def __init__(
        self,
        *,
        seed: int | None = 3407,
        include_formula: bool = True,
        include_rule_context: bool = False,
        max_rule_evidence: int = 12,
        task_weights: Mapping[str, float] | None = None,
        candidate_map: Mapping[str, Sequence[str]] | None = None,
        prompt_template_index: int | None = None,
        target_stereochemistry: str = "preserve",
    ) -> None:
        self.seed = int(seed or 0)
        self.rng = np.random.default_rng(seed)
        self.include_formula = bool(include_formula)
        self.include_rule_context = bool(include_rule_context)
        self.max_rule_evidence = int(max_rule_evidence)
        self.task_weights = normalize_task_weights(task_weights)
        self.target_stereochemistry = target_stereochemistry
        self.prompt_template_index = (
            None
            if prompt_template_index is None
            else int(prompt_template_index)
        )
        if self.prompt_template_index is not None:
            select_structure_prompt(self.prompt_template_index)
        self.task_names = tuple(self.task_weights)
        self.candidate_map = dict(candidate_map or {})

    def __call__(self, batch: dict[str, Any]) -> dict[str, list[Any]]:
        """Build text prompts and task-specific targets."""
        samples = self._as_list(batch["sample"])
        messages_batch: list[list[dict[str, str]]] = []

        for sample in samples:
            task = select_weighted_task(
                str(sample.get("id", "")),
                seed=self.seed,
                weights=self.task_weights,
            )
            sample_id = str(sample.get("id", ""))
            candidates = self.candidate_map.get(sample_id)
            if task == CANDIDATE_RANKING and not candidates:
                task = STRUCTURE_PREDICTION
            if self.prompt_template_index is None:
                prompt = str(self.rng.choice(structure_prompts()))
            else:
                prompt = select_structure_prompt(self.prompt_template_index)
            example = build_task_example(
                sample,
                task,
                candidates=candidates,
                structure_prompt=prompt,
                include_formula=self.include_formula,
                include_rule_context=self.include_rule_context,
                max_rule_evidence=self.max_rule_evidence,
                target_stereochemistry=self.target_stereochemistry,
            )
            messages_batch.append(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": example.prompt},
                    {"role": "assistant", "content": example.target},
                ]
            )
        return {"messages": messages_batch}

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        """Wrap scalar transform inputs in a one-element list."""
        return value if isinstance(value, list) else [value]


class LazyNMRJsonlDataset(Dataset):
    """Offset-indexed JSONL dataset for million-scale text SFT."""

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
        prompt_template_index: int | None = None,
        target_stereochemistry: str = "preserve",
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.jsonl_path = self.dataset_dir / "samples.jsonl"
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"JSONL dataset not found: {self.jsonl_path}")

        self.split = split
        self._jsonl_handle: BinaryIO | None = None
        self._jsonl_handle_pid: int | None = None
        self.transform = NMRMessageTransform(
            seed=seed,
            include_formula=include_formula,
            include_rule_context=include_rule_context,
            max_rule_evidence=max_rule_evidence,
            task_weights=task_weights,
            prompt_template_index=prompt_template_index,
            target_stereochemistry=target_stereochemistry,
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
        """Return one text-only NMR supervision instruction."""
        sample = self._load_sample_at(self.offsets[idx])
        transformed = self.transform({"sample": [sample]})
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
    prompt_template_index: int | None = None,
    target_stereochemistry: str = "preserve",
) -> LazyNMRJsonlDataset:
    """Create the active lazy JSONL text training dataset."""
    return LazyNMRJsonlDataset(
        dataset_dir,
        split=split,
        include_formula=include_formula,
        include_rule_context=include_rule_context,
        max_rule_evidence=max_rule_evidence,
        task_weights=task_weights,
        candidate_sidecar_path=candidate_sidecar_path,
        seed=seed,
        prompt_template_index=prompt_template_index,
        target_stereochemistry=target_stereochemistry,
    )


def load_raw_nmr_samples(
    dataset_dir: str | Path,
    *,
    split: str | None = None,
) -> list[dict[str, Any]]:
    """Load raw JSONL samples for bounded validation or inference splits."""
    return _load_jsonl_samples(Path(dataset_dir), split)
