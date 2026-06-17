"""HuggingFace ``datasets``-backed multimodal NMR-to-structure dataset.

Each sample provides separate 1H and 13C spectrum images stored as Arrow
binary columns (decoded to PIL on access).  Task, prompt and target are
generated dynamically in ``__getitem__`` so they vary across epochs.

Features
--------
- **Smart loading**: pre-split ``train.pkl`` / ``test.pkl`` files are
  loaded directly when they exist, skipping the scaffold-split step.
- **Disk cache with invalidation**: rendered PNGs are cached to disk with
  a manifest that tracks SNR parameters; stale caches are automatically
  cleared when rendering parameters change.
- **Zero serialization overhead**: sample metadata is accessed directly
  from an in-memory list — no JSON dump/load per access.
- **NumPy-backed randomness**: uses ``numpy.random.Generator`` (PCG64)
  for reproducible, statistically robust task and SNR sampling.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any, Generator

import numpy as np
from datasets import Dataset, Image as HFImage
from PIL import Image as PILImage

from src.io import load_pickle_list
from src.spectra.render import carbon_to_spectra, hydrogen_to_spectra
from src.data.molecules import sample_smiles, sample_fg
from src.training.prompts import (
    FUNCTIONAL_GROUP_PROMPTS,
    STRUCTURE_PROMPTS,
    build_structure_prompt,
)

# Constants
_CACHE_MANIFEST_FILENAME = ".cache_manifest.json"

_DEFAULT_TASK_PROBS: dict[str, float] = {
    "structure": 0.8,
    "functional_group": 0.2,
}

# Atomic I/O helpers
def _safe_cache_key(sample_id: str) -> str:
    """Sanitize a sample ID into a filesystem-safe cache key.

    Path-traversal characters are stripped so the key can be safely
    joined with a cache directory.  If the resulting string is empty
    or excessively long, a hex digest is used instead.

    Parameters
    ----------
    sample_id
        Raw sample identifier from a data file.

    Returns
    -------
    str
        Filesystem-safe cache key.
    """
    safe = "".join(c for c in sample_id if c.isalnum() or c in "_-.")
    safe = safe.strip(". ") or hashlib.sha256(sample_id.encode()).hexdigest()[:16]
    
    if len(safe) > 128:
        safe = hashlib.sha256(sample_id.encode()).hexdigest()[:16]
        
    return safe


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write *data* to *path* atomically via a temporary file rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.rename(path)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write *text* to *path* atomically via a temporary file rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.rename(path)


def _resolve_and_load_samples(
    dataset_path: str | Path,
    split: str | None,
    train_size: int,
) -> list[dict[str, Any]]:
    """Load samples with smart-loading priority.

    Priority chain when *split* is given:

    1. ``<dataset_path>/<split>.pkl`` — pre-split file, loaded directly
       (fast path; no scaffold splitting needed).
    2. ``<dataset_path>/selected.pkl`` — full set, split by scaffold.

    When *split* is ``None``, *dataset_path* is treated as a direct file
    path.

    Parameters
    ----------
    dataset_path
        Path to a ``.pkl`` file, or a directory containing split files.
    split
        ``"train"``, ``"test"``, or ``None``.
    train_size
        Number of training samples used when splitting
        ``selected.pkl`` (ignored on the fast path).

    Returns
    -------
    list[dict[str, Any]]
        Loaded sample dictionaries.

    Raises
    ------
    FileNotFoundError
        If no matching dataset file can be located.
    """
    base = Path(dataset_path)

    if split is None:
        if not base.exists():
            raise FileNotFoundError(f"Dataset file not found: {base}")
        return load_pickle_list(str(base))

    # Priority 1: pre-split file already exists → instant load
    split_file = base / f"{split}.pkl"
    if split_file.exists():
        return load_pickle_list(str(split_file))

    # Priority 2: selected.pkl → scaffold split
    selected_file = base / "selected.pkl"
    if selected_file.exists():
        all_samples = load_pickle_list(str(selected_file))
        return _split_by_scaffold(all_samples, train_size, split)

    raise FileNotFoundError(
        f"No dataset file found in '{base}' for split='{split}'. "
        f"Expected either '{split_file}' or '{selected_file}'."
    )


def _split_by_scaffold(
    samples: list[dict[str, Any]],
    train_size: int,
    split: str,
) -> list[dict[str, Any]]:
    """Split samples into train / test by scaffold priority.

    Train gets each scaffold's first occurrence.  Remaining samples
    from the same scaffold go to test.

    Parameters
    ----------
    samples
        All selected samples.
    train_size
        Number of training samples.
    split
        ``"train"`` or ``"test"``.

    Returns
    -------
    list[dict[str, Any]]
        Samples for the requested split.
    """
    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    train_scaffolds: set[str] = set()

    for sample in samples:
        scaffold = sample.get("murcko_scaffold", "")
        if scaffold not in train_scaffolds and len(train) < train_size:
            train.append(sample)
            train_scaffolds.add(scaffold)
        else:
            test.append(sample)

    return train if split == "train" else test


class NmrReasoningDataset:
    """Multitask NMR dataset backed by :class:`datasets.Dataset`.

    Images are pre-rendered once at construction time and stored as Arrow
    binary columns.  Task selection, prompt construction and target
    generation happen lazily in :meth:`__getitem__`, giving different
    task mixes each epoch.

    Parameters
    ----------
    dataset_path
        Path to a ``.pkl`` file, or a directory containing
        ``train.pkl`` / ``test.pkl`` / ``selected.pkl``.
    split
        Which split to load (``"train"`` or ``"test"``).  When set and a
        pre-split ``<split>.pkl`` exists in *dataset_path*, it is loaded
        directly.  Otherwise ``selected.pkl`` is split by scaffold.
    train_size
        Number of training samples when splitting ``selected.pkl``
        (default 1000).  Ignored when a pre-split file is loaded.
    task_probs
        Task sampling probabilities (default: 80 % structure,
        20 % functional group).
    cache_dir
        Optional directory for rendered-spectrum PNG cache.  When set,
        images use a fixed SNR and are saved to disk; subsequent
        instantiations skip rendering.  A manifest file tracks the
        rendering parameters so the cache is automatically invalidated
        when parameters change.
    h_snr
        1H signal-to-noise ratio (only used with *cache_dir*).
    c_snr
        13C signal-to-noise ratio (only used with *cache_dir*).
    cache_version
        Optional version tag written to the cache manifest.  Bump this
        to force cache invalidation after changing rendering code
        without changing SNR parameters.
    seed
        Optional seed for the NumPy random generator.  When ``None``
        (default), system entropy is used.  Set to an integer for
        reproducible task sampling and SNR randomization.
    """

    def __init__(
        self,
        dataset_path: str,
        split: str | None = None,
        train_size: int = 1000,
        task_probs: dict[str, float] | None = None,
        cache_dir: str | None = None,
        h_snr: float = 500.0,
        c_snr: float = 300.0,
        cache_version: str | None = None,
        seed: int | None = None,
    ) -> None:
        # -- sample loading --------------------------------------------------
        self._samples = _resolve_and_load_samples(dataset_path, split, train_size)

        # -- task configuration ----------------------------------------------
        tp = task_probs or _DEFAULT_TASK_PROBS
        self._tasks = list(tp.keys())
        self._task_weights = np.asarray(list(tp.values()), dtype=np.float64)

        # -- random generator ------------------------------------------------
        self._rng = np.random.default_rng(seed)

        # -- cache configuration ---------------------------------------------
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._h_snr = h_snr
        self._c_snr = c_snr
        self._cache_version = cache_version
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._validate_cache()

        # -- build Arrow dataset (images only) -------------------------------
        self.ds = self._build_arrow_dataset()

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return one multimodal chat-style training example.

        Task, prompt and target are sampled on every access so they
        vary across training epochs.  Sample metadata is read directly
        from the in-memory list — no JSON deserialization.
        """
        row = self.ds[idx]
        sample = self._samples[idx]

        task = self._rng.choice(self._tasks, p=self._task_weights)

        if task == "structure":
            prompt = build_structure_prompt(
                sample,
                prompt=self._rng.choice(STRUCTURE_PROMPTS),
            )
            target = sample_smiles(sample)
        else:
            prompt = self._rng.choice(FUNCTIONAL_GROUP_PROMPTS).format(
                peak_tables="(see spectra above)",
            )
            groups = sample_fg(sample)
            target = ", ".join(groups) if groups else "Unknown"

        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": row["h_image"]},
                        {"type": "image", "image": row["c_image"]},
                        {"type": "text", "text": prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": target}],
                },
            ]
        }

    # Arrow dataset construction
    def _build_arrow_dataset(self) -> Dataset:
        """Build a :class:`Dataset` with pre-rendered spectrum images.

        Only ``id``, ``h_image`` and ``c_image`` columns are stored.
        Sample metadata lives in :attr:`_samples` for zero-copy access.
        """
        ds = Dataset.from_generator(self._iter_samples)
        ds = ds.cast_column("h_image", HFImage())
        ds = ds.cast_column("c_image", HFImage())
        return ds

    def _iter_samples(self) -> Generator[dict[str, Any], None, None]:
        """Yield Arrow-compatible dicts with pre-rendered image bytes."""
        for sample in self._samples:
            yield {
                "id": sample.get("id", ""),
                "h_image": self._render(sample, "1h"),
                "c_image": self._render(sample, "13c"),
            }

    def _render(self, sample: dict[str, Any], nucleus: str) -> bytes:
        """Render one NMR spectrum image, returning raw PNG bytes.

        When *cache_dir* is configured, rendered PNGs are saved to disk
        and reused on subsequent instantiations.  Cache hits read raw
        bytes directly, avoiding a PIL decode/encode round-trip.

        Parameters
        ----------
        sample
            Sample dictionary with at least an ``"id"`` key.
        nucleus
            ``"1h"`` or ``"13c"``.

        Returns
        -------
        bytes
            PNG-encoded image bytes.
        """
        cache_key = _safe_cache_key(sample.get("id", "unknown"))

        # --- cache hit: read raw bytes directly -------------------------
        if self._cache_dir:
            cache_path = self._cache_dir / f"{cache_key}_{nucleus}.png"
            if cache_path.exists():
                return cache_path.read_bytes()

        # --- render fresh -----------------------------------------------
        if nucleus == "1h":
            snr = self._h_snr if self._cache_dir else self._rng.uniform(300.0, 600.0)
            image = hydrogen_to_spectra(sample, snr=snr)
        else:
            snr = self._c_snr if self._cache_dir else self._rng.uniform(100.0, 400.0)
            image = carbon_to_spectra(sample, snr=snr)

        if not isinstance(image, PILImage.Image):
            image = PILImage.fromarray(image)
        image = image.convert("RGB")

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        # --- persist to cache (atomic write via tempfile) --------------
        if self._cache_dir:
            cache_path = self._cache_dir / f"{cache_key}_{nucleus}.png"
            _atomic_write_bytes(cache_path, png_bytes)

        return png_bytes

    def _validate_cache(self) -> None:
        """Validate or invalidate the disk cache via a JSON manifest.

        The manifest records the rendering parameters that produced the
        cached PNGs.  If the current parameters differ from the stored
        manifest (or the manifest is missing / corrupt), all cached PNGs
        are deleted so they will be re-rendered.

        This prevents silent use of stale spectra after changing SNR
        values, rendering code, or bumping *cache_version*.
        """
        assert self._cache_dir is not None  # only called when cache_dir is set

        manifest_path = self._cache_dir / _CACHE_MANIFEST_FILENAME
        current_manifest = {
            "h_snr": self._h_snr,
            "c_snr": self._c_snr,
            "version": self._cache_version or "1",
        }

        if manifest_path.exists():
            try:
                stored = json.loads(manifest_path.read_text())
                if stored == current_manifest:
                    return  # cache is valid — nothing to do
            except json.JSONDecodeError:
                pass  # corrupt manifest → invalidate

        # Manifest missing, stale, or corrupt — clear all cached PNGs
        for png_file in self._cache_dir.glob("*.png"):
            png_file.unlink()

        _atomic_write_text(
            manifest_path,
            json.dumps(current_manifest, ensure_ascii=False, indent=2),
        )
