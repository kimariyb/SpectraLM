"""HuggingFace ``datasets``-backed multimodal NMR instruction dataset.

Architecture
------------
1. :class:`NMRDatasetBuilder` extends :class:`datasets.GeneratorBasedBuilder`;
2. Raw sample dicts, 1H images, and 13C images are stored in Arrow columns;
3. Prompts, tasks, and targets are generated dynamically via
   :meth:`Dataset.with_transform` using :class:`NMRMessageTransform`;
4. The resulting :class:`~datasets.Dataset` supports all standard HF
   operations: ``shuffle``, ``select``, ``filter``, ``map``,
   ``save_to_disk``, etc.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
from pathlib import Path
from typing import Any

import datasets
import numpy as np
from datasets import (
    BuilderConfig,
    DatasetInfo,
    Features,
    GeneratorBasedBuilder,
    Sequence,
    Value,
)
from PIL import Image as PILImage
from PIL.Image import Resampling
from torch.utils.data import Dataset

from src.data.molecules import sample_smiles, sample_fg
from src.evaluation.prompts import (
    FUNCTIONAL_GROUP_PROMPTS,
    STRUCTURE_PROMPTS,
    build_reasoning_target,
    build_structure_prompt,
)
from src.io import load_pickle_list
from src.spectra.render import carbon_to_spectra, hydrogen_to_spectra


# Constants
_CACHE_MANIFEST_FILENAME = ".cache_manifest.json"

_DEFAULT_TASK_PROBS: dict[str, float] = {
    "structure": 0.8,
    "functional_group": 0.2,
}


def _safe_cache_key(sample_id: str) -> str:
    """Sanitise a sample ID into a filesystem-safe cache key.

    Path-traversal characters are stripped.  Overly long or empty keys
    are replaced with a SHA-256 hex digest.
    """
    safe = "".join(c for c in str(sample_id) if c.isalnum() or c in "_-.")
    safe = safe.strip(". ")
    if not safe or len(safe) > 128:
        safe = hashlib.sha256(str(sample_id).encode("utf-8")).hexdigest()[:16]
    return safe


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write *data* to *path* atomically via a temporary-file rename."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write *text* to *path* atomically via a temporary-file rename."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _split_by_scaffold(
    samples: list[dict[str, Any]],
    train_size: float,
    split: str,
) -> list[dict[str, Any]]:
    """Split samples into train / test by Murcko-scaffold priority.

    Train receives each scaffold's first occurrence up to
    ``int(len(samples) * train_size)`` samples; remaining samples
    from the same scaffold go to test.

    Parameters
    ----------
    samples
        All selected samples.
    train_size
        Fraction of samples to allocate to training (0.0 – 1.0).
    split
        ``"train"`` or ``"test"``.

    Returns
    -------
    list[dict[str, Any]]
        Samples for the requested split.
    """
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    if train_size <= 0:
        return [] if split == "train" else list(samples)
    if train_size >= 1:
        return list(samples) if split == "train" else []

    max_train = int(len(samples) * train_size)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for idx, sample in enumerate(samples):
        scaffold = (
            sample.get("murcko_scaffold")
            or sample.get("canonical_smiles")
            or sample.get("id")
            or f"row:{idx}"
        )
        grouped.setdefault(str(scaffold), []).append(sample)

    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []

    for group in grouped.values():
        if not train or len(train) + len(group) <= max_train:
            train.extend(group)
        else:
            test.extend(group)

    return train if split == "train" else test


def _stable_render_seed(
    base_seed: int | None,
    sample_id: str,
    nucleus: str,
) -> int | None:
    """Derive a stable per-sample render seed.

    Parameters
    ----------
    base_seed
        Base render seed. ``None`` keeps stochastic rendering.
    sample_id
        Sample identifier.
    nucleus
        Nucleus label such as ``"1h"`` or ``"13c"``.

    Returns
    -------
    int | None
        Deterministic NumPy-compatible seed, or ``None``.
    """
    if base_seed is None:
        return None
    payload = f"{base_seed}:{sample_id}:{nucleus}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _jsonl_sample_iter(path: Path):
    """Yield sample dictionaries from a JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                yield json.loads(text)


def _split_ids_path(base: Path, split: str) -> Path:
    """Return the id-list path for a requested split name."""
    aliases = {"validation": "val"}
    split_key = aliases.get(split, split)
    direct = base / f"{split_key}_ids.txt"
    if direct.exists():
        return direct
    return base / "subsets" / f"{split_key}_ids.txt"


def _load_jsonl_samples(
    base: Path,
    split: str | None,
) -> list[dict[str, Any]]:
    """Load samples from ``samples.jsonl`` with optional split-id filtering."""
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


def _resolve_and_load_samples(
    dataset_path: str | Path,
    split: str | None,
    train_size: float,
) -> list[dict[str, Any]]:
    """Load samples with smart-loading priority.

    Priority when *split* is given:

    1. ``<dataset_path>/samples.jsonl`` + ``<split>_ids.txt``.
    2. ``<dataset_path>/<split>.pkl`` — pre-split, loaded directly.
    3. ``<dataset_path>/selected.pkl`` — split by scaffold.

    When *split* is ``None``, file paths are treated as full datasets.
    """
    base = Path(dataset_path)

    if split is None:
        if not base.exists():
            raise FileNotFoundError(f"Dataset file not found: {base}")
        if base.is_dir() and (base / "samples.jsonl").exists():
            return _load_jsonl_samples(base, split=None)
        if base.suffix == ".jsonl":
            return list(_jsonl_sample_iter(base))
        return load_pickle_list(str(base))

    if base.is_dir() and (base / "samples.jsonl").exists():
        return _load_jsonl_samples(base, split=split)

    split_file = base / f"{split}.pkl"
    if split_file.exists():
        return load_pickle_list(str(split_file))

    if split == "validation":
        val_file = base / "val.pkl"
        if val_file.exists():
            return load_pickle_list(str(val_file))

    selected_file = base / "selected.pkl"
    if selected_file.exists():
        all_samples = load_pickle_list(str(selected_file))
        return _split_by_scaffold(all_samples, train_size, split)

    raise FileNotFoundError(
        f"No dataset file found for split={split!r}. "
        f"Expected {split_file} or {selected_file}."
    )


def _normalise_task_probs(
    task_probs: dict[str, float] | None,
) -> tuple[list[str], np.ndarray]:
    """Validate and normalise task sampling probabilities.

    Returns task names and a normalised weight array (sum = 1).
    """
    probs = task_probs or _DEFAULT_TASK_PROBS

    if not probs:
        raise ValueError("task_probs must not be empty.")

    tasks = list(probs.keys())
    weights = np.asarray(list(probs.values()), dtype=np.float64)

    if np.any(weights < 0):
        raise ValueError(f"task_probs must be non-negative, got {probs}")

    total = float(weights.sum())
    if total <= 0:
        raise ValueError(f"Sum of task_probs must be positive, got {probs}")

    return tasks, weights / total


class NMRDatasetConfig(BuilderConfig):
    """Configuration dataclass for :class:`NMRDatasetBuilder`.

    Parameters
    ----------
    dataset_path
        Path to a ``.pkl`` file or a directory with split files.
    train_size
        Fraction of samples for the training split (0.0 – 1.0).
    render_cache_dir
        Optional directory for rendered PNG caching.
    h_snr
        1H signal-to-noise ratio (used when *render_cache_dir* is set).
    c_snr
        13C signal-to-noise ratio.
    render_cache_version
        Version tag embedded in the cache manifest — bump to invalidate.
    render_seed
        Base seed for deterministic per-sample rendering. ``None`` keeps
        stochastic rendering.
    image_size
        Optional ``(width, height)`` to resize rendered images.
    """

    def __init__(
        self,
        *,
        dataset_path: str | None = None,
        train_size: float = 0.8,
        render_cache_dir: str | None = None,
        h_snr: float = 500.0,
        c_snr: float = 300.0,
        render_cache_version: str = "1",
        render_seed: int | None = 3407,
        image_size: tuple[int, int] | list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.dataset_path = dataset_path
        self.train_size = float(train_size)
        self.render_cache_dir = render_cache_dir
        self.h_snr = float(h_snr)
        self.c_snr = float(c_snr)
        self.render_cache_version = str(render_cache_version)
        self.render_seed = int(render_seed) if render_seed is not None else None
        self.image_size = (
            tuple(image_size) if image_size is not None else None
        )


class NMRDatasetBuilder(GeneratorBasedBuilder):
    """HuggingFace :class:`~datasets.GeneratorBasedBuilder` for NMR spectra.

    Converts pickle samples into an Arrow :class:`~datasets.Dataset` with
    pre-rendered spectrum images.  Supports on-disk PNG caching with
    automatic invalidation when rendering parameters change.
    """

    BUILDER_CONFIG_CLASS = NMRDatasetConfig
    DEFAULT_CONFIG_NAME = "default"

    BUILDER_CONFIGS = [
        NMRDatasetConfig(
            name="default",
            version=datasets.Version("1.0.0"),
            description="NMR spectra → molecular structure instruction dataset.",
        )
    ]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        if self.config.render_cache_dir is not None:
            cache_dir = Path(self.config.render_cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._validate_render_cache(cache_dir)

    def _info(self) -> DatasetInfo:
        """Return the Arrow schema for the generated dataset."""
        return DatasetInfo(
            description="Multimodal NMR spectra instruction dataset.",
            features=Features(
                {
                    "id": Value("string"),
                    # Pickled sample dict — keeps metadata & images together
                    # after shuffle / select / filter.
                    "sample_pickle": Value("binary"),
                    # Redundant columns for convenient filtering / statistics.
                    "smiles": Value("string"),
                    "functional_groups": Sequence(Value("string")),
                    # HF Image decodes to PIL.Image on __getitem__.
                    "h_image": datasets.Image(),
                    "c_image": datasets.Image(),
                }
            ),
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> list[datasets.SplitGenerator]:
        """Return split generators derived from the dataset directory.

        Four source layouts are supported:

        1. A single ``.pkl`` file → ``Split.TRAIN``.
        2. ``samples.jsonl`` + split id files.
        3. Pre-split ``train.pkl`` / ``validation.pkl`` / ``test.pkl``.
        4. ``selected.pkl`` → internal scaffold split into train / test.
        """
        if self.config.dataset_path is None:
            raise ValueError("dataset_path must be provided.")

        base = Path(self.config.dataset_path)

        # --- single file -----------------------------------------------
        if base.is_file():
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={"split_key": None},
                )
            ]

        if not base.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {base}")

        # --- JSONL with id-list splits -----------------------------------
        if (base / "samples.jsonl").exists():
            generators: list[datasets.SplitGenerator] = []
            for split_name, hf_split in [
                ("train", datasets.Split.TRAIN),
                ("validation", datasets.Split.VALIDATION),
                ("test", datasets.Split.TEST),
            ]:
                if _split_ids_path(base, split_name).exists():
                    generators.append(
                        datasets.SplitGenerator(
                            name=hf_split,
                            gen_kwargs={"split_key": split_name},
                        )
                    )

            if generators:
                return generators

            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={"split_key": None},
                )
            ]

        # --- pre-split files -------------------------------------------
        generators = []
        for split_name, hf_split in [
            ("train", datasets.Split.TRAIN),
            ("validation", datasets.Split.VALIDATION),
            ("test", datasets.Split.TEST),
        ]:
            if (base / f"{split_name}.pkl").exists():
                generators.append(
                    datasets.SplitGenerator(
                        name=hf_split,
                        gen_kwargs={"split_key": split_name},
                    )
                )

        if generators:
            return generators

        # --- selected.pkl → scaffold split ----------------------------
        if (base / "selected.pkl").exists():
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={"split_key": "train"},
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={"split_key": "test"},
                ),
            ]

        raise FileNotFoundError(
            f"No valid dataset file found in {base}. "
            f"Expected train.pkl / test.pkl / validation.pkl or selected.pkl."
        )

    def _generate_examples(
        self, split_key: str | None
    ):
        """Yield ``(idx, row_dict)`` pairs for the Arrow table.

        Parameters
        ----------
        split_key
            ``"train"``, ``"test"``, ``None``, etc. — forwarded to
            :func:`_resolve_and_load_samples`.
        """
        samples = _resolve_and_load_samples(
            dataset_path=self.config.dataset_path,
            split=split_key,
            train_size=self.config.train_size,
        )

        for idx, sample in enumerate(samples):
            sample_id = str(sample.get("id", idx))

            h_png = self._render(sample, nucleus="1h")
            c_png = self._render(sample, nucleus="13c")

            smiles = sample_smiles(sample) or ""
            groups = sample_fg(sample) or []

            yield idx, {
                "id": sample_id,
                "sample_pickle": pickle.dumps(
                    sample, protocol=pickle.HIGHEST_PROTOCOL
                ),
                "smiles": smiles,
                "functional_groups": list(groups),
                "h_image": {"bytes": h_png, "path": None},
                "c_image": {"bytes": c_png, "path": None},
            }

    def _render(self, sample: dict[str, Any], nucleus: str) -> bytes:
        """Render one NMR spectrum, returning raw PNG bytes.

        When *render_cache_dir* is configured, rendered images are cached
        to disk and reused across instantiations.  Cache hits read
        raw bytes directly, avoiding a PIL decode / encode round-trip.

        Parameters
        ----------
        sample
            Sample dictionary with an ``"id"`` key.
        nucleus
            ``"1h"`` or ``"13c"``.

        Returns
        -------
        bytes
            PNG-encoded image bytes.
        """
        sample_id = str(sample.get("id", "unknown"))
        cache_key = _safe_cache_key(sample_id)

        render_cache_dir = (
            Path(self.config.render_cache_dir)
            if self.config.render_cache_dir is not None
            else None
        )

        # --- cache hit -------------------------------------------------
        if render_cache_dir is not None:
            cache_path = render_cache_dir / f"{cache_key}_{nucleus}.png"
            if cache_path.exists():
                return cache_path.read_bytes()

        # --- render ----------------------------------------------------
        if nucleus == "1h":
            image = hydrogen_to_spectra(
                sample,
                snr=self.config.h_snr,
                seed=_stable_render_seed(
                    self.config.render_seed,
                    sample_id,
                    nucleus,
                ),
            )
        elif nucleus == "13c":
            image = carbon_to_spectra(
                sample,
                snr=self.config.c_snr,
                seed=_stable_render_seed(
                    self.config.render_seed,
                    sample_id,
                    nucleus,
                ),
            )
        else:
            raise ValueError(f"Unsupported nucleus: {nucleus!r}")

        if not isinstance(image, PILImage.Image):
            image = PILImage.fromarray(image)
        image = image.convert("RGB")

        if self.config.image_size is not None:
            image = image.resize(
                tuple(self.config.image_size), Resampling.LANCZOS
            )

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        # --- persist to cache ------------------------------------------
        if render_cache_dir is not None:
            cache_path = render_cache_dir / f"{cache_key}_{nucleus}.png"
            _atomic_write_bytes(cache_path, png_bytes)

        return png_bytes

    def _validate_render_cache(self, cache_dir: Path) -> None:
        """Check the cache manifest; delete stale PNGs if params changed.

        The manifest records the SNR values, image size, and version tag
        that produced the cached PNGs.  When any of these differ from the
        current configuration, all cached ``.png`` files are removed so
        they will be re-rendered.
        """
        manifest_path = cache_dir / _CACHE_MANIFEST_FILENAME

        current_manifest = {
            "h_snr": self.config.h_snr,
            "c_snr": self.config.c_snr,
            "image_size": (
                list(self.config.image_size)
                if self.config.image_size else None
            ),
            "render_seed": self.config.render_seed,
            "version": self.config.render_cache_version,
        }

        if manifest_path.exists():
            try:
                stored = json.loads(manifest_path.read_text(encoding="utf-8"))
                if stored == current_manifest:
                    return  # cache is valid
            except json.JSONDecodeError:
                pass  # corrupt manifest → invalidate

        for png_file in cache_dir.glob("*.png"):
            png_file.unlink()

        _atomic_write_text(
            manifest_path,
            json.dumps(current_manifest, ensure_ascii=False, indent=2),
        )


class NMRMessageTransform:
    """Dynamically converts Arrow rows into multimodal chat messages.

    Applied via :meth:`datasets.Dataset.with_transform` so that task
    selection, prompt construction, and target generation vary across
    epochs without modifying the underlying Arrow data.

    Parameters
    ----------
    task_probs
        Task sampling probabilities (default: 80 % structure,
        20 % functional group).
    seed
        Optional seed for the internal NumPy RNG used for task sampling.
    """

    def __init__(
        self,
        task_probs: dict[str, float] | None = None,
        seed: int | None = None,
        target_format: str = "smiles",
        include_formula: bool = True,
    ) -> None:
        self.tasks, self.weights = _normalise_task_probs(task_probs)
        self.rng = np.random.default_rng(seed)
        if target_format not in {"smiles", "reasoning"}:
            raise ValueError(
                "target_format must be 'smiles' or 'reasoning', "
                f"got {target_format!r}"
            )
        self.target_format = target_format
        self.include_formula = bool(include_formula)

    def __call__(self, batch: dict[str, Any]) -> dict[str, list[Any]]:
        """Transform a batch of Arrow rows into chat messages.

        Parameters
        ----------
        batch
            Dictionary with ``"h_image"``, ``"c_image"``, and
            ``"sample_pickle"`` columns.

        Returns
        -------
        dict[str, list[Any]]
            ``{"messages": [[user_msg, assistant_msg], ...]}``.
        """
        h_images = self._as_list(batch["h_image"])
        c_images = self._as_list(batch["c_image"])
        sample_blobs = self._as_list(batch["sample_pickle"])

        messages_batch: list[list[dict[str, Any]]] = []

        for h_image, c_image, sample_blob in zip(
            h_images, c_images, sample_blobs
        ):
            sample = self._load_sample(sample_blob)
            task = str(self.rng.choice(self.tasks, p=self.weights))

            if task == "structure":
                prompt = build_structure_prompt(
                    sample,
                    prompt=str(self.rng.choice(STRUCTURE_PROMPTS)),
                    include_formula=self.include_formula,
                )
                if self.target_format == "reasoning":
                    target = build_reasoning_target(sample)
                else:
                    target = sample_smiles(sample) or ""
            elif task == "functional_group":
                prompt = str(
                    self.rng.choice(FUNCTIONAL_GROUP_PROMPTS)
                ).format(peak_tables="(see spectra above)")
                groups = sample_fg(sample) or []
                target = ", ".join(groups) if groups else "Unknown"
            else:
                raise ValueError(f"Unsupported task: {task!r}")

            messages_batch.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": h_image},
                        {"type": "image", "image": c_image},
                        {"type": "text", "text": prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": target}],
                },
            ])

        return {"messages": messages_batch}

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        """Wrap a non-list *value* in a single-element list."""
        return value if isinstance(value, list) else [value]

    @staticmethod
    def _load_sample(blob: bytes | bytearray | memoryview) -> dict[str, Any]:
        """Deserialise a pickled sample dictionary."""
        if isinstance(blob, memoryview):
            blob = blob.tobytes()
        return pickle.loads(blob)


def _resize_image(
    image: PILImage.Image,
    image_size: tuple[int, int] | list[int] | None,
) -> PILImage.Image:
    """Resize an RGB PIL image when a target size is configured."""
    image = image.convert("RGB")
    if image_size is None:
        return image
    return image.resize(tuple(image_size), Resampling.LANCZOS)


def render_sample_images(
    sample: dict[str, Any],
    *,
    h_snr: float = 500.0,
    c_snr: float = 300.0,
    render_seed: int | None = 3407,
    image_size: tuple[int, int] | list[int] | None = None,
) -> tuple[PILImage.Image, PILImage.Image]:
    """Render 1H and 13C spectrum images for one sample."""
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


class LazyNMRJsonlDataset(Dataset):
    """Lazy JSONL-backed NMR instruction dataset.

    This path is intended for million-scale experiments.  It keeps only line
    offsets in memory, reads one sample at a time, renders spectra on demand,
    and returns the same ``{"messages": ...}`` shape expected by the VLM SFT
    collator.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        split: str = "train",
        task_probs: dict[str, float] | None = None,
        target_format: str = "smiles",
        include_formula: bool = True,
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
            raise ValueError(
                "image_backend must be 'lazy_render' or 'pre_rendered', "
                f"got {image_backend!r}"
            )
        if missing_image_policy not in {"error", "lazy_render"}:
            raise ValueError(
                "missing_image_policy must be 'error' or 'lazy_render', "
                f"got {missing_image_policy!r}"
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
        self.transform = NMRMessageTransform(
            task_probs=task_probs,
            seed=seed,
            target_format=target_format,
            include_formula=include_formula,
        )
        self.offsets = self._index_offsets()

    def _index_offsets(self) -> list[int]:
        """Build line offsets for the configured split."""
        split_ids_path = _split_ids_path(self.dataset_dir, self.split)
        split_ids = None
        if split_ids_path.exists():
            split_ids = {
                line.strip()
                for line in split_ids_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }

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
        return offsets

    def __len__(self) -> int:
        """Return number of samples in the selected split."""
        return len(self.offsets)

    def _load_sample_at(self, offset: int) -> dict[str, Any]:
        """Load one JSONL sample by byte offset."""
        with self.jsonl_path.open("rb") as handle:
            handle.seek(offset)
            return json.loads(handle.readline())

    def _render_images(self, sample: dict[str, Any]) -> tuple[PILImage.Image, PILImage.Image]:
        """Return 1H and 13C images for one sample."""
        if self.image_backend == "pre_rendered":
            return self._load_pre_rendered_images(sample)
        return render_sample_images(
            sample,
            h_snr=self.h_snr,
            c_snr=self.c_snr,
            render_seed=self.render_seed,
            image_size=self.image_size,
        )

    def _rendered_image_path(self, sample: dict[str, Any], nucleus: str) -> Path:
        """Return the expected pre-rendered PNG path for one sample image."""
        if self.rendered_image_dir is None:
            raise RuntimeError("rendered_image_dir is not configured.")
        sample_id = str(sample.get("id", "unknown"))
        cache_key = _safe_cache_key(sample_id)
        return self.rendered_image_dir / f"{cache_key}_{nucleus}.png"

    def _load_pre_rendered_images(
        self,
        sample: dict[str, Any],
    ) -> tuple[PILImage.Image, PILImage.Image]:
        """Load pre-rendered 1H and 13C PNGs for one sample."""
        h_path = self._rendered_image_path(sample, "1h")
        c_path = self._rendered_image_path(sample, "13c")
        missing = [str(path) for path in [h_path, c_path] if not path.exists()]
        if missing:
            if self.missing_image_policy == "lazy_render":
                return render_sample_images(
                    sample,
                    h_snr=self.h_snr,
                    c_snr=self.c_snr,
                    render_seed=self.render_seed,
                    image_size=self.image_size,
                )
            raise FileNotFoundError(
                "Missing pre-rendered NMR image(s): " + ", ".join(missing)
            )

        with PILImage.open(h_path) as h_image:
            h_loaded = _resize_image(h_image, self.image_size)
        with PILImage.open(c_path) as c_image:
            c_loaded = _resize_image(c_image, self.image_size)
        return h_loaded, c_loaded

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return one multimodal instruction sample."""
        sample = self._load_sample_at(self.offsets[idx])
        h_image, c_image = self._render_images(sample)
        transformed = self.transform(
            {
                "h_image": [h_image],
                "c_image": [c_image],
                "sample_pickle": [pickle.dumps(sample)],
            }
        )
        return {"messages": transformed["messages"][0]}


def load_lazy_nmr_dataset(
    dataset_dir: str | Path,
    *,
    split: str = "train",
    task_probs: dict[str, float] | None = None,
    target_format: str = "smiles",
    include_formula: bool = True,
    seed: int | None = 3407,
    h_snr: float = 500.0,
    c_snr: float = 300.0,
    render_seed: int | None = 3407,
    image_size: tuple[int, int] | list[int] | None = None,
    image_backend: str = "lazy_render",
    rendered_image_dir: str | Path | None = None,
    missing_image_policy: str = "error",
) -> LazyNMRJsonlDataset:
    """Load a lazy JSONL-backed NMR instruction dataset."""
    return LazyNMRJsonlDataset(
        dataset_dir,
        split=split,
        task_probs=task_probs,
        target_format=target_format,
        include_formula=include_formula,
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
    dataset_path: str | Path,
    *,
    split: str | None = None,
    train_size: float = 0.8,
) -> list[dict[str, Any]]:
    """Load raw normalized NMR sample dictionaries without rendering images."""
    return _resolve_and_load_samples(dataset_path, split=split, train_size=train_size)


def load_nmr_dataset(
    dataset_path: str | Path,
    *,
    split: str | None = None,
    train_size: float = 0.8,
    hf_cache_dir: str | None = None,
    render_cache_dir: str | None = None,
    h_snr: float = 500.0,
    c_snr: float = 300.0,
    render_cache_version: str = "1",
    render_seed: int | None = 3407,
    image_size: tuple[int, int] | None = None,
    with_messages: bool = True,
    task_probs: dict[str, float] | None = None,
    target_format: str = "smiles",
    include_formula: bool = True,
    seed: int | None = None,
):
    """Load the NMR dataset as a :class:`~datasets.Dataset` or dict of splits.

    Parameters
    ----------
    dataset_path
        Path to a ``.pkl`` file or a directory with split files.
    split
        Optional split name forwarded to :meth:`Builder.as_dataset`.
    train_size
        Fraction of samples allocated to training when splitting
        ``selected.pkl`` (default 0.8).
    hf_cache_dir
        HuggingFace datasets cache directory.
    render_cache_dir
        Optional PNG render-cache directory.
    h_snr / c_snr
        SNR values for 1H / 13C rendering (used with *render_cache_dir*).
    render_cache_version
        Cache-manifest version tag.
    render_seed
        Base seed for deterministic per-sample spectrum rendering. Set to
        ``None`` to keep stochastic rendering.
    image_size
        Optional ``(width, height)`` resize target.
    with_messages
        When ``True`` (default), apply :class:`NMRMessageTransform` so
        ``__getitem__`` returns chat messages.  When ``False``, return
        the raw Arrow columns.
    task_probs
        Task sampling probabilities.
    target_format
        Structure-task target format. ``"smiles"`` trains direct canonical
        SMILES output; ``"reasoning"`` trains a compact reasoning + SELFIES
        + canonical SMILES target.
    include_formula
        Whether to include the molecular formula in structure prompts.
    seed
        RNG seed for task sampling.

    Returns
    -------
    datasets.Dataset or datasets.DatasetDict
    """
    builder = NMRDatasetBuilder(
        config_name="default",
        cache_dir=hf_cache_dir,
        dataset_path=str(dataset_path),
        train_size=train_size,
        render_cache_dir=render_cache_dir,
        h_snr=h_snr,
        c_snr=c_snr,
        render_cache_version=render_cache_version,
        render_seed=render_seed,
        image_size=image_size,
    )

    builder.download_and_prepare()
    ds = builder.as_dataset(split=split)

    if not with_messages:
        return ds

    transform = NMRMessageTransform(
        task_probs=task_probs,
        seed=seed,
        target_format=target_format,
        include_formula=include_formula,
    )
    return ds.with_transform(
        transform,
        columns=["h_image", "c_image", "sample_pickle"],
        output_all_columns=False,
    )
