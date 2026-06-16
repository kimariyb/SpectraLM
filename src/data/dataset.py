"""HuggingFace ``datasets``-backed multimodal NMR reasoning dataset.

Each sample provides separate 1H and 13C spectrum images stored as Arrow
binary columns (decoded to PIL on access).  Task, prompt and target are
generated dynamically in ``__getitem__`` so they vary across epochs.

An optional disk cache avoids redundant spectrum re-rendering.
"""

from __future__ import annotations

import io
import json
import random
from pathlib import Path
from typing import Any

from datasets import Dataset, Image as HFImage
from PIL import Image as PILImage

from src.data.molecules import sample_selfies, sample_smiles
from src.io import load_pickle_list
from src.spectra.render import carbon_to_spectra, hydrogen_to_spectra
from src.training.prompts import (
    FUNCTIONAL_GROUP_PROMPTS,
    STRUCTURE_PROMPTS,
    build_reasoning_target,
    build_structure_prompt,
)

_DEFAULT_TASK_PROBS: dict[str, float] = {
    "structure_reasoning": 0.8,
    "functional_group": 0.1,
    "structure": 0.1,
}


class NmrReasoningDataset:
    """Multitask NMR dataset backed by :class:`datasets.Dataset`.

    Images are pre-rendered once at construction time and stored as Arrow
    binary columns.  Task selection, prompt construction, and target
    generation happen lazily in :meth:`__getitem__`, giving different
    task mixes each epoch.

    Parameters
    ----------
    dataset_path
        Pickle file containing normalised SpectraLM samples.
    task_probs
        Task sampling probabilities.
    cache_dir
        Optional directory for rendered-spectrum PNG cache.  When set,
        images use a fixed SNR and are saved to disk; subsequent
        instantiations skip rendering.
    h_snr
        1H signal-to-noise ratio (only used with *cache_dir*).
    c_snr
        13C signal-to-noise ratio (only used with *cache_dir*).
    """
    def __init__(
        self,
        dataset_path: str,
        task_probs: dict[str, float] | None = None,
        cache_dir: str | None = None,
        h_snr: float = 500.0,
        c_snr: float = 300.0,
    ) -> None:
        # --- load samples ----------------------------------------------------
        self._samples = load_pickle_list(dataset_path)

        tp = task_probs or _DEFAULT_TASK_PROBS
        self._tasks = list(tp.keys())
        self._task_weights = list(tp.values())

        # --- cache -----------------------------------------------------------
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._h_snr = h_snr
        self._c_snr = c_snr
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

        # --- build Arrow dataset (images only) -------------------------------
        self.ds = self._build_arrow_dataset()

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict[str, list[dict[str, Any]]]:
        """Return one multimodal chat-style training example.

        Task, prompt, and target are sampled on every access so they
        vary across training epochs.
        """
        row = self.ds[idx]

        # Reconstruct the sample dictionary from stored JSON
        sample = json.loads(row["sample_json"])

        # Random task selection
        task = random.choices(self._tasks, weights=self._task_weights, k=1)[0]

        if task in ("structure", "structure_reasoning", "reasoning"):
            prompt = build_structure_prompt(
                sample, prompt=random.choice(STRUCTURE_PROMPTS),
            )
        else:
            prompt = random.choice(FUNCTIONAL_GROUP_PROMPTS).format(
                peak_tables="(see spectra above)",
            )

        target = _build_target(sample, task)

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

    def _build_arrow_dataset(self) -> Dataset:
        """Build a :class:`Dataset` with pre-rendered images and serialised
        sample metadata.
        """

        def _generator():
            for sample in self._samples:
                # Render (or load from cache)
                h_img = self._render(sample, "1h")
                c_img = self._render(sample, "13c")

                h_buf = io.BytesIO()
                c_buf = io.BytesIO()
                h_img.save(h_buf, format="PNG")
                c_img.save(c_buf, format="PNG")

                yield {
                    "id": sample.get("id", ""),
                    "h_image": h_buf.getvalue(),
                    "c_image": c_buf.getvalue(),
                    "sample_json": json.dumps(sample, ensure_ascii=False),
                }

        ds = Dataset.from_generator(_generator)
        ds = ds.cast_column("h_image", HFImage())
        ds = ds.cast_column("c_image", HFImage())
        return ds

    def _render(self, sample: dict[str, Any], nucleus: str) -> PILImage.Image:
        """Render one NMR image, with optional disk caching.

        Parameters
        ----------
        sample
            Sample dictionary.
        nucleus
            ``"1h"`` or ``"13c"``.

        Returns
        -------
        PILImage.Image
            RGB spectrum image.
        """
        sample_id = sample.get("id", "unknown")

        if self._cache_dir:
            cache_path = self._cache_dir / f"{sample_id}_{nucleus}.png"
            if cache_path.exists():
                return PILImage.open(cache_path).convert("RGB")

        # Render
        if nucleus == "1h":
            snr = self._h_snr if self._cache_dir else random.uniform(300, 600)
            image = hydrogen_to_spectra(sample, snr=snr)
        else:
            snr = self._c_snr if self._cache_dir else random.uniform(100, 400)
            image = carbon_to_spectra(sample, snr=snr)

        if not isinstance(image, PILImage.Image):
            image = PILImage.fromarray(image)
        image = image.convert("RGB")

        if self._cache_dir:
            image.save(cache_path, format="PNG")

        return image

def _build_target(sample: dict[str, Any], task: str) -> str:
    """Build a target string for a sampled task."""
    if task in ("structure_reasoning", "reasoning"):
        return build_reasoning_target(sample)
    if task == "structure":
        return sample_selfies(sample) or sample_smiles(sample)
    if task == "functional_group":
        groups = sample.get("functional_groups", [])
        return ", ".join(groups) if groups else "Unknown"
    return sample_selfies(sample) or sample_smiles(sample)


# Backward-compatibility alias.
NMRexpDataset = NmrReasoningDataset
