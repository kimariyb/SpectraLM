"""PyTorch datasets for multimodal NMR reasoning fine-tuning."""

from __future__ import annotations

import random
from io import BytesIO
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from spectralm.data.molecules import sample_selfies
from spectralm.io import load_pickle_list
from spectralm.spectra.render import combine_spectra
from spectralm.training.prompts import (
    FUNCTIONAL_GROUP_PROMPTS,
    STRUCTURE_PROMPTS,
    build_reasoning_target,
    build_structure_prompt,
    canonical_smiles,
)

IGNORE_INDEX = -100


def pil_to_bytes(image: Image.Image) -> bytes:
    """Serialize a PIL image as PNG bytes.

    Parameters
    ----------
    image
        Input PIL image.

    Returns
    -------
    bytes
        PNG-encoded image bytes.
    """
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class NmrReasoningDataset(Dataset):
    """Multitask NMR image and peak-table dataset for VLM fine-tuning.

    Parameters
    ----------
    dataset_path
        Pickle file containing normalized samples.
    task_probs
        Optional task sampling probabilities.
    """

    def __init__(self, dataset_path: str, task_probs: dict[str, float] | None = None) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        dataset_path
            Pickle file containing normalized samples.
        task_probs
            Optional task sampling probabilities.
        """
        self.samples = load_pickle_list(dataset_path)
        if task_probs is None:
            task_probs = {
                "structure_reasoning": 0.8,
                "functional_group": 0.1,
                "structure": 0.1,
            }
        self.task_probs = task_probs
        self.tasks = list(task_probs.keys())
        self.task_weights = list(task_probs.values())

    def __len__(self) -> int:
        """Return the number of samples.

        Returns
        -------
        int
            Dataset size.
        """
        return len(self.samples)

    def build_structure_reasoning_prompt(self, sample: dict[str, Any]) -> str:
        """Build a structure reasoning prompt for one sample.

        Parameters
        ----------
        sample
            Sample dictionary.

        Returns
        -------
        str
            Prompt text.
        """
        return build_structure_prompt(sample, prompt=random.choice(STRUCTURE_PROMPTS))

    def build_fg_prompt(self, sample: dict[str, Any]) -> str:
        """Build a functional-group prompt for one sample.

        Parameters
        ----------
        sample
            Sample dictionary.

        Returns
        -------
        str
            Prompt text.
        """
        return random.choice(FUNCTIONAL_GROUP_PROMPTS)

    def build_target(self, sample: dict[str, Any], task: str) -> str:
        """Build a target string for a sampled task.

        Parameters
        ----------
        sample
            Sample dictionary.
        task
            Task name.

        Returns
        -------
        str
            Target text.
        """
        if task == "structure_reasoning" or task == "reasoning":
            return build_reasoning_target(sample)
        if task == "structure":
            return sample_selfies(sample) or canonical_smiles(sample)
        if task == "functional_group":
            if "functional_groups" in sample:
                return ", ".join(sample["functional_groups"])
            return "Unknown"
        return sample_selfies(sample) or canonical_smiles(sample)

    def build_image(self, sample: dict[str, Any]) -> Image.Image:
        """Render the combined NMR spectrum image for one sample.

        Parameters
        ----------
        sample
            Sample dictionary.

        Returns
        -------
        PIL.Image.Image
            RGB combined spectrum image.
        """
        image = combine_spectra(
            sample=sample,
            h_snr=random.uniform(300, 600),
            c_snr=random.uniform(100, 400),
        )
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        return image.convert("RGB")

    def __getitem__(self, idx: int) -> dict[str, list[dict[str, Any]]]:
        """Return one multimodal chat-style training example.

        Parameters
        ----------
        idx
            Sample index.

        Returns
        -------
        dict[str, list[dict[str, Any]]]
            Messages formatted for VLM supervised fine-tuning.
        """
        sample = self.samples[idx]
        task = random.choices(self.tasks, weights=self.task_weights, k=1)[0]
        if task in ("structure", "structure_reasoning", "reasoning"):
            prompt = self.build_structure_reasoning_prompt(sample)
        else:
            prompt = self.build_fg_prompt(sample)
        target = self.build_target(sample, task)
        image = self.build_image(sample)
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": target}],
                },
            ]
        }


# Short-term package-internal compatibility alias.
NMRexpDataset = NmrReasoningDataset

