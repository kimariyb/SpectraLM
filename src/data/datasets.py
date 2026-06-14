import random
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset

try:
    from ..utils.io_utils import load_pickle_list
    from ..utils.nmr_utils import build_reasoning_target, build_structure_prompt, canonical_smiles, selfies
    from .spectra import CombineSpectra
except ImportError:
    from utils.io_utils import load_pickle_list
    from utils.nmr_utils import build_reasoning_target, build_structure_prompt, canonical_smiles, selfies
    from data.spectra import CombineSpectra


IGNORE_INDEX = -100


# Prompt Templates
STRUCTURE_PROMPTS = [
    "Predict the molecular structure from the provided 1H and 13C NMR spectra.",
    "Analyze the multimodal NMR evidence and infer the molecular structure.",
    "Use the spectra and peak tables to determine the molecular structure.",
]

FG_PROMPTS = [
    "Identify the functional groups from the spectra.",
]


def pil_to_bytes(image: Image.Image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class NMRexpDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        task_probs=None,
    ):
        # load dataset
        self.samples = load_pickle_list(dataset_path)

        # task sampling
        if task_probs is None:
            task_probs = {
                "structure_reasoning": 0.8,
                "functional_group": 0.1,
                "structure": 0.1,
            }

        self.task_probs   = task_probs
        self.tasks        = list(task_probs.keys())
        self.task_weights = list(task_probs.values())

    def __len__(self):
        return len(self.samples)

    def build_structure_reasoning_prompt(self, sample):
        return build_structure_prompt(sample, prompt=random.choice(STRUCTURE_PROMPTS))

    def build_structure_prompt(self, sample):
        return self.build_structure_reasoning_prompt(sample)

    def build_reasoning_prompt(self, sample):
        return self.build_structure_reasoning_prompt(sample)

    def build_fg_prompt(self, sample):
        prompt = random.choice(FG_PROMPTS)
        return prompt

    def build_target(self, sample, task):
        if task == "structure_reasoning":
            return self.build_reasoning_target(sample)

        # structure prediction
        if task == "structure":
            return selfies(sample) or canonical_smiles(sample)

        # reasoning
        elif task == "reasoning":
            return self.build_reasoning_target(sample)

        # functional group
        elif task == "functional_group":
            if "functional_groups" in sample:
                return ", ".join(
                    sample["functional_groups"]
                )
            return "Unknown"

        return selfies(sample) or canonical_smiles(sample)

    def build_reasoning_target(self, sample):
        return build_reasoning_target(sample)

    def build_image(self, sample):
        image = CombineSpectra(
            data=sample,
            h_snr=random.uniform(300, 600),
            c_snr=random.uniform(100, 400),
        )

        # numpy -> PIL
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")

        return image

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # task sampling
        task = random.choices(
            self.tasks, weights=self.task_weights, k=1
        )[0]

        # prompt
        if task in ("structure", "structure_reasoning"):
            prompt = self.build_structure_reasoning_prompt(sample)
        elif task == "reasoning":
            prompt = self.build_reasoning_prompt(sample)
        else:
            prompt = self.build_fg_prompt(sample)

        target = self.build_target(sample, task)
        image  = self.build_image(sample)

        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text",  "text": prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": target},
                    ],
                },
            ]
        }
