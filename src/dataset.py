import random
import pickle
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset

try:
    from .spectra import CombineSpectra, HydrogenToSpectra, CarbonToSpectra
except ImportError:
    from spectra import CombineSpectra, HydrogenToSpectra, CarbonToSpectra


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
        with open(dataset_path, "rb") as f:
            self.samples = pickle.load(f)

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

    def _peaks(self, sample, nucleus):
        nmr = sample.get(nucleus, {})
        return nmr.get("peaks", nmr.get("data", [])) or []

    def _selfies(self, sample):
        return sample.get("selfies") or sample.get("SELFIES") or ""

    def _canonical_smiles(self, sample):
        return (
            sample.get("canonical_smiles")
            or sample.get("canonical_SMILES")
            or sample.get("SMILES")
            or sample.get("smiles")
            or ""
        )

    def format_1h_peak(self, peak):
        shift = float(peak["shift"])
        mult = peak.get("multiplicity", "s")
        integration = peak.get("integration", 1)
        J = peak.get("J", [])
        integration_text = f"{integration:g}H" if isinstance(integration, (int, float)) else str(integration)
        if J:
            j_str = ", ".join([f"{float(j):.1f}" for j in J])
            return f"{shift:.2f} ppm ({mult}, J={j_str} Hz, {integration_text})"
        return f"{shift:.2f} ppm ({mult}, {integration_text})"

    def format_13c_peak(self, peak):
        shift = peak["shift"] if isinstance(peak, dict) else peak
        if isinstance(shift, list):
            return "/".join(f"{float(x):.1f}" for x in shift)
        return f"{float(shift):.1f}"

    def build_structure_reasoning_prompt(self, sample):
        prompt = random.choice(STRUCTURE_PROMPTS)
        text = []
        text.append(prompt)
        text.append(
            "Return a concise spectral reasoning process first, then provide Final SELFIES "
            "and Final canonical SMILES."
        )

        # 1H peaks
        h_nmr = sample.get("1H_NMR", {})
        h_peaks = self._peaks(sample, "1H_NMR")
        h_strings = []
        for p in h_peaks[:30]:
            h_strings.append(self.format_1h_peak(p))

        text.append(
            f"1H NMR metadata: frequency={h_nmr.get('frequency', 'unknown')}, "
            f"solvent={h_nmr.get('solvent', 'unknown')}\n"
            "1H NMR peak table:\n"
            + "\n".join(h_strings)
        )

        # 13C peaks
        c_nmr = sample.get("13C_NMR", {})
        c_peaks = self._peaks(sample, "13C_NMR")
        c_strings = [self.format_13c_peak(p) for p in c_peaks[:80]]
        text.append(
            f"13C NMR metadata: frequency={c_nmr.get('frequency', 'unknown')}, "
            f"solvent={c_nmr.get('solvent', 'unknown')}\n"
            "13C NMR peak table:\n"
            + ", ".join(c_strings)
        )

        text.append(
            "NMR rules to consider:\n"
            "- 1H integration constrains the number of equivalent hydrogens.\n"
            "- Multiplicity and J coupling suggest neighboring proton environments.\n"
            "- 13C peak count approximates distinct carbon environments.\n"
            "- Chemical shift regions suggest functional groups and hybridization.\n"
            "- The final structure must be consistent with both 1H and 13C evidence."
        )

        return "\n\n".join(text)

    def build_structure_prompt(self, sample):
        return self.build_structure_reasoning_prompt(sample)

    def build_reasoning_prompt(self, sample):
        return self.build_structure_reasoning_prompt(sample)

    def build_fg_prompt(self, sample):
        prompt = random.choice(FG_PROMPTS)
        return prompt

    def build_target(self, sample, task):
        selfies = self._selfies(sample)
        smiles = self._canonical_smiles(sample)

        if task == "structure_reasoning":
            return self.build_reasoning_target(sample, selfies, smiles)

        # structure prediction
        if task == "structure":
            return selfies or smiles

        # reasoning
        elif task == "reasoning":
            return self.build_reasoning_target(sample, selfies, smiles)

        # functional group
        elif task == "functional_group":
            if "functional_groups" in sample:
                return ", ".join(
                    sample["functional_groups"]
                )
            return "Unknown"

        return selfies or smiles

    def build_reasoning_target(self, sample, selfies, smiles):
        h_peaks = self._peaks(sample, "1H_NMR")
        c_peaks = self._peaks(sample, "13C_NMR")
        fg = sample.get("functional_groups") or []
        formula = sample.get("molecular_formula", "unknown")
        h_total = sum(float(p.get("integration", 0)) for p in h_peaks if isinstance(p, dict))

        lines = [
            "Spectral reasoning:",
            f"- The 1H NMR spectrum contains {len(h_peaks)} reported proton environments with total integration about {h_total:g}H.",
            f"- The 13C NMR spectrum contains {len(c_peaks)} reported carbon environments.",
        ]
        if fg:
            lines.append(f"- Functional-group evidence is consistent with: {', '.join(fg)}.")
        lines.extend(
            [
                f"- The proposed molecular formula is {formula}.",
                "- The final structure should satisfy the reported 1H integration, splitting patterns, and 13C environment count.",
                "",
                f"Final SELFIES: {selfies}",
                f"Final canonical SMILES: {smiles}",
            ]
        )
        return "\n".join(lines)

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
