import pickle
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.dataset import NMRexpDataset


def sample_row():
    return {
        "id": "sample-1",
        "canonical_smiles": "CCO",
        "selfies": "[C][C][O]",
        "molecular_formula": "C2H6O",
        "functional_groups": ["alcohol"],
        "1H_NMR": {
            "frequency": "400 MHz",
            "solvent": "CDCl3",
            "peaks": [
                {"shift": 3.65, "multiplicity": "q", "J": [7.1], "integration": 2.0},
                {"shift": 1.20, "multiplicity": "t", "J": [7.1], "integration": 3.0},
            ],
        },
        "13C_NMR": {
            "frequency": "101 MHz",
            "solvent": "CDCl3",
            "peaks": [
                {"shift": 58.1},
                {"shift": 18.2},
            ],
        },
    }


class DatasetTests(unittest.TestCase):
    def build_dataset(self):
        tmp = tempfile.TemporaryDirectory()
        path = Path(tmp.name) / "rows.pkl"
        with path.open("wb") as f:
            pickle.dump([sample_row()], f)
        dataset = NMRexpDataset(path, task_probs={"structure_reasoning": 1.0})
        return tmp, dataset

    def test_prompt_contains_peak_table_and_rules(self):
        tmp, dataset = self.build_dataset()
        self.addCleanup(tmp.cleanup)

        prompt = dataset.build_structure_reasoning_prompt(dataset.samples[0])

        self.assertIn("1H NMR peak table", prompt)
        self.assertIn("3.65 ppm (q, J=7.1 Hz, 2H)", prompt)
        self.assertIn("13C NMR peak table", prompt)
        self.assertIn("58.1", prompt)
        self.assertIn("NMR rules", prompt)

    def test_target_contains_reasoning_then_selfies_and_smiles(self):
        tmp, dataset = self.build_dataset()
        self.addCleanup(tmp.cleanup)

        target = dataset.build_target(dataset.samples[0], "structure_reasoning")

        self.assertIn("Spectral reasoning:", target)
        self.assertIn("Final SELFIES:", target)
        self.assertIn("[C][C][O]", target)
        self.assertIn("Final canonical SMILES:", target)
        self.assertIn("CCO", target)

    def test_getitem_returns_multimodal_messages_with_rgb_image(self):
        tmp, dataset = self.build_dataset()
        self.addCleanup(tmp.cleanup)

        item = dataset[0]
        content = item["messages"][0]["content"]
        image = content[0]["image"]
        text = content[1]["text"]

        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.mode, "RGB")
        self.assertIn("molecular structure", text)
        self.assertIn("assistant", item["messages"][1]["role"])


if __name__ == "__main__":
    unittest.main()
