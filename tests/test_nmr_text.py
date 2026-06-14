import unittest

from src.utils.nmr_utils import (
    build_reasoning_target,
    build_structure_prompt,
    canonical_smiles,
    format_1h_peak,
    format_13c_peak,
    sample_peaks,
    selfies,
)


def sample_row():
    return {
        "canonical_smiles": "CCO",
        "selfies": "[C][C][O]",
        "molecular_formula": "C2H6O",
        "functional_groups": ["alcohol"],
        "1H_NMR": {
            "frequency": "400 MHz",
            "solvent": "CDCl3",
            "peaks": [
                {"shift": 3.65, "multiplicity": "q", "J": [7.1], "integration": 2.0},
            ],
        },
        "13C_NMR": {
            "frequency": "101 MHz",
            "solvent": "CDCl3",
            "peaks": [{"shift": 58.1}],
        },
    }


class NmrTextTests(unittest.TestCase):
    def test_sample_accessors_support_current_schema(self):
        row = sample_row()
        self.assertEqual(selfies(row), "[C][C][O]")
        self.assertEqual(canonical_smiles(row), "CCO")
        self.assertEqual(len(sample_peaks(row, "1H_NMR")), 1)

    def test_peak_formatters_are_stable(self):
        self.assertEqual(
            format_1h_peak({"shift": 3.65, "multiplicity": "q", "J": [7.1], "integration": 2.0}),
            "3.65 ppm (q, J=7.1 Hz, 2H)",
        )
        self.assertEqual(format_13c_peak({"shift": 58.1}), "58.1")

    def test_prompt_and_target_contain_required_sections(self):
        row = sample_row()
        prompt = build_structure_prompt(row, prompt="Predict the molecular structure.")
        target = build_reasoning_target(row)

        self.assertIn("1H NMR peak table", prompt)
        self.assertIn("NMR rules", prompt)
        self.assertIn("Spectral reasoning:", target)
        self.assertIn("Final SELFIES: [C][C][O]", target)
        self.assertIn("Final canonical SMILES: CCO", target)


if __name__ == "__main__":
    unittest.main()
