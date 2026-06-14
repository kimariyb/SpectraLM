import unittest

import src.sample_splits as sample_splits
from src.sample_splits import (
    assign_complexity_bin,
    functional_group_labels,
    representative_sample,
)


requires_rdkit = unittest.skipIf(sample_splits.Chem is None, "RDKit is not installed")


@requires_rdkit
class SampleSplitsTests(unittest.TestCase):
    def test_functional_group_labels_identifies_common_groups(self):
        self.assertIn("alcohol", functional_group_labels("CCO"))
        self.assertIn("carboxylic_acid", functional_group_labels("CC(=O)O"))
        self.assertIn("aromatic", functional_group_labels("c1ccccc1"))

    def test_assign_complexity_bin_uses_molecule_and_peak_counts(self):
        row = {
            "canonical_smiles": "CCO",
            "1H_NMR": {"peaks": [{"shift": 1.0}, {"shift": 3.5}]},
            "13C_NMR": {"peaks": [{"shift": 10.0}, {"shift": 60.0}]},
        }
        label = assign_complexity_bin(row)
        self.assertIn("heavy_atoms:low", label)
        self.assertIn("h_peaks:low", label)
        self.assertIn("c_peaks:low", label)

    def test_representative_sample_respects_split_and_scaffold_cap(self):
        rows = [
            {
                "id": "a",
                "split": "train",
                "canonical_smiles": "CCO",
                "murcko_scaffold": "acyclic:CCO",
                "1H_NMR": {"peaks": [1]},
                "13C_NMR": {"peaks": [1]},
            },
            {
                "id": "b",
                "split": "train",
                "canonical_smiles": "CCN",
                "murcko_scaffold": "acyclic:CCN",
                "1H_NMR": {"peaks": [1]},
                "13C_NMR": {"peaks": [1]},
            },
            {
                "id": "c",
                "split": "train",
                "canonical_smiles": "c1ccccc1",
                "murcko_scaffold": "c1ccccc1",
                "1H_NMR": {"peaks": [1, 2]},
                "13C_NMR": {"peaks": [1, 2, 3]},
            },
            {
                "id": "d",
                "split": "test",
                "canonical_smiles": "CC(=O)O",
                "murcko_scaffold": "acyclic:CC(=O)O",
                "1H_NMR": {"peaks": [1]},
                "13C_NMR": {"peaks": [1, 2]},
            },
        ]

        selected = representative_sample(
            rows,
            split_name="train",
            target_size=3,
            max_per_scaffold=1,
            seed=1,
        )

        self.assertEqual(len(selected), 3)
        self.assertEqual({row["split"] for row in selected}, {"train"})
        self.assertEqual(len({row["murcko_scaffold"] for row in selected}), 3)

    def test_representative_sample_filters_long_or_large_molecules(self):
        rows = [
            {
                "id": "a",
                "split": "train",
                "canonical_smiles": "CCO",
                "selfies": "[C][C][O]",
                "murcko_scaffold": "acyclic:CCO",
                "1H_NMR": {"peaks": [1]},
                "13C_NMR": {"peaks": [1]},
            },
            {
                "id": "b",
                "split": "train",
                "canonical_smiles": "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
                "selfies": "[C]" * 80,
                "murcko_scaffold": "acyclic:long",
                "1H_NMR": {"peaks": [1]},
                "13C_NMR": {"peaks": [1]},
            },
        ]

        selected = representative_sample(
            rows,
            split_name="train",
            target_size=2,
            max_per_scaffold=1,
            max_heavy_atoms=20,
            max_selfies_length=40,
            seed=1,
        )

        self.assertEqual([row["id"] for row in selected], ["a"])


if __name__ == "__main__":
    unittest.main()
