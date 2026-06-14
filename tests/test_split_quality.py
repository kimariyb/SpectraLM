import unittest

import src.split_quality as split_quality
from src.split_quality import (
    build_quality_report,
    canonical_smiles,
    molecule_formula,
    murcko_scaffold,
    scaffold_split,
)


requires_rdkit = unittest.skipIf(split_quality.Chem is None, "RDKit is not installed")


@requires_rdkit
class SplitQualityTests(unittest.TestCase):
    def test_canonical_smiles_normalizes_equivalent_strings(self):
        self.assertEqual(canonical_smiles("CCO"), canonical_smiles("OCC"))

    def test_murcko_scaffold_groups_same_ring_core(self):
        self.assertEqual(murcko_scaffold("Cc1ccccc1"), murcko_scaffold("Oc1ccccc1"))
        self.assertNotEqual(murcko_scaffold("Cc1ccccc1"), murcko_scaffold("c1ccncc1"))

    def test_molecule_formula_handles_valid_and_invalid_smiles(self):
        self.assertEqual(molecule_formula("CCO"), "C2H6O")
        self.assertIsNone(molecule_formula("not-a-smiles"))

    def test_scaffold_split_keeps_scaffolds_in_single_partition(self):
        samples = [
            {"id": "a", "canonical_smiles": "Cc1ccccc1", "1H_NMR": {"peaks": [1]}, "13C_NMR": {"peaks": [1]}},
            {"id": "b", "canonical_smiles": "Oc1ccccc1", "1H_NMR": {"peaks": [1]}, "13C_NMR": {"peaks": [1]}},
            {"id": "c", "canonical_smiles": "c1ccncc1", "1H_NMR": {"peaks": [1]}, "13C_NMR": {"peaks": [1]}},
            {"id": "d", "canonical_smiles": "CCO", "1H_NMR": {"peaks": [1]}, "13C_NMR": {"peaks": [1]}},
            {"id": "e", "canonical_smiles": "CCN", "1H_NMR": {"peaks": [1]}, "13C_NMR": {"peaks": [1]}},
        ]

        split = scaffold_split(samples, ratios=(0.6, 0.2, 0.2), seed=7)

        seen = {}
        for name, rows in split.items():
            for row in rows:
                scaffold = murcko_scaffold(row["canonical_smiles"])
                if scaffold in seen:
                    self.assertEqual(seen[scaffold], name)
                else:
                    seen[scaffold] = name

    def test_quality_report_counts_invalid_missing_and_duplicates(self):
        samples = [
            {"id": "a", "canonical_smiles": "CCO", "1H_NMR": {"peaks": [1]}, "13C_NMR": {"peaks": [1]}},
            {"id": "b", "canonical_smiles": "OCC", "1H_NMR": {"peaks": [1]}, "13C_NMR": {"peaks": [1]}},
            {"id": "c", "canonical_smiles": "bad", "1H_NMR": {"peaks": []}, "13C_NMR": {"peaks": [1]}},
            {"id": "d", "canonical_smiles": "CCN", "1H_NMR": {"peaks": [1]}, "13C_NMR": {"peaks": []}},
        ]

        report = build_quality_report(samples)

        self.assertEqual(report["total_samples"], 4)
        self.assertEqual(report["valid_molecules"], 3)
        self.assertEqual(report["duplicate_canonical_smiles"], 1)
        self.assertEqual(report["missing_1h"], 1)
        self.assertEqual(report["missing_13c"], 1)


if __name__ == "__main__":
    unittest.main()
