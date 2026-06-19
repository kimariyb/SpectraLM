"""Tests for raw CSV preprocessing into paired NMR samples."""

from __future__ import annotations

import pandas as pd

from script.run_preprocess import merge_1h_13c_rows, row_to_spectra


def _row(smiles: str, nmr_type: str, processed: str, solvent: str = "CDCl3") -> dict:
    return {
        "Filename": "paper-a",
        "SMILES": smiles,
        "Page_in_file_mol": 1,
        "Page_in_file_para": 2,
        "Location_in_page_mol": "[0, 0, 1, 1]",
        "Location_in_page_para": "[0, 0, 1, 1]",
        "NMR_type": nmr_type,
        "NMR_frequency": "400 MHz" if nmr_type == "1H NMR" else "101 MHz",
        "NMR_solvent": solvent,
        "NMR_shift_text": "",
        "NMR_note": "",
        "NMR_processed": processed,
        "Atom_number": 3,
        "Atom_number_diff_env": 2,
        "Atom_number_abstract": 3,
    }


def test_merge_pairs_equivalent_canonical_smiles() -> None:
    """Equivalent raw SMILES strings should still pair via canonical SMILES."""
    df = pd.DataFrame(
        [
            _row("CCO", "13C NMR", "[(58.1,), (18.2,)]"),
            _row("OCC", "1H NMR", "[(1.18, 't', ['7.0Hz'], '3H')]"),
        ]
    )

    merged = merge_1h_13c_rows(df)

    assert len(merged) == 1
    assert merged.iloc[0]["canonical_smiles"] == "CCO"


def test_row_to_spectra_uses_stable_id() -> None:
    """The same merged row should produce the same deterministic sample id."""
    df = pd.DataFrame(
        [
            _row("CCO", "13C NMR", "[(58.1,), (18.2,)]"),
            _row("OCC", "1H NMR", "[(1.18, 't', ['7.0Hz'], '3H')]"),
        ]
    )
    row = merge_1h_13c_rows(df).iloc[0]

    sample_a = row_to_spectra(row)
    sample_b = row_to_spectra(row)

    assert sample_a is not None
    assert sample_b is not None
    assert sample_a["id"] == sample_b["id"]
    assert sample_a["canonical_smiles"] == "CCO"
    assert sample_a["meta"]["source_13c"]["filename"] == "paper-a"
