"""Tests for scalable paired JSONL dataset construction."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from script.build_paired_jsonl import build_jsonl_from_index, index_raw_csv


def _raw_row(smiles: str, nmr_type: str, processed: str, solvent: str = "CDCl3") -> dict:
    return {
        "Filename": "paper-a",
        "SMILES": smiles,
        "Page_in_file_mol": 1,
        "Page_in_file_para": 2,
        "Location_in_page_mol": "[0,0,1,1]",
        "Location_in_page_para": "[0,0,1,1]",
        "NMR_type": nmr_type,
        "NMR_frequency": "400 MHz" if nmr_type == "1H NMR" else "101 MHz",
        "NMR_solvent": solvent,
        "NMR_processed": processed,
        "Atom_number": 3,
        "Atom_number_diff_env": 2,
        "Atom_number_abstract": 3,
    }


def test_build_paired_jsonl_end_to_end(tmp_path: Path) -> None:
    """Small CSV input should produce paired JSONL and split manifest files."""
    csv_path = tmp_path / "raw.csv"
    db_path = tmp_path / "candidates.sqlite"
    out_dir = tmp_path / "paired"
    rows = [
        _raw_row("CCO", "13C NMR", "[(58.1,), (18.2,)]"),
        _raw_row("OCC", "1H NMR", "[(1.18, 't', ['7.0Hz'], '3H')]"),
        _raw_row("CCN", "13C NMR", "[(45.0,), (18.0,)]"),
        _raw_row("NCC", "1H NMR", "[(1.10, 't', ['7.0Hz'], '3H')]"),
        _raw_row("CCCl", "19F NMR", "[]"),
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    index_report = index_raw_csv(csv_path, db_path, chunksize=2)
    summary = build_jsonl_from_index(
        db_path,
        out_dir,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=1,
    )

    samples = [
        json.loads(line)
        for line in (out_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert index_report["candidate_rows"] == 4
    assert index_report["unsupported_nmr_rows"] == 1
    assert summary["jsonl_samples_written"] == 2
    assert {sample["canonical_smiles"] for sample in samples} == {"CCO", "CCN"}
    assert (out_dir / "manifest.csv").exists()
    assert (out_dir / "train_ids.txt").exists()
