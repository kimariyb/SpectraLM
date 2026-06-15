"""Preprocess raw NMR CSV rows into normalized SpectraLM samples."""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from spectralm.config import add_config_argument, load_config
from spectralm.data.molecules import canonicalize_smiles, smiles_to_selfies
from spectralm.data.nmr import process_13c_peaks, process_1h_peaks, safe_literal_eval
from spectralm.io import write_pickle


def row_to_spectra(row: pd.Series) -> dict[str, Any] | None:
    """Convert a merged 1H/13C row into a normalized sample.

    Parameters
    ----------
    row
        Merged pandas row containing molecule and NMR columns.

    Returns
    -------
    dict[str, Any] | None
        Normalized sample, or ``None`` when structure encoding fails.
    """
    smiles = row["SMILES"]
    canonical_smiles = canonicalize_smiles(smiles)
    if canonical_smiles is None:
        return None
    selfies = smiles_to_selfies(canonical_smiles)
    if selfies is None:
        return None
    c_raw = safe_literal_eval(row["NMR_processed_13C"])
    h_raw = safe_literal_eval(row["NMR_processed_1H"])
    return {
        "id": str(uuid.uuid4()),
        "smiles": smiles,
        "canonical_smiles": canonical_smiles,
        "selfies": selfies,
        "meta": {"source": "experimental"},
        "13C_NMR": {
            "frequency": row.get("NMR_frequency_13C"),
            "solvent": row.get("NMR_solvent_13C"),
            "peaks": process_13c_peaks(c_raw),
        },
        "1H_NMR": {
            "frequency": row.get("NMR_frequency_1H"),
            "solvent": row.get("NMR_solvent_1H"),
            "peaks": process_1h_peaks(h_raw),
        },
        "spectrum": {"1H_image": None, "13C_image": None, "combined_image": None},
    }


def merge_1h_13c_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Merge raw 1H and 13C NMR rows by SMILES.

    Parameters
    ----------
    df
        Raw dataframe containing ``NMR_type`` and ``SMILES`` columns.

    Returns
    -------
    pandas.DataFrame
        One row per molecule with paired 1H and 13C data.
    """
    df_13c = df[df["NMR_type"] == "13C NMR"].copy().add_suffix("_13C")
    df_1h = df[df["NMR_type"] == "1H NMR"].copy().add_suffix("_1H")
    merged = pd.merge(df_13c, df_1h, left_on="SMILES_13C", right_on="SMILES_1H", how="inner")
    merged["solvent_match_priority"] = (
        merged["NMR_solvent_13C"] == merged["NMR_solvent_1H"]
    ).astype(int)
    merged = merged.sort_values(
        by=["SMILES_13C", "solvent_match_priority"],
        ascending=[True, False],
    )
    merged = merged.drop_duplicates(subset=["SMILES_13C"], keep="first")
    merged["SMILES"] = merged["SMILES_13C"]
    return merged.drop(["SMILES_13C", "SMILES_1H", "solvent_match_priority"], axis=1)


def build_spectra_dataset(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Build normalized paired-NMR samples from a raw dataframe.

    Parameters
    ----------
    df
        Raw dataframe loaded from the source CSV.

    Returns
    -------
    list[dict[str, Any]]
        Normalized SpectraLM samples.
    """
    dataset = []
    merged = merge_1h_13c_rows(df)
    for _, row in tqdm(merged.iterrows(), total=len(merged), desc="Processing rows"):
        try:
            sample = row_to_spectra(row)
        except Exception as exc:
            print(f"Skipping malformed row: {exc}")
            continue
        if sample is not None:
            dataset.append(sample)
    return dataset


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the preprocessing CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(description="Preprocess raw NMR CSV into SpectraLM samples.")
    add_config_argument(parser)
    parser.add_argument("--input-csv", default=None, help="Input raw CSV path.")
    parser.add_argument("--output-pickle", default=None, help="Output normalized pickle path.")
    return parser


def main() -> None:
    """Run the preprocessing workflow from the command line."""
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    input_csv = args.input_csv or config.get("input_csv", "dataset/NMRexp_10to24_1_1004.csv")
    output_pickle = args.output_pickle or config.get("output_pickle", "dataset/NMRexp_spectra_dataset.pkl")
    df = pd.read_csv(Path(input_csv))
    dataset = build_spectra_dataset(df)
    write_pickle(output_pickle, dataset)
    print(f"Wrote {output_pickle}; samples={len(dataset)}")


if __name__ == "__main__":
    main()

