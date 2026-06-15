"""Scaffold split and data-quality reporting for SpectraLM samples."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from spectralm.config import add_config_argument, load_config
from spectralm.data.molecules import molecule_descriptors, sample_smiles
from spectralm.data.nmr import peak_count
from spectralm.io import load_pickle_list, write_json, write_rows_csv


def is_complete_nmr_sample(sample: dict[str, Any]) -> bool:
    """Check whether a sample contains both 1H and 13C peaks.

    Parameters
    ----------
    sample
        Sample dictionary.

    Returns
    -------
    bool
        ``True`` when both peak tables are non-empty.
    """
    return peak_count(sample, "1H_NMR") > 0 and peak_count(sample, "13C_NMR") > 0


def analyze_samples(
    samples: list[dict[str, Any]],
    progress_every: int = 10000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Analyze sample quality and attach molecular descriptors.

    Parameters
    ----------
    samples
        Input sample dictionaries.
    progress_every
        Progress logging interval. Use zero to disable progress logging.

    Returns
    -------
    tuple[list[dict[str, Any]], dict[str, Any]]
        Valid descriptor-enriched rows and a quality report.
    """
    cache: dict[str, dict[str, str | None]] = {}
    valid_rows = []
    canonical_values = []
    scaffold_values = []
    h_counts = []
    c_counts = []
    missing_1h = 0
    missing_13c = 0
    total = len(samples)
    for idx, sample in enumerate(samples, 1):
        raw_smiles = sample_smiles(sample)
        cache_key = str(raw_smiles or "")
        if cache_key not in cache:
            cache[cache_key] = molecule_descriptors(raw_smiles)
        desc = cache[cache_key]
        h_count = peak_count(sample, "1H_NMR")
        c_count = peak_count(sample, "13C_NMR")
        h_counts.append(h_count)
        c_counts.append(c_count)
        missing_1h += h_count == 0
        missing_13c += c_count == 0
        canonical = desc["canonical_smiles"]
        if canonical is not None:
            canonical_values.append(canonical)
            scaffold_values.append(desc["murcko_scaffold"])
        if canonical is not None and h_count > 0 and c_count > 0:
            row = dict(sample)
            row.update(desc)
            valid_rows.append(row)
        if progress_every and (idx % progress_every == 0 or idx == total):
            print(
                f"Processed {idx:,}/{total:,} samples; valid complete rows: {len(valid_rows):,}",
                file=sys.stderr,
                flush=True,
            )
    report = {
        "total_samples": total,
        "valid_molecules": len(canonical_values),
        "invalid_molecules": total - len(canonical_values),
        "complete_1h_13c_samples": sum(1 for h, c in zip(h_counts, c_counts) if h > 0 and c > 0),
        "missing_1h": missing_1h,
        "missing_13c": missing_13c,
        "unique_canonical_smiles": len(set(canonical_values)),
        "duplicate_canonical_smiles": len(canonical_values) - len(set(canonical_values)),
        "unique_scaffolds": len(set(scaffold_values)),
        "top_scaffolds": Counter(scaffold_values).most_common(20),
        "avg_1h_peak_count": sum(h_counts) / len(h_counts) if h_counts else 0,
        "avg_13c_peak_count": sum(c_counts) / len(c_counts) if c_counts else 0,
    }
    return valid_rows, report


def valid_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return descriptor-enriched complete samples.

    Parameters
    ----------
    samples
        Input sample dictionaries.

    Returns
    -------
    list[dict[str, Any]]
        Complete samples with canonical SMILES and scaffold fields.
    """
    rows, _ = analyze_samples(samples, progress_every=0)
    return rows


def scaffold_split(
    samples: list[dict[str, Any]],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 3407,
) -> dict[str, list[dict[str, Any]]]:
    """Split samples by Bemis-Murcko scaffold.

    Parameters
    ----------
    samples
        Input samples, optionally already descriptor-enriched.
    ratios
        Train, validation, and test ratios that sum to one.
    seed
        Random seed for scaffold group order.

    Returns
    -------
    dict[str, list[dict[str, Any]]]
        Split rows keyed by ``train``, ``val``, and ``test``.
    """
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("ratios must contain train/val/test values that sum to 1.0")
    rows = samples if samples and "murcko_scaffold" in samples[0] else valid_samples(samples)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in rows:
        groups[sample["murcko_scaffold"]].append(sample)
    rng = random.Random(seed)
    grouped_rows = list(groups.values())
    rng.shuffle(grouped_rows)
    grouped_rows.sort(key=len, reverse=True)
    total = sum(len(group) for group in grouped_rows)
    targets = {"train": total * ratios[0], "val": total * ratios[1], "test": total * ratios[2]}
    split = {"train": [], "val": [], "test": []}
    for group in grouped_rows:
        name = min(split, key=lambda part: len(split[part]) / (targets[part] or 1))
        split[name].extend(group)
    return split


def build_quality_report(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a dataset quality report.

    Parameters
    ----------
    samples
        Input sample dictionaries.

    Returns
    -------
    dict[str, Any]
        Quality report.
    """
    _, report = analyze_samples(samples, progress_every=0)
    return report


SPLIT_CSV_FIELDS = ["split", "id", "canonical_smiles", "murcko_scaffold", "molecular_formula"]


def split_csv_rows(split: dict[str, list[dict[str, Any]]]) -> list[dict[str, str]]:
    """Convert split rows to CSV row dictionaries.

    Parameters
    ----------
    split
        Split dictionary.

    Returns
    -------
    list[dict[str, str]]
        CSV rows.
    """
    rows = []
    for name, split_rows in split.items():
        for row in split_rows:
            rows.append(
                {
                    "split": name,
                    "id": row.get("id", ""),
                    "canonical_smiles": row.get("canonical_smiles", ""),
                    "murcko_scaffold": row.get("murcko_scaffold", ""),
                    "molecular_formula": row.get("molecular_formula", ""),
                }
            )
    return rows


def split_summary(split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Summarize split sizes and uniqueness.

    Parameters
    ----------
    split
        Split dictionary.

    Returns
    -------
    dict[str, Any]
        Summary by split name.
    """
    return {
        name: {
            "samples": len(rows),
            "unique_scaffolds": len({row["murcko_scaffold"] for row in rows}),
            "unique_molecules": len({row["canonical_smiles"] for row in rows}),
        }
        for name, rows in split.items()
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the scaffold split CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(description="Build scaffold split and NMR quality report.")
    add_config_argument(parser)
    parser.add_argument("--input", default=None, help="Input sample pickle path.")
    parser.add_argument("--out-dir", default=None, help="Output split directory.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--ratios", nargs=3, type=float, default=None, help="Train val test ratios.")
    return parser


def main() -> None:
    """Run scaffold splitting from the command line."""
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    input_path = args.input or config.get("input", "dataset/NMRexp_spectra_dataset.pkl")
    out_dir = Path(args.out_dir or config.get("out_dir", "dataset/splits"))
    seed = args.seed if args.seed is not None else int(config.get("seed", 3407))
    ratios_raw = args.ratios if args.ratios is not None else config.get("ratios", [0.8, 0.1, 0.1])
    ratios = tuple(float(value) for value in ratios_raw)
    samples = load_pickle_list(input_path)
    valid_rows, quality = analyze_samples(samples)
    split = scaffold_split(valid_rows, ratios=ratios, seed=seed)
    quality["split_summary"] = split_summary(split)
    write_json(out_dir / "quality_report.json", quality)
    write_rows_csv(out_dir / "scaffold_split.csv", split_csv_rows(split), SPLIT_CSV_FIELDS)
    print(json.dumps(quality["split_summary"], ensure_ascii=False, indent=2))
    print(f"Wrote {out_dir / 'quality_report.json'}")
    print(f"Wrote {out_dir / 'scaffold_split.csv'}")


if __name__ == "__main__":
    main()

