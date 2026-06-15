"""Representative subset sampling for small SpectraLM pilot runs."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from spectralm.config import add_config_argument, load_config
from spectralm.data.molecules import (
    functional_group_labels,
    heavy_atom_count,
    molecular_weight,
    ring_count,
    sample_smiles,
)
from spectralm.data.nmr import peak_count
from spectralm.io import load_pickle_list, write_json, write_pickle, write_rows_csv


def bin_value(value: int | float, low: int | float, high: int | float) -> str:
    """Assign a numeric value to a coarse low/mid/high bin.

    Parameters
    ----------
    value
        Numeric value to bin.
    low
        Upper bound for the low bin.
    high
        Upper bound for the mid bin.

    Returns
    -------
    str
        ``low``, ``mid``, or ``high``.
    """
    if value <= low:
        return "low"
    if value <= high:
        return "mid"
    return "high"


def assign_complexity_bin(row: dict[str, Any]) -> str:
    """Assign a molecule and spectrum complexity bin.

    Parameters
    ----------
    row
        Descriptor-enriched sample row.

    Returns
    -------
    str
        Composite complexity label.
    """
    smiles = row.get("canonical_smiles") or sample_smiles(row)
    return "|".join(
        [
            f"heavy_atoms:{bin_value(heavy_atom_count(smiles), 20, 40)}",
            f"rings:{bin_value(ring_count(smiles), 0, 2)}",
            f"h_peaks:{bin_value(peak_count(row, '1H_NMR'), 6, 14)}",
            f"c_peaks:{bin_value(peak_count(row, '13C_NMR'), 10, 25)}",
        ]
    )


def enrich_row(row: dict[str, Any]) -> dict[str, Any]:
    """Attach sampling metadata to a row.

    Parameters
    ----------
    row
        Sample row with split metadata.

    Returns
    -------
    dict[str, Any]
        Enriched row.
    """
    out = dict(row)
    smiles = out.get("canonical_smiles") or sample_smiles(out)
    out["functional_groups"] = functional_group_labels(smiles)
    out["complexity_bin"] = assign_complexity_bin(out)
    out["heavy_atoms"] = heavy_atom_count(smiles)
    out["mol_weight"] = molecular_weight(smiles)
    out["h_peak_count"] = peak_count(out, "1H_NMR")
    out["c_peak_count"] = peak_count(out, "13C_NMR")
    return out


def coverage_gain(row: dict[str, Any], covered: Counter) -> int:
    """Measure how many new coverage keys a row contributes.

    Parameters
    ----------
    row
        Enriched sample row.
    covered
        Counter of already covered bins, functional groups, and scaffolds.

    Returns
    -------
    int
        Count of newly covered keys.
    """
    keys = [f"complexity:{row['complexity_bin']}", f"scaffold:{row['murcko_scaffold']}"]
    keys.extend(f"fg:{label}" for label in row["functional_groups"])
    return sum(1 for key in keys if covered[key] == 0)


def representative_sample(
    rows: list[dict[str, Any]],
    split_name: str,
    target_size: int,
    max_per_scaffold: int = 1,
    max_heavy_atoms: int | None = None,
    max_selfies_length: int | None = None,
    seed: int = 3407,
) -> list[dict[str, Any]]:
    """Select a representative sample subset from one split.

    Parameters
    ----------
    rows
        Candidate rows with split metadata.
    split_name
        Split to sample from.
    target_size
        Desired number of rows.
    max_per_scaffold
        Maximum rows per scaffold.
    max_heavy_atoms
        Optional heavy atom upper bound.
    max_selfies_length
        Optional SELFIES length upper bound.
    seed
        Random seed.

    Returns
    -------
    list[dict[str, Any]]
        Selected rows.
    """
    rng = random.Random(seed)
    candidates = []
    for row in rows:
        if row.get("split") != split_name:
            continue
        enriched = enrich_row(row)
        if max_heavy_atoms is not None and enriched["heavy_atoms"] > max_heavy_atoms:
            continue
        if max_selfies_length is not None and len(enriched.get("selfies", "")) > max_selfies_length:
            continue
        candidates.append(enriched)
    rng.shuffle(candidates)
    selected = []
    covered = Counter()
    scaffold_counts = Counter()
    molecule_seen = set()
    by_bin: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_bin[row["complexity_bin"]].append(row)
    bin_names = list(by_bin)
    rng.shuffle(bin_names)
    while by_bin and len(selected) < target_size:
        progressed = False
        for bin_name in list(bin_names):
            bucket = by_bin.get(bin_name, [])
            if not bucket:
                by_bin.pop(bin_name, None)
                if bin_name in bin_names:
                    bin_names.remove(bin_name)
                continue
            best_idx = None
            best_score = None
            for idx, row in enumerate(bucket):
                scaffold = row["murcko_scaffold"]
                smiles = row.get("canonical_smiles")
                if scaffold_counts[scaffold] >= max_per_scaffold or smiles in molecule_seen:
                    continue
                score = (
                    coverage_gain(row, covered),
                    len(row["functional_groups"]),
                    row["h_peak_count"] + row["c_peak_count"],
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is None:
                by_bin.pop(bin_name, None)
                if bin_name in bin_names:
                    bin_names.remove(bin_name)
                continue
            row = bucket.pop(best_idx)
            selected.append(row)
            scaffold_counts[row["murcko_scaffold"]] += 1
            molecule_seen.add(row.get("canonical_smiles"))
            covered[f"complexity:{row['complexity_bin']}"] += 1
            covered[f"scaffold:{row['murcko_scaffold']}"] += 1
            for label in row["functional_groups"]:
                covered[f"fg:{label}"] += 1
            progressed = True
            if len(selected) >= target_size:
                break
        if not progressed:
            break
    return selected


def load_split_csv(path: Path) -> dict[str, dict[str, str]]:
    """Load scaffold split metadata by sample ID.

    Parameters
    ----------
    path
        Split CSV path.

    Returns
    -------
    dict[str, dict[str, str]]
        Split rows keyed by sample ID.
    """
    by_id = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_id[row["id"]] = row
    return by_id


def attach_split_metadata(
    samples: list[dict[str, Any]],
    split_rows: dict[str, dict[str, str]],
    keep_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Attach split CSV metadata to sample dictionaries.

    Parameters
    ----------
    samples
        Full sample list.
    split_rows
        Split metadata keyed by sample ID.
    keep_ids
        Optional sample IDs to keep.

    Returns
    -------
    list[dict[str, Any]]
        Rows with split metadata.
    """
    rows = []
    for sample in samples:
        sample_id = sample.get("id")
        if keep_ids is not None and sample_id not in keep_ids:
            continue
        split_row = split_rows.get(sample_id)
        if not split_row:
            continue
        row = dict(sample)
        row.update(
            {
                "split": split_row["split"],
                "canonical_smiles": split_row["canonical_smiles"],
                "murcko_scaffold": split_row["murcko_scaffold"],
                "molecular_formula": split_row["molecular_formula"],
            }
        )
        rows.append(row)
    return rows


def preselect_split_ids(
    split_rows: dict[str, dict[str, str]],
    split_name: str,
    pool_size: int,
    max_per_scaffold: int,
    seed: int,
) -> set[str]:
    """Preselect candidate IDs without loading the full pickle payload.

    Parameters
    ----------
    split_rows
        Split metadata keyed by sample ID.
    split_name
        Split to sample from.
    pool_size
        Maximum number of candidate IDs.
    max_per_scaffold
        Maximum rows per scaffold.
    seed
        Random seed.

    Returns
    -------
    set[str]
        Candidate sample IDs.
    """
    rng = random.Random(seed)
    rows = [row for row in split_rows.values() if row.get("split") == split_name]
    rng.shuffle(rows)
    selected = []
    scaffold_counts = Counter()
    molecule_seen = set()
    covered_fg = set()
    for row in rows:
        smiles = row.get("canonical_smiles")
        scaffold = row.get("murcko_scaffold")
        if not smiles or not scaffold:
            continue
        if scaffold_counts[scaffold] >= max_per_scaffold or smiles in molecule_seen:
            continue
        functional_groups = functional_group_labels(smiles)
        if set(functional_groups).issubset(covered_fg):
            continue
        selected.append(row)
        scaffold_counts[scaffold] += 1
        molecule_seen.add(smiles)
        covered_fg.update(functional_groups)
        if len(selected) >= pool_size:
            return {item["id"] for item in selected}
    for row in rows:
        if len(selected) >= pool_size:
            break
        smiles = row.get("canonical_smiles")
        scaffold = row.get("murcko_scaffold")
        if not smiles or not scaffold:
            continue
        if scaffold_counts[scaffold] >= max_per_scaffold or smiles in molecule_seen:
            continue
        selected.append(row)
        scaffold_counts[scaffold] += 1
        molecule_seen.add(smiles)
    return {row["id"] for row in selected}


def sample_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize selected sample coverage.

    Parameters
    ----------
    rows
        Selected rows.

    Returns
    -------
    dict[str, Any]
        Summary statistics.
    """
    fg_counts = Counter()
    complexity_counts = Counter()
    scaffold_counts = Counter()
    for row in rows:
        fg_counts.update(row["functional_groups"])
        complexity_counts[row["complexity_bin"]] += 1
        scaffold_counts[row["murcko_scaffold"]] += 1
    return {
        "samples": len(rows),
        "unique_molecules": len({row.get("canonical_smiles") for row in rows}),
        "unique_scaffolds": len(scaffold_counts),
        "max_per_scaffold": max(scaffold_counts.values()) if scaffold_counts else 0,
        "functional_groups": fg_counts.most_common(),
        "complexity_bins": complexity_counts.most_common(),
        "avg_1h_peak_count": sum(row["h_peak_count"] for row in rows) / len(rows) if rows else 0,
        "avg_13c_peak_count": sum(row["c_peak_count"] for row in rows) / len(rows) if rows else 0,
        "avg_heavy_atoms": sum(row["heavy_atoms"] for row in rows) / len(rows) if rows else 0,
        "avg_mol_weight": sum(row["mol_weight"] for row in rows) / len(rows) if rows else 0,
    }


SAMPLE_CSV_FIELDS = [
    "split",
    "id",
    "canonical_smiles",
    "murcko_scaffold",
    "molecular_formula",
    "functional_groups",
    "complexity_bin",
    "heavy_atoms",
    "mol_weight",
    "h_peak_count",
    "c_peak_count",
]


def selected_sample_csv_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert selected rows to CSV row dictionaries.

    Parameters
    ----------
    rows
        Selected sample rows.

    Returns
    -------
    list[dict[str, Any]]
        CSV-ready row dictionaries.
    """
    return [
        {
            "split": row["split"],
            "id": row.get("id", ""),
            "canonical_smiles": row.get("canonical_smiles", ""),
            "murcko_scaffold": row.get("murcko_scaffold", ""),
            "molecular_formula": row.get("molecular_formula", ""),
            "functional_groups": ";".join(row["functional_groups"]),
            "complexity_bin": row["complexity_bin"],
            "heavy_atoms": row["heavy_atoms"],
            "mol_weight": f"{row['mol_weight']:.4f}",
            "h_peak_count": row["h_peak_count"],
            "c_peak_count": row["c_peak_count"],
        }
        for row in rows
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the representative sampling CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(description="Build a representative SpectraLM subset.")
    add_config_argument(parser)
    parser.add_argument("--dataset", default=None, help="Input sample pickle path.")
    parser.add_argument("--split-csv", default=None, help="Scaffold split CSV path.")
    parser.add_argument("--out-dir", default=None, help="Output subset directory.")
    parser.add_argument("--train-size", type=int, default=None, help="Training subset size.")
    parser.add_argument("--test-size", type=int, default=None, help="Test subset size.")
    parser.add_argument("--max-per-scaffold", type=int, default=None, help="Maximum rows per scaffold.")
    parser.add_argument("--max-heavy-atoms", type=int, default=None, help="Optional heavy atom limit.")
    parser.add_argument("--max-selfies-length", type=int, default=None, help="Optional SELFIES length limit.")
    parser.add_argument("--pool-multiplier", type=int, default=None, help="Candidate pool multiplier.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    return parser


def config_value(args: argparse.Namespace, config: dict[str, Any], name: str, default: Any) -> Any:
    """Return an argparse value with config and default fallback.

    Parameters
    ----------
    args
        Parsed CLI arguments.
    config
        Loaded configuration dictionary.
    name
        Argument and config key.
    default
        Fallback value.

    Returns
    -------
    Any
        Resolved value.
    """
    value = getattr(args, name)
    return value if value is not None else config.get(name, default)


def main() -> None:
    """Run representative sampling from the command line."""
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    dataset_path = config_value(args, config, "dataset", "dataset/NMRexp_spectra_dataset.pkl")
    split_csv = config_value(args, config, "split_csv", "dataset/splits/scaffold_split.csv")
    out_dir = Path(config_value(args, config, "out_dir", "dataset/subsets/spectralm_500_100"))
    train_size = int(config_value(args, config, "train_size", 500))
    test_size = int(config_value(args, config, "test_size", 100))
    max_per_scaffold = int(config_value(args, config, "max_per_scaffold", 1))
    max_heavy_atoms = config_value(args, config, "max_heavy_atoms", None)
    max_selfies_length = config_value(args, config, "max_selfies_length", None)
    pool_multiplier = int(config_value(args, config, "pool_multiplier", 20))
    seed = int(config_value(args, config, "seed", 3407))
    split_rows = load_split_csv(Path(split_csv))
    train_pool_size = train_size * pool_multiplier
    test_pool_size = test_size * pool_multiplier
    print(f"Preselecting {train_pool_size:,} train candidate IDs...", file=sys.stderr, flush=True)
    train_ids = preselect_split_ids(split_rows, "train", train_pool_size, max_per_scaffold, seed)
    print(f"Preselecting {test_pool_size:,} test candidate IDs...", file=sys.stderr, flush=True)
    test_ids = preselect_split_ids(split_rows, "test", test_pool_size, max_per_scaffold, seed + 1)
    keep_ids = train_ids | test_ids
    print(f"Loading dataset and attaching {len(keep_ids):,} candidate samples...", file=sys.stderr)
    rows = attach_split_metadata(load_pickle_list(Path(dataset_path)), split_rows, keep_ids=keep_ids)
    train = representative_sample(
        rows,
        split_name="train",
        target_size=train_size,
        max_per_scaffold=max_per_scaffold,
        max_heavy_atoms=max_heavy_atoms,
        max_selfies_length=max_selfies_length,
        seed=seed,
    )
    test = representative_sample(
        rows,
        split_name="test",
        target_size=test_size,
        max_per_scaffold=max_per_scaffold,
        max_heavy_atoms=max_heavy_atoms,
        max_selfies_length=max_selfies_length,
        seed=seed + 1,
    )
    write_pickle(out_dir / "train.pkl", train)
    write_pickle(out_dir / "test.pkl", test)
    write_rows_csv(out_dir / "selected_samples.csv", selected_sample_csv_rows(train + test), SAMPLE_CSV_FIELDS)
    report = {
        "train": sample_report(train),
        "test": sample_report(test),
        "scaffold_overlap_train_test": len(
            {row["murcko_scaffold"] for row in train} & {row["murcko_scaffold"] for row in test}
        ),
    }
    write_json(out_dir / "sample_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()

