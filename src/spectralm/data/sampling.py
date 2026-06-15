"""Butina cluster representative subset construction for SpectraLM."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from spectralm.config import add_config_argument, load_config
from spectralm.data.clustering import cluster_samples
from spectralm.data.features import build_sample_features, save_feature_outputs
from spectralm.data.molecules import functional_group_labels, heavy_atom_count, murcko_scaffold, sample_smiles
from spectralm.io import load_pickle_list, write_json, write_pickle, write_rows_csv


SELECTED_FIELDS = [
    "split",
    "id",
    "cluster",
    "canonical_smiles",
    "murcko_scaffold",
    "functional_groups",
]


@dataclass
class ClusterSelection:
    """Container for Butina representative sampling outputs.

    Parameters
    ----------
    train
        Selected training samples.
    test
        Selected test samples.
    report
        Sampling report.
    selected_rows
        CSV-ready selected metadata rows.
    """

    train: list[dict[str, Any]]
    test: list[dict[str, Any]]
    report: dict[str, Any]
    selected_rows: list[dict[str, Any]]


def row_quality_score(row: dict[str, Any]) -> tuple[int, int]:
    """Score a structure row by coarse chemical annotation richness.

    Parameters
    ----------
    row
        Feature metadata row.

    Returns
    -------
    tuple[int, int]
        Higher-is-better quality score.
    """
    groups = row.get("functional_groups", [])
    useful_groups = [group for group in groups if group not in {"invalid", "none_detected"}]
    return (len(useful_groups), len(row.get("canonical_smiles", "")))


def candidate_rows(
    samples: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    labels: np.ndarray,
) -> list[dict[str, Any]]:
    """Join samples, feature metadata, and Butina labels.

    Parameters
    ----------
    samples
        Original samples.
    feature_rows
        Feature metadata rows.
    labels
        Butina cluster labels.

    Returns
    -------
    list[dict[str, Any]]
        Candidate rows with sample payloads.
    """
    sample_by_id = {sample.get("id"): sample for sample in samples}
    candidates = []
    for idx, row in enumerate(feature_rows):
        sample = sample_by_id.get(row.get("id"))
        if sample is None:
            continue
        candidates.append({"sample": sample, "row": row, "cluster": int(labels[idx])})
    return candidates


def sort_candidates(candidates: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Sort candidates by Butina cluster coverage and structure richness.

    Parameters
    ----------
    candidates
        Candidate rows.
    seed
        Random seed used for deterministic tie-breaking.

    Returns
    -------
    list[dict[str, Any]]
        Sorted candidates.
    """
    rng = random.Random(seed)
    decorated = [(rng.random(), item) for item in candidates]
    cluster_counts = Counter(item["cluster"] for item in candidates)
    return [
        item
        for _, item in sorted(
            decorated,
            key=lambda pair: (
                -cluster_counts[pair[1]["cluster"]],
                pair[1]["cluster"],
                tuple(-part for part in row_quality_score(pair[1]["row"])),
                pair[0],
            ),
        )
    ]


def filter_candidates(
    candidates: list[dict[str, Any]],
    max_heavy_atoms: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Filter candidate rows by molecule-level constraints.

    Parameters
    ----------
    candidates
        Candidate rows.
    max_heavy_atoms
        Optional maximum heavy atom count.

    Returns
    -------
    tuple[list[dict[str, Any]], dict[str, int]]
        Filtered candidates and filter counts.
    """
    if max_heavy_atoms is None:
        return candidates, {"filtered_by_max_heavy_atoms": 0}
    kept = []
    filtered_by_heavy_atoms = 0
    for item in candidates:
        smiles = item["row"].get("canonical_smiles") or sample_smiles(item["sample"])
        if heavy_atom_count(smiles) > max_heavy_atoms:
            filtered_by_heavy_atoms += 1
            continue
        kept.append(item)
    return kept, {"filtered_by_max_heavy_atoms": filtered_by_heavy_atoms}


def select_one_split(
    candidates: list[dict[str, Any]],
    target_size: int,
    max_per_scaffold: int,
    blocked_scaffolds: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Select representatives for one split.

    Parameters
    ----------
    candidates
        Sorted candidate rows.
    target_size
        Desired split size.
    max_per_scaffold
        Maximum samples per scaffold.
    blocked_scaffolds
        Scaffolds that cannot be selected.

    Returns
    -------
    list[dict[str, Any]]
        Selected candidate rows.
    """
    blocked = blocked_scaffolds or set()
    selected = []
    scaffold_counts = Counter()
    seen_molecules = set()
    seen_clusters = set()
    for pass_name in ("new_cluster", "fill"):
        for item in candidates:
            if len(selected) >= target_size:
                return selected
            row = item["row"]
            scaffold = row.get("murcko_scaffold", "")
            smiles = row.get("canonical_smiles", "")
            if scaffold in blocked or scaffold_counts[scaffold] >= max_per_scaffold or smiles in seen_molecules:
                continue
            if pass_name == "new_cluster" and item["cluster"] in seen_clusters:
                continue
            selected.append(item)
            scaffold_counts[scaffold] += 1
            seen_molecules.add(smiles)
            seen_clusters.add(item["cluster"])
    return selected


def materialize_samples(items: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    """Convert selected candidates to sample dictionaries.

    Parameters
    ----------
    items
        Selected candidate rows.
    split_name
        Split name to assign.

    Returns
    -------
    list[dict[str, Any]]
        Selected sample dictionaries.
    """
    output = []
    for item in items:
        row = item["row"]
        sample = dict(item["sample"])
        sample["split"] = split_name
        sample["cluster"] = item["cluster"]
        sample["canonical_smiles"] = row.get("canonical_smiles", sample.get("canonical_smiles", ""))
        sample["murcko_scaffold"] = row.get("murcko_scaffold") or murcko_scaffold(sample_smiles(sample))
        sample["functional_groups"] = row.get("functional_groups") or functional_group_labels(sample_smiles(sample))
        output.append(sample)
    return output


def selected_csv_rows(items: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    """Convert selected candidate items to CSV rows.

    Parameters
    ----------
    items
        Selected candidate rows.
    split_name
        Split name.

    Returns
    -------
    list[dict[str, Any]]
        CSV-ready rows.
    """
    rows = []
    for item in items:
        row = item["row"]
        rows.append(
            {
                "split": split_name,
                "id": row.get("id", ""),
                "cluster": item["cluster"],
                "canonical_smiles": row.get("canonical_smiles", ""),
                "murcko_scaffold": row.get("murcko_scaffold", ""),
                "functional_groups": ";".join(row.get("functional_groups", [])),
            }
        )
    return rows


def split_report(train: list[dict[str, Any]], test: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a report for Butina representative subsets.

    Parameters
    ----------
    train
        Training samples.
    test
        Test samples.

    Returns
    -------
    dict[str, Any]
        Sampling report.
    """
    train_scaffolds = {row.get("murcko_scaffold") for row in train}
    test_scaffolds = {row.get("murcko_scaffold") for row in test}
    return {
        "train": {
            "samples": len(train),
            "unique_scaffolds": len(train_scaffolds),
            "unique_clusters": len({row.get("cluster") for row in train}),
        },
        "test": {
            "samples": len(test),
            "unique_scaffolds": len(test_scaffolds),
            "unique_clusters": len({row.get("cluster") for row in test}),
        },
        "scaffold_overlap_train_test": len(train_scaffolds & test_scaffolds),
    }


def select_cluster_representatives(
    samples: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    labels: np.ndarray,
    fingerprints: np.ndarray,
    config: dict[str, Any] | None = None,
) -> ClusterSelection:
    """Select train and test representatives from Butina clusters.

    Parameters
    ----------
    samples
        Original samples.
    feature_rows
        Feature metadata rows matching ``fingerprints``.
    labels
        Butina cluster labels matching ``fingerprints``.
    fingerprints
        Morgan fingerprint matrix. Included to keep the public API explicit.
    config
        Selection configuration.

    Returns
    -------
    ClusterSelection
        Selected splits and report.
    """
    _ = fingerprints
    cfg = config or {}
    train_size = int(cfg.get("train_size", 1000))
    test_size = int(cfg.get("test_size", 300))
    max_per_scaffold = int(cfg.get("max_per_scaffold", 1))
    max_heavy_atoms = cfg.get("max_heavy_atoms")
    max_heavy_atoms = int(max_heavy_atoms) if max_heavy_atoms is not None else None
    seed = int(cfg.get("seed", 3407))
    candidates, filter_report = filter_candidates(
        candidate_rows(samples, feature_rows, labels),
        max_heavy_atoms=max_heavy_atoms,
    )
    candidates = sort_candidates(candidates, seed)
    train_items = select_one_split(candidates, train_size, max_per_scaffold)
    train_scaffolds = {item["row"].get("murcko_scaffold", "") for item in train_items}
    remaining = [item for item in candidates if item not in train_items]
    test_items = select_one_split(remaining, test_size, max_per_scaffold, blocked_scaffolds=train_scaffolds)
    train = materialize_samples(train_items, "train")
    test = materialize_samples(test_items, "test")
    report = split_report(train, test)
    report.update(
        {
            "target_train_size": train_size,
            "target_test_size": test_size,
            "candidate_samples": len(candidates),
            "clusters": len(set(int(item) for item in labels)),
            "max_per_scaffold": max_per_scaffold,
            "max_heavy_atoms": max_heavy_atoms,
        }
    )
    report.update(filter_report)
    rows = selected_csv_rows(train_items, "train") + selected_csv_rows(test_items, "test")
    return ClusterSelection(train=train, test=test, report=report, selected_rows=rows)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the Butina sampling CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(description="Build Morgan FP + Butina representative subsets.")
    add_config_argument(parser)
    parser.add_argument("--dataset", default=None, help="Input sample pickle path.")
    parser.add_argument("--out-dir", default=None, help="Output subset directory.")
    parser.add_argument("--train-size", type=int, default=None, help="Training subset size.")
    parser.add_argument("--test-size", type=int, default=None, help="Test subset size.")
    parser.add_argument("--butina-cutoff", type=float, default=None, help="Tanimoto similarity cutoff.")
    parser.add_argument("--bucketed", action="store_true", default=None, help="Enable scaffold-stratified bucketed Butina.")
    parser.add_argument("--max-bucket-size", type=int, default=None, help="Maximum molecules per Butina bucket.")
    parser.add_argument("--max-heavy-atoms", type=int, default=None, help="Maximum allowed heavy atoms per molecule.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    return parser


def resolved_config(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    """Merge CLI overrides into a config dictionary.

    Parameters
    ----------
    args
        Parsed arguments.
    config
        YAML configuration.

    Returns
    -------
    dict[str, Any]
        Resolved configuration.
    """
    merged = dict(config)
    for key in (
        "dataset",
        "out_dir",
        "train_size",
        "test_size",
        "butina_cutoff",
        "bucketed",
        "max_bucket_size",
        "max_heavy_atoms",
        "seed",
    ):
        value = getattr(args, key)
        if value is not None:
            merged[key] = value
    return merged


def main(argv: list[str] | None = None) -> None:
    """Run Butina representative sampling from the command line.

    Parameters
    ----------
    argv
        Optional command-line argument list.
    """
    args = build_arg_parser().parse_args(argv)
    config = resolved_config(args, load_config(args.config))
    dataset = config.get("dataset", "dataset/NMRexp_spectra_dataset.pkl")
    out_dir = Path(config.get("out_dir", "dataset/subsets/spectralm_butina_1000_300"))
    samples = load_pickle_list(dataset)
    fingerprints, feature_rows = build_sample_features(samples, config)
    fingerprint_path = config.get("fingerprints", "dataset/features/morgan_fingerprints.npz")
    index_path = config.get("index", "dataset/features/morgan_feature_index.csv")
    save_feature_outputs(fingerprints, feature_rows, fingerprint_path, index_path)
    result = cluster_samples(fingerprints, config, feature_rows)
    selection = select_cluster_representatives(samples, feature_rows, result.labels, fingerprints, config)
    write_pickle(out_dir / "train.pkl", selection.train)
    write_pickle(out_dir / "test.pkl", selection.test)
    write_rows_csv(out_dir / "selected_samples.csv", selection.selected_rows, SELECTED_FIELDS)
    report = dict(selection.report)
    report["method"] = result.method
    report["butina_cutoff"] = result.cutoff
    report["bucket_count"] = result.bucket_count
    report["max_bucket_size"] = max(result.bucket_sizes or [0])
    write_json(out_dir / "cluster_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
