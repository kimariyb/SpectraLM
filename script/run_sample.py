"""Cluster representative subset construction with MiniBatchKMeans clustering."""

from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Allow running from project root without PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.data.clustering import (
    cluster_samples,
    elbow_n_clusters,
    pca_reduce,
    plot_tsne_selection,
    sanitise_fingerprints,
    tsne_project,
)
from src.data.molecules import functional_group_labels, heavy_atom_count, murcko_scaffold, sample_smiles
from src.io import load_pickle_list, write_json, write_pickle, write_rows_csv
from script.run_featurize import build_sample_features, load_feature_outputs, save_feature_outputs


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
    """Container for cluster representative sampling outputs.

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


def candidate_rows(
    samples: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    labels: np.ndarray,
) -> list[dict[str, Any]]:
    """Join samples, feature metadata, and cluster labels.

    Parameters
    ----------
    samples
        Original samples.
    feature_rows
        Feature metadata rows.
    labels
        Cluster labels (one per feature row).

    Returns
    -------
    list[dict[str, Any]]
        Candidate rows with ``sample``, ``row``, and ``cluster`` keys.
    """
    sample_by_id = {sample.get("id"): sample for sample in samples}
    candidates: list[dict[str, Any]] = []
    for idx, row in enumerate(feature_rows):
        sample = sample_by_id.get(row.get("id"))
        if sample is None:
            continue
        candidates.append({"sample": sample, "row": row, "cluster": int(labels[idx])})
    return candidates


def sort_candidates(candidates: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Sort candidates by cluster coverage with random intra-cluster order.

    Large clusters are prioritised; within a cluster candidates are ordered
    randomly so both simple and complex molecules have equal probability.

    Parameters
    ----------
    candidates
        Candidate rows.
    seed
        Random seed.

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
                pair[0],
            ),
        )
    ]


def filter_candidates(
    candidates: list[dict[str, Any]],
    max_heavy_atoms: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Exclude candidates whose heavy-atom count exceeds *max_heavy_atoms*.

    Parameters
    ----------
    candidates
        Candidate rows.
    max_heavy_atoms
        Maximum heavy atom count (``None`` disables the filter).

    Returns
    -------
    tuple[list[dict[str, Any]], dict[str, int]]
        Filtered candidates and filter-count summary.
    """
    if max_heavy_atoms is None:
        return candidates, {"filtered_by_max_heavy_atoms": 0}

    kept: list[dict[str, Any]] = []
    filtered = 0
    for item in candidates:
        smiles = item["row"].get("canonical_smiles") or sample_smiles(item["sample"])
        if heavy_atom_count(smiles) > max_heavy_atoms:
            filtered += 1
            continue
        kept.append(item)
    return kept, {"filtered_by_max_heavy_atoms": filtered}


def select_one_split(
    candidates: list[dict[str, Any]],
    target_size: int,
    max_per_scaffold: int,
    blocked_scaffolds: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Select *target_size* representatives via two-pass greedy.

    **Pass 1 (new_cluster):** pick at most one molecule per cluster
    (guarantees cluster coverage).  **Pass 2 (fill):** fill remaining
    slots regardless of cluster.

    Both passes respect ``max_per_scaffold`` and disallow duplicate
    scaffolds / SMILES.

    Parameters
    ----------
    candidates
        Sorted candidate rows.
    target_size
        Desired number of selected items.
    max_per_scaffold
        Maximum items allowed per Murcko scaffold.
    blocked_scaffolds
        Scaffolds that must not be selected.

    Returns
    -------
    list[dict[str, Any]]
        Selected candidate rows (may be fewer than *target_size* if
        the candidate pool is exhausted).
    """
    blocked = blocked_scaffolds or set()
    selected: list[dict[str, Any]] = []
    scaffold_counts: Counter[str] = Counter()
    seen_molecules: set[str] = set()
    seen_clusters: set[int] = set()

    for pass_name in ("new_cluster", "fill"):
        for item in candidates:
            if len(selected) >= target_size:
                return selected

            row = item["row"]
            scaffold = row.get("murcko_scaffold", "")
            smiles = row.get("canonical_smiles", "")
            cluster = item["cluster"]

            if (
                scaffold in blocked
                or scaffold_counts[scaffold] >= max_per_scaffold
                or smiles in seen_molecules
            ):
                continue
            if pass_name == "new_cluster" and cluster in seen_clusters:
                continue

            selected.append(item)
            scaffold_counts[scaffold] += 1
            seen_molecules.add(smiles)
            seen_clusters.add(cluster)

    return selected


def materialize_samples(items: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    """Convert candidate items into full sample dictionaries.

    Parameters
    ----------
    items
        Selected candidate rows.
    split_name
        Split label (``"train"`` or ``"test"``).

    Returns
    -------
    list[dict[str, Any]]
        Sample dictionaries with ``split``, ``cluster``, and molecule
        annotations filled in.
    """
    output: list[dict[str, Any]] = []
    for item in items:
        row = item["row"]
        sample = dict(item["sample"])
        sample["split"] = split_name
        sample["cluster"] = item["cluster"]
        sample["canonical_smiles"] = row.get("canonical_smiles", sample.get("canonical_smiles", ""))
        sample["murcko_scaffold"] = row.get("murcko_scaffold") or murcko_scaffold(sample_smiles(sample))
        sample["functional_groups"] = row.get("functional_groups") or functional_group_labels(
            sample_smiles(sample)
        )
        output.append(sample)
    return output


def selected_csv_rows(items: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    """Convert candidate items to CSV-ready row dictionaries.

    Parameters
    ----------
    items
        Selected candidate rows.
    split_name
        Split label.

    Returns
    -------
    list[dict[str, Any]]
        CSV rows with semicolon-joined functional groups.
    """
    rows: list[dict[str, Any]] = []
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
    """Build summary statistics for a train / test split.

    Parameters
    ----------
    train
        Training samples.
    test
        Test samples.

    Returns
    -------
    dict[str, Any]
        Per-split sample / scaffold / cluster counts.
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
    """Select train / test representatives from KMeans clusters.

    Diversity is enforced at three levels:

    1. **Cluster coverage** — one molecule per cluster in the first pass.
    2. **Scaffold disjointness** — at most ``max_per_scaffold`` molecules
       per Murcko scaffold; train and test scaffolds are disjoint.
    3. **Molecular uniqueness** — no duplicate canonical SMILES in a split.

    Parameters
    ----------
    samples
        Original samples.
    feature_rows
        Feature metadata rows aligned with *fingerprints*.
    labels
        Cluster label per fingerprint row.
    fingerprints
        Fingerprint matrix (unused; kept for API compatibility).
    config
        Selection configuration (``train_size``, ``test_size``,
        ``max_per_scaffold``, ``max_heavy_atoms``, ``seed``).

    Returns
    -------
    ClusterSelection
        Selected splits and report.
    """
    _ = fingerprints  # kept for API compatibility
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
    test_items = select_one_split(
        remaining, test_size, max_per_scaffold, blocked_scaffolds=train_scaffolds,
    )

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


def run(config: dict[str, Any]) -> None:
    """Run MiniBatchKMeans clustering → TSNE → representative sampling.

    Parameters
    ----------
    config
        Configuration dictionary (see ``configs/sample.yaml``).
    """
    dataset = config.get("dataset", "dataset/NMRexp_spectra_dataset.pkl")
    out_dir = Path(config.get("out_dir", "dataset/subsets/"))
    fingerprint_path = config.get("fingerprints", "dataset/features/morgan_fingerprints.npz")
    index_path = config.get("index", "dataset/features/morgan_feature_index.csv")

    samples = load_pickle_list(dataset)

    # --- Fingerprints -------------------------------------------------------
    if Path(fingerprint_path).exists() and Path(index_path).exists():
        print(f"Loading cached fingerprints from {fingerprint_path}")
        fingerprints, feature_rows = load_feature_outputs(fingerprint_path, index_path)
    else:
        print("Computing fingerprints ...")
        fingerprints, feature_rows = build_sample_features(samples, config)
        save_feature_outputs(fingerprints, feature_rows, fingerprint_path, index_path)

    # --- Clustering ---------------------------------------------------------
    fingerprints, feature_rows = sanitise_fingerprints(fingerprints, feature_rows)

    # Elbow analysis mode: scan k values and exit
    suggest_k = config.get("suggest_n_clusters")
    if suggest_k:
        elbow_n_clusters(
            fingerprints,
            pca_components=int(config.get("pca_components", 50)),
            batch_size=int(config.get("batch_size", 10000)),
            seed=int(config.get("seed", 3407)),
            output_path=config.get("elbow_output", "img/elbow.png"),
        )
        return

    result = cluster_samples(fingerprints, config, feature_rows)

    # --- Representative sampling --------------------------------------------
    selection = select_cluster_representatives(
        samples, feature_rows, result.labels, fingerprints, config,
    )

    # --- TSNE visualisation -------------------------------------------------
    tsne_output = config.get("tsne_output")
    if tsne_output:
        pca_components = int(config.get("pca_components", 50))
        seed = int(config.get("seed", 3407))
        reduced, _ = pca_reduce(fingerprints, n_components=pca_components, random_state=seed)
        coords = tsne_project(reduced, random_state=seed)

        # Map selected sample ids → feature row indices
        id_to_row = {row["id"]: int(row["row_index"]) for row in feature_rows}
        train_indices = [id_to_row[s["id"]] for s in selection.train if s["id"] in id_to_row]
        test_indices = [id_to_row[s["id"]] for s in selection.test if s["id"] in id_to_row]

        plot_tsne_selection(coords, train_indices, test_indices, tsne_output)
        print(f"Wrote TSNE plot to {tsne_output}")
    write_pickle(out_dir / "train.pkl", selection.train)
    write_pickle(out_dir / "test.pkl", selection.test)
    write_rows_csv(out_dir / "selected_samples.csv", selection.selected_rows, SELECTED_FIELDS)

    # --- Report -------------------------------------------------------------
    report = dict(selection.report)
    report["method"] = result.method
    report["n_clusters"] = result.n_clusters
    report["pca_variance"] = result.pca_variance
    report["inertia"] = result.inertia
    write_json(out_dir / "cluster_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script/run_sample.py <config.yaml>")
        sys.exit(1)
    run(load_config(sys.argv[1]))
