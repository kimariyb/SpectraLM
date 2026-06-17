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
    selected
        All selected samples (single collection, split happens downstream).
    report
        Sampling report.
    selected_rows
        CSV-ready selected metadata rows.
    """

    selected: list[dict[str, Any]]
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


def _sample_from_cluster(
    cluster_items: list[dict[str, Any]],
    pca_coords: np.ndarray,
    n_pick: int,
) -> list[dict[str, Any]]:
    """Pick *n_pick* molecules from a single cluster.

    1. **Centroid** — molecule closest to the cluster mean in PCA space.
    2. **MaxMin** — remaining picks via manual MaxMin on Euclidean distance.

    Parameters
    ----------
    cluster_items
        Candidate items belonging to one cluster.
    pca_coords
        PCA-reduced matrix indexed by ``row_index``.
    n_pick
        Number of molecules to select.

    Returns
    -------
    list[dict[str, Any]]
        Selected items.
    """
    if not cluster_items or n_pick <= 0:
        return []

    n_pick = min(n_pick, len(cluster_items))
    ridx = np.array([int(c["row"]["row_index"]) for c in cluster_items])
    coords = pca_coords[ridx]  # (n_items, n_components)

    selected_pos: list[int] = []
    unselected = list(range(len(coords)))

    # --- Centroid pick ---
    centroid = coords.mean(axis=0, keepdims=True)
    dists_to_centroid = np.linalg.norm(coords - centroid, axis=1)
    first = int(np.argmin(dists_to_centroid))
    selected_pos.append(first)
    unselected.remove(first)

    # --- MaxMin picks ---
    unsel_coords = coords[unselected]
    sel_coords = coords[selected_pos]

    for _ in range(n_pick - 1):
        if not unselected:
            break
        # dists: (n_unselected, n_selected)
        dists = np.linalg.norm(
            unsel_coords[:, None, :] - sel_coords[None, :, :], axis=2,
        )
        min_dists = dists.min(axis=1)
        best = int(np.argmax(min_dists))
        selected_pos.append(unselected[best])
        unselected.pop(best)
        # Rebuild arrays for next iteration
        unsel_coords = coords[unselected]
        sel_coords = coords[selected_pos]

    return [cluster_items[p] for p in selected_pos]


def _sample_all_clusters(
    candidates: list[dict[str, Any]],
    pca_coords: np.ndarray,
    labels: np.ndarray,
    n_per_cluster: int,
    max_per_scaffold: int,
) -> list[dict[str, Any]]:
    """Sample *n_per_cluster* molecules from every cluster.

    Parameters
    ----------
    candidates
        All candidate rows.
    pca_coords
        PCA-reduced matrix indexed by ``row_index``.
    labels
        Cluster label per fingerprint row.
    n_per_cluster
        Number of molecules to pick per cluster.
    max_per_scaffold
        Max per scaffold (enforced post-hoc).

    Returns
    -------
    list[dict[str, Any]]
        Selected items across all clusters.
    """
    # Group candidates by cluster
    by_cluster: dict[int, list[dict[str, Any]]] = {}
    for item in candidates:
        by_cluster.setdefault(item["cluster"], []).append(item)

    selected: list[dict[str, Any]] = []
    scaffold_counts: Counter[str] = Counter()
    seen_molecules: set[str] = set()

    for cluster_id in sorted(by_cluster):
        items = by_cluster[cluster_id]

        eligible = []
        for item in items:
            row = item["row"]
            scaffold = row.get("murcko_scaffold", "")
            smiles = row.get("canonical_smiles", "")
            if scaffold_counts[scaffold] >= max_per_scaffold:
                continue
            if smiles in seen_molecules:
                continue
            eligible.append(item)

        picks = _sample_from_cluster(eligible, pca_coords, n_per_cluster)

        for item in picks:
            row = item["row"]
            scaffold_counts[row.get("murcko_scaffold", "")] += 1
            seen_molecules.add(row.get("canonical_smiles", ""))

        selected.extend(picks)

    return selected


def select_one_split(
    candidates: list[dict[str, Any]],
    target_size: int,
    max_per_scaffold: int,
    pca_coords: np.ndarray,
    seed: int,
    labels: np.ndarray,
) -> list[dict[str, Any]]:
    """Select *target_size* representatives.

    Each cluster contributes up to *n_per_cluster* molecules
    (centroid + MaxMin picks within that cluster).

    Parameters
    ----------
    candidates
        All candidate rows.
    target_size
        Desired number of items.
    max_per_scaffold
        Max per Murcko scaffold.
    pca_coords
        PCA-reduced matrix indexed by ``row_index``.
    seed
        Random seed.
    labels
        Cluster label per fingerprint row.

    Returns
    -------
    list[dict[str, Any]]
        Selected items.
    """
    n_clusters = int(labels.max()) + 1
    n_per_cluster = max(1, target_size // max(n_clusters, 1))

    result = _sample_all_clusters(
        candidates, pca_coords, labels, n_per_cluster, max_per_scaffold,
    )

    # If we fell short, run a second pass with doubled quota
    if len(result) < target_size:
        already_picked = {item["row"].get("id", "") for item in result}
        remaining = [c for c in candidates if c["row"].get("id", "") not in already_picked]
        extra = _sample_all_clusters(
            remaining, pca_coords, labels, max(1, n_per_cluster * 2), max_per_scaffold,
        )
        result.extend(extra[: target_size - len(result)])

    return result[:target_size]


def materialize_samples(items: list[dict[str, Any]], tag: str = "selected") -> list[dict[str, Any]]:
    """Convert candidate items into full sample dictionaries.

    Parameters
    ----------
    items
        Selected candidate rows.
    tag
        Label stored in the ``split`` field.

    Returns
    -------
    list[dict[str, Any]]
        Sample dictionaries with annotations filled in.
    """
    output: list[dict[str, Any]] = []
    for item in items:
        row = item["row"]
        sample = dict(item["sample"])
        sample["split"] = tag
        sample["cluster"] = item["cluster"]
        sample["canonical_smiles"] = row.get("canonical_smiles", sample.get("canonical_smiles", ""))
        sample["murcko_scaffold"] = row.get("murcko_scaffold") or murcko_scaffold(sample_smiles(sample))
        sample["functional_groups"] = row.get("functional_groups") or functional_group_labels(
            sample_smiles(sample)
        )
        output.append(sample)
    return output


def selected_csv_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert candidate items to CSV-ready row dictionaries.

    Parameters
    ----------
    items
        Selected candidate rows.

    Returns
    -------
    list[dict[str, Any]]
        CSV rows.
    """
    rows: list[dict[str, Any]] = []
    for item in items:
        row = item["row"]
        rows.append(
            {
                "id": row.get("id", ""),
                "cluster": item["cluster"],
                "canonical_smiles": row.get("canonical_smiles", ""),
                "murcko_scaffold": row.get("murcko_scaffold", ""),
                "functional_groups": ";".join(row.get("functional_groups", [])),
            }
        )
    return rows


def select_cluster_representatives(
    samples: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    labels: np.ndarray,
    fingerprints: np.ndarray,
    config: dict[str, Any] | None = None,
) -> ClusterSelection:
    """Select representatives from clusters.

    All candidates are sampled together into a single diverse set.

    Parameters
    ----------
    samples
        Original samples.
    feature_rows
        Feature metadata rows aligned with *fingerprints*.
    labels
        Cluster label per fingerprint row.
    fingerprints
        Fingerprint matrix (used for PCA).
    config
        Selection configuration (``sample_size``, ``max_per_scaffold``,
        ``max_heavy_atoms``, ``seed``).

    Returns
    -------
    ClusterSelection
        Selected samples and report.
    """
    cfg = config or {}
    seed = int(cfg.get("seed", 3407))

    # PCA reduction
    pca_components = int(cfg.get("pca_components", 50))
    pca_coords, _ = pca_reduce(fingerprints, n_components=pca_components, random_state=seed)

    sample_size = int(cfg.get("sample_size", 1200))
    max_per_scaffold = int(cfg.get("max_per_scaffold", 1))
    max_heavy_atoms = cfg.get("max_heavy_atoms")
    max_heavy_atoms = int(max_heavy_atoms) if max_heavy_atoms is not None else None

    candidates, filter_report = filter_candidates(
        candidate_rows(samples, feature_rows, labels),
        max_heavy_atoms=max_heavy_atoms,
    )
    candidates = sort_candidates(candidates, seed)

    all_picks = select_one_split(
        candidates, sample_size, max_per_scaffold, pca_coords, seed, labels,
    )

    selected = materialize_samples(all_picks, "selected")
    report = {
        "target_size": sample_size,
        "actual_size": len(selected),
        "candidate_samples": len(candidates),
        "clusters": len(set(int(item) for item in labels)),
        "max_per_scaffold": max_per_scaffold,
        "max_heavy_atoms": max_heavy_atoms,
    }
    report.update(filter_report)

    rows = selected_csv_rows(all_picks)
    return ClusterSelection(selected=selected, report=report, selected_rows=rows)


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
            pca_components=int(config.get("pca_components", 10)),
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
        sel_indices = [id_to_row[s["id"]] for s in selection.selected if s["id"] in id_to_row]
        plot_tsne_selection(coords, sel_indices, [], tsne_output,
                            title="Selected Molecules (t-SNE)")
        print(f"Wrote TSNE plot to {tsne_output}")
    write_pickle(out_dir / "selected.pkl", selection.selected)
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
