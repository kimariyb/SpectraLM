"""Butina clustering utilities for Morgan fingerprints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina


@dataclass
class ClusterResult:
    """Container for Butina clustering outputs.

    Parameters
    ----------
    labels
        Cluster label for each fingerprint row.
    clusters
        Butina clusters as tuples of row indices.
    method
        Clustering method name.
    cutoff
        Tanimoto similarity cutoff used to convert distances.
    bucket_count
        Number of buckets clustered.
    bucket_sizes
        Number of molecules in each bucket.
    """

    labels: np.ndarray
    clusters: list[tuple[int, ...]]
    method: str
    cutoff: float
    bucket_count: int = 1
    bucket_sizes: list[int] | None = None

    @property
    def cluster_count(self) -> int:
        """Return the number of clusters.

        Returns
        -------
        int
            Cluster count.
        """
        return len(self.clusters)


def fingerprint_rows_to_bitvects(fingerprints: np.ndarray) -> list[DataStructs.ExplicitBitVect]:
    """Convert a binary fingerprint matrix into RDKit bit vectors.

    Parameters
    ----------
    fingerprints
        Binary Morgan fingerprint matrix.

    Returns
    -------
    list[rdkit.DataStructs.ExplicitBitVect]
        RDKit bit vectors.
    """
    bitvects = []
    for row in fingerprints:
        bits = DataStructs.ExplicitBitVect(int(row.shape[0]))
        on_bits = np.flatnonzero(row > 0)
        for bit in on_bits:
            bits.SetBit(int(bit))
        bitvects.append(bits)
    return bitvects


def tanimoto_distance_vector(bitvects: list[DataStructs.ExplicitBitVect]) -> list[float]:
    """Build the lower-triangle Tanimoto distance vector required by Butina.

    Parameters
    ----------
    bitvects
        RDKit fingerprint bit vectors.

    Returns
    -------
    list[float]
        Lower-triangle distance vector.
    """
    distances = []
    for idx in range(1, len(bitvects)):
        similarities = DataStructs.BulkTanimotoSimilarity(bitvects[idx], bitvects[:idx])
        distances.extend(1.0 - similarity for similarity in similarities)
    return distances


def butina_clusters_for_indices(
    bitvects: list[DataStructs.ExplicitBitVect],
    indices: list[int],
    distance_cutoff: float,
) -> list[tuple[int, ...]]:
    """Cluster a subset of fingerprint indices with Butina.

    Parameters
    ----------
    bitvects
        All RDKit fingerprint bit vectors.
    indices
        Global row indices to cluster.
    distance_cutoff
        Butina distance cutoff.

    Returns
    -------
    list[tuple[int, ...]]
        Butina clusters using global row indices.
    """
    if len(indices) == 1:
        return [(indices[0],)]
    local_bitvects = [bitvects[idx] for idx in indices]
    local_clusters = Butina.ClusterData(
        tanimoto_distance_vector(local_bitvects),
        len(local_bitvects),
        distance_cutoff,
        isDistData=True,
    )
    return [tuple(indices[local_idx] for local_idx in cluster) for cluster in local_clusters]


def row_bucket_key(row: dict[str, Any]) -> str:
    """Build a coarse structure bucket key from feature metadata.

    Parameters
    ----------
    row
        Feature metadata row.

    Returns
    -------
    str
        Bucket key based on scaffold, functional groups, and SMILES length.
    """
    scaffold = row.get("murcko_scaffold") or "missing_scaffold"
    groups = row.get("functional_groups", [])
    useful_groups = sorted(group for group in groups if group not in {"invalid", "none_detected"})
    signature = ".".join(useful_groups) if useful_groups else "no_fg"
    smiles_len = len(row.get("canonical_smiles", ""))
    size_bin = min(smiles_len // 10, 12)
    return f"{scaffold}|fg:{signature}|size:{size_bin}"


def bucket_indices(
    rows: list[dict[str, Any]],
    max_bucket_size: int,
) -> list[list[int]]:
    """Create bounded scaffold-stratified buckets for Butina clustering.

    Parameters
    ----------
    rows
        Feature metadata rows.
    max_bucket_size
        Maximum number of molecules per bucket.

    Returns
    -------
    list[list[int]]
        Global row index buckets.
    """
    groups: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        groups.setdefault(row_bucket_key(row), []).append(idx)
    buckets = []
    for indices in groups.values():
        indices = sorted(indices, key=lambda idx: rows[idx].get("canonical_smiles", ""))
        for start in range(0, len(indices), max_bucket_size):
            buckets.append(indices[start : start + max_bucket_size])
    return buckets


def cluster_samples(
    fingerprints: np.ndarray,
    config: dict[str, Any] | None = None,
    rows: list[dict[str, Any]] | None = None,
) -> ClusterResult:
    """Cluster Morgan fingerprints with Butina clustering.

    Parameters
    ----------
    fingerprints
        Binary Morgan fingerprint matrix.
    config
        Clustering configuration. Supports ``butina_cutoff``, ``bucketed``,
        and ``max_bucket_size``.
    rows
        Optional feature metadata rows used for scaffold-stratified bucketing.

    Returns
    -------
    ClusterResult
        Cluster labels and Butina cluster memberships.

    Raises
    ------
    ValueError
        If the fingerprint matrix is empty or not two-dimensional.
    """
    cfg = config or {}
    if fingerprints.ndim != 2 or len(fingerprints) == 0:
        raise ValueError("fingerprints must be a non-empty two-dimensional matrix")
    cutoff = float(cfg.get("butina_cutoff", 0.7))
    bitvects = fingerprint_rows_to_bitvects(fingerprints)
    distance_cutoff = 1.0 - cutoff
    use_buckets = bool(cfg.get("bucketed", False))
    if use_buckets:
        if rows is None:
            raise ValueError("rows are required when bucketed Butina clustering is enabled")
        buckets = bucket_indices(rows, max_bucket_size=int(cfg.get("max_bucket_size", 5000)))
        clusters = []
        for bucket in buckets:
            clusters.extend(butina_clusters_for_indices(bitvects, bucket, distance_cutoff))
        method = "bucketed_butina"
        bucket_sizes = [len(bucket) for bucket in buckets]
    else:
        clusters = butina_clusters_for_indices(bitvects, list(range(len(bitvects))), distance_cutoff)
        method = "butina"
        bucket_sizes = [len(bitvects)]
    labels = np.empty((len(bitvects),), dtype=np.int32)
    for cluster_id, cluster in enumerate(clusters):
        for row_idx in cluster:
            labels[int(row_idx)] = cluster_id
    return ClusterResult(
        labels=labels,
        clusters=clusters,
        method=method,
        cutoff=cutoff,
        bucket_count=len(bucket_sizes),
        bucket_sizes=bucket_sizes,
    )
