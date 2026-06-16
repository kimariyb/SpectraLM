"""MaxMin clustering for molecular fingerprints.

Pipeline: sanitise → MaxMin.  PCA + t-SNE helpers are kept for
post-hoc visualisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from skfp.clustering import MaxMinClustering


@dataclass
class ClusterResult:
    """Container for MaxMin clustering outputs.

    Parameters
    ----------
    labels
        Cluster label (0 .. n_clusters-1) for each fingerprint row.
    clusters
        Clusters as tuples of row indices.
    method
        Clustering method name (``"maxmin"``).
    distance_threshold
        MaxMin Tanimoto distance threshold.
    """

    labels: np.ndarray
    clusters: list[tuple[int, ...]]
    method: str
    distance_threshold: float

    @property
    def cluster_count(self) -> int:
        """Return the number of clusters.

        Returns
        -------
        int
            Cluster count.
        """
        return len(self.clusters)


def sanitise_fingerprints(
    fingerprints: np.ndarray,
    rows: list[dict[str, Any]] | None,
) -> tuple[np.ndarray, list[dict[str, Any]] | None]:
    """Diagnose and clean fingerprint matrix before clustering.

    Replaces NaN/Inf with 0 and warns about all-zero rows.

    Parameters
    ----------
    fingerprints
        Input fingerprint matrix.
    rows
        Optional metadata rows (kept in sync if rows are dropped).

    Returns
    -------
    tuple[numpy.ndarray, list[dict[str, Any]] | None]
        Sanitised fingerprints and (optionally filtered) rows.
    """
    n_total = fingerprints.shape[0]

    nan_mask = ~np.isfinite(fingerprints)
    nan_count = int(nan_mask.sum())
    if nan_count > 0:
        print(f"[clustering] Replacing {nan_count} NaN/Inf values with 0")
        fingerprints = np.nan_to_num(fingerprints, nan=0.0, posinf=0.0, neginf=0.0)

    zero_rows = ~fingerprints.any(axis=1)
    n_zero = int(zero_rows.sum())
    if n_zero > 0:
        print(
            f"[clustering] Dropping {n_zero}/{n_total} all-zero fingerprint rows "
            f"({n_zero / n_total:.1%})"
        )
        keep = ~zero_rows
        fingerprints = fingerprints[keep]
        if rows is not None:
            rows = [r for r, k in zip(rows, keep) if k]

    if n_total > 0:
        nonzero_frac = float((fingerprints > 0).mean())
        print(
            f"[clustering] {fingerprints.shape[0]:,} fingerprints, "
            f"{fingerprints.shape[1]} bits, {nonzero_frac:.2%} non-zero"
        )

    return fingerprints.astype(np.float32), rows


def _labels_to_clusters(labels: np.ndarray) -> list[tuple[int, ...]]:
    """Convert a flat label array into per-cluster index tuples.

    Parameters
    ----------
    labels
        1-D integer cluster labels.

    Returns
    -------
    list[tuple[int, ...]]
        One tuple of global row indices per cluster.
    """
    clusters: list[tuple[int, ...]] = []
    for cluster_id in range(int(labels.max()) + 1):
        indices = tuple(int(i) for i in np.where(labels == cluster_id)[0])
        if indices:
            clusters.append(indices)
    return clusters


def _assign_to_nearest(
    data: np.ndarray,
    centroids: np.ndarray,
    centroid_labels: np.ndarray,
) -> np.ndarray:
    """Assign each row in *data* to the nearest centroid by Tanimoto distance.

    Parameters
    ----------
    data
        ``(n_samples, n_features)`` full fingerprint matrix.
    centroids
        ``(n_centroids, n_features)`` fingerprint rows used as cluster centers.
    centroid_labels
        Integer labels for each centroid row.

    Returns
    -------
    numpy.ndarray
        ``(n_samples,)`` int32 cluster labels.
    """
    labels = np.empty(data.shape[0], dtype=np.int32)
    chunk_size = 20000

    for start in range(0, data.shape[0], chunk_size):
        end = min(start + chunk_size, data.shape[0])
        chunk = data[start:end]

        # Compute Tanimoto distance matrix: chunk × centroids
        # Tanimoto(a,b) = (a·b) / (|a|+|b| - a·b)
        # Distance = 1 - similarity
        a_bin = chunk > 0
        c_bin = centroids > 0
        a_sum = a_bin.sum(axis=1, keepdims=True)  # (chunk, 1)
        c_sum = c_bin.sum(axis=1)                   # (centroids,)
        intersection = chunk @ centroids.T           # (chunk, centroids)
        # Avoid division by zero
        denom = np.maximum(a_sum + c_sum - intersection, 1)
        similarity = intersection / denom
        distance = 1.0 - similarity
        labels[start:end] = centroid_labels[distance.argmin(axis=1)]

    return labels


def cluster_samples(
    fingerprints: np.ndarray,
    config: dict[str, Any] | None = None,
    rows: list[dict[str, Any]] | None = None,
) -> ClusterResult:
    """Cluster molecular fingerprints with MaxMin (Tanimoto distance).

    Parameters
    ----------
    fingerprints
        Floating-point fingerprint matrix of shape ``(n_samples, n_features)``.
    config
        Clustering configuration.  Supported keys:

        - ``distance_threshold`` (float, default 0.4): MaxMin Tanimoto
          distance cutoff.  Lower values produce more clusters.
        - ``seed`` (int, default 3407): random seed.
    rows
        Ignored; kept for backward compatibility.

    Returns
    -------
    ClusterResult
        MaxMin labels, cluster memberships, and threshold.

    Raises
    ------
    ValueError
        If the fingerprint matrix is empty or not two-dimensional.
    """
    _ = rows  # kept for backward compat
    cfg = config or {}

    if fingerprints.ndim != 2 or len(fingerprints) == 0:
        raise ValueError("fingerprints must be a non-empty two-dimensional matrix")

    distance_threshold = float(cfg.get("distance_threshold", 0.4))
    seed = int(cfg.get("seed", 3407))
    max_cluster_samples = int(cfg.get("max_cluster_samples", 50000))

    # On large datasets, fit MaxMin on a random subset then assign the rest
    n_total = fingerprints.shape[0]
    if n_total > max_cluster_samples:
        rng = np.random.default_rng(seed)
        fit_indices = rng.choice(n_total, size=max_cluster_samples, replace=False)
        fit_indices.sort()
        print(
            f"[clustering] Fitting MaxMin on {max_cluster_samples:,} / {n_total:,} "
            f"samples ({max_cluster_samples / n_total:.1%}) ..."
        )
    else:
        fit_indices = np.arange(n_total)

    clusterer = MaxMinClustering(
        distance_threshold=distance_threshold,
        random_state=seed,
    )
    fit_labels = clusterer.fit_predict(fingerprints[fit_indices]).astype(np.int32)

    if n_total > max_cluster_samples:
        # Assign remaining samples to nearest cluster centroid
        print(
            f"[clustering] Assigning remaining {n_total - max_cluster_samples:,} "
            f"samples to {fit_labels.max() + 1} clusters ..."
        )
        labels = _assign_to_nearest(fingerprints, fingerprints[fit_indices], fit_labels)
    else:
        labels = fit_labels

    clusters = _labels_to_clusters(labels)

    return ClusterResult(
        labels=labels,
        clusters=clusters,
        method="maxmin",
        distance_threshold=distance_threshold,
    )


def pca_reduce(
    fingerprints: np.ndarray,
    n_components: int = 50,
    random_state: int = 3407,
) -> tuple[np.ndarray, float]:
    """Reduce fingerprint dimensionality with PCA.

    Parameters
    ----------
    fingerprints
        Input matrix of shape ``(n_samples, n_features)``.
    n_components
        Target number of principal components (clamped to min of the
        matrix dimensions).
    random_state
        Random seed.

    Returns
    -------
    tuple[numpy.ndarray, float]
        PCA-reduced matrix (float32) and cumulative explained variance ratio.
    """
    n_components = min(n_components, fingerprints.shape[0], fingerprints.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    reduced = pca.fit_transform(fingerprints).astype(np.float32)
    return reduced, float(pca.explained_variance_ratio_.sum())


def tsne_project(
    data: np.ndarray,
    perplexity: float = 30.0,
    random_state: int = 3407,
) -> np.ndarray:
    """Project high-dimensional data to 2-D with t-SNE.

    Parameters
    ----------
    data
        Input matrix of shape ``(n_samples, n_features)``.  Typically the
        PCA-reduced matrix or raw fingerprints.
    perplexity
        t-SNE perplexity (clamped to ``n_samples - 1``).
    random_state
        Random seed.

    Returns
    -------
    numpy.ndarray
        ``(n_samples, 2)`` float32 t-SNE coordinates.
    """
    n_samples = data.shape[0]
    effective_perp = min(perplexity, max(2.0, float(n_samples - 1)))
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perp,
        random_state=random_state,
    )
    return tsne.fit_transform(data).astype(np.float32)


def plot_tsne_clusters(
    tsne_coords: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
    title: str = "MaxMin Clusters (t-SNE)",
    palette: str = "tab10",
) -> None:
    """Plot a 2-D t-SNE projection coloured by cluster label.

    Parameters
    ----------
    tsne_coords
        ``(n_samples, 2)`` t-SNE coordinates.
    labels
        Integer cluster label per sample.
    output_path
        Output image path (PNG recommended).
    title
        Plot title.
    palette
        Seaborn colour palette name.
    """
    sns.set_theme(style="ticks", context="notebook", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(10, 8))

    n_clusters = int(labels.max()) + 1
    colors = sns.color_palette(palette, n_colors=max(n_clusters, 1))

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        ax.scatter(
            tsne_coords[mask, 0],
            tsne_coords[mask, 1],
            s=12,
            c=[colors[cluster_id]],
            alpha=0.7,
            edgecolors="white",
            linewidths=0.3,
            label=f"Cluster {cluster_id} ({mask.sum():,})",
        )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(frameon=False, loc="best", fontsize=9)
    sns.despine(ax=ax)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
