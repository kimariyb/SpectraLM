"""MiniBatchKMeans clustering with PCA for molecular fingerprints.

Pipeline: sanitise → PCA → MiniBatchKMeans.  PCA + t-SNE helpers are
kept for post-hoc visualisation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


@dataclass
class ClusterResult:
    """Container for MiniBatchKMeans clustering outputs.

    Parameters
    ----------
    labels
        Cluster label (0 .. n_clusters-1) for each fingerprint row.
    clusters
        Clusters as tuples of row indices.
    method
        Clustering method name (``"minibatch_kmeans"``).
    n_clusters
        Number of clusters.
    pca_variance
        Cumulative explained variance from PCA (0..1).
    inertia
        MiniBatchKMeans inertia.
    """

    labels: np.ndarray
    clusters: list[tuple[int, ...]]
    method: str
    n_clusters: int
    pca_variance: float
    inertia: float

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
            f"[clustering] Dropping {n_zero}/{n_total} all-zero rows "
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


def cluster_samples(
    fingerprints: np.ndarray,
    config: dict[str, Any] | None = None,
    rows: list[dict[str, Any]] | None = None,
) -> ClusterResult:
    """Cluster fingerprints with PCA → MiniBatchKMeans.

    PCA reduces dimensionality first; MiniBatchKMeans then clusters the
    reduced data using mini-batches for low memory footprint.

    Parameters
    ----------
    fingerprints
        Float fingerprint matrix ``(n_samples, n_features)``.
    config
        Clustering configuration:

        - ``n_clusters`` (int, default 100): number of clusters.
        - ``pca_components`` (int, default 50): PCA target dimensionality.
        - ``batch_size`` (int, default 10000): MiniBatchKMeans batch size.
        - ``seed`` (int, default 3407): random seed.
    rows
        Ignored; kept for backward compatibility.

    Returns
    -------
    ClusterResult
        Cluster labels, memberships, and diagnostics.

    Raises
    ------
    ValueError
        If the fingerprint matrix is empty or not 2-D.
    """
    _ = rows
    cfg = config or {}

    if fingerprints.ndim != 2 or len(fingerprints) == 0:
        raise ValueError("fingerprints must be a non-empty two-dimensional matrix")

    n_clusters = int(cfg.get("n_clusters", 100))
    n_components = int(cfg.get("pca_components", 50))
    batch_size = int(cfg.get("batch_size", 10000))
    seed = int(cfg.get("seed", 3407))

    # --- PCA ----------------------------------------------------------------
    reduced, pca_variance = pca_reduce(fingerprints, n_components, seed)

    # --- MiniBatchKMeans ----------------------------------------------------
    print(
        f"[clustering] MiniBatchKMeans: {reduced.shape[0]:,} samples → "
        f"{n_clusters} clusters (batch_size={batch_size:,}) ..."
    )
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        random_state=seed,
        n_init=3,
        max_iter=100,
        reassignment_ratio=0.01,
        verbose=1,
    )
    labels = kmeans.fit_predict(reduced).astype(np.int32)
    clusters = _labels_to_clusters(labels)

    return ClusterResult(
        labels=labels,
        clusters=clusters,
        method="minibatch_kmeans",
        n_clusters=n_clusters,
        pca_variance=pca_variance,
        inertia=float(kmeans.inertia_),
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
        Input matrix ``(n_samples, n_features)``.
    n_components
        Target number of principal components (clamped to matrix dims).
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
        Input matrix ``(n_samples, n_features)``.
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
        verbose=1
    )
    return tsne.fit_transform(data).astype(np.float32)


def elbow_n_clusters(
    fingerprints: np.ndarray,
    ks: list[int] | None = None,
    pca_components: int = 50,
    batch_size: int = 10000,
    seed: int = 3407,
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Run MiniBatchKMeans for multiple *k* values and return diagnostics.

    Parameters
    ----------
    fingerprints
        Fingerprint matrix.
    ks
        Candidate *k* values.  Defaults to a geometric sweep from 10 to
        ``sqrt(n)``.
    pca_components
        PCA target dimensions.
    batch_size
        MiniBatchKMeans batch size.
    seed
        Random seed.
    output_path
        If given, save an elbow plot to this path.

    Returns
    -------
    list[dict[str, Any]]
        One dict per *k* with keys ``k``, ``inertia``, ``silhouette``,
        ``time_s``.
    """
    from sklearn.metrics import silhouette_score
    import time

    if ks is None:
        n = fingerprints.shape[0]
        ks = sorted(
            {max(10, int(np.sqrt(n) * f)) for f in [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]}
        )

    reduced, _ = pca_reduce(fingerprints, pca_components, seed)
    results: list[dict[str, Any]] = []

    for k in ks:
        t0 = time.perf_counter()
        kmeans = MiniBatchKMeans(
            n_clusters=k, batch_size=batch_size, random_state=seed, n_init=3, max_iter=100,
        )
        labels = kmeans.fit_predict(reduced)
        elapsed = time.perf_counter() - t0

        inertia = float(kmeans.inertia_)
        # Silhouette on a subsample for speed if n > 50000
        if reduced.shape[0] > 50000:
            rng = np.random.default_rng(seed)
            idx = rng.choice(reduced.shape[0], size=50000, replace=False)
            sil = float(silhouette_score(reduced[idx], labels[idx]))
        else:
            sil = float(silhouette_score(reduced, labels))

        results.append({"k": k, "inertia": inertia, "silhouette": sil, "time_s": round(elapsed, 1)})
        print(f"  k={k:5d}  inertia={inertia:12.1f}  silhouette={sil:.4f}  time={elapsed:.1f}s")

    if output_path:
        plot_elbow(results, output_path)

    return results


def plot_elbow(results: list[dict[str, Any]], output_path: str | Path) -> None:
    """Plot inertia and silhouette vs *k*."""
    ks = [r["k"] for r in results]
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(ks, [r["inertia"] for r in results], "b-o", label="Inertia")
    ax1.set_xlabel("n_clusters")
    ax1.set_ylabel("Inertia", color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    ax2 = ax1.twinx()
    ax2.plot(ks, [r["silhouette"] for r in results], "r-s", label="Silhouette")
    ax2.set_ylabel("Silhouette Score", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    fig.suptitle("Elbow Method: Inertia + Silhouette vs n_clusters")
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote elbow plot to {output_path}")


def plot_tsne_selection(
    tsne_coords: np.ndarray,
    train_row_indices: list[int],
    test_row_indices: list[int],
    output_path: str | Path,
    title: str = "Selected Molecules (t-SNE)",
) -> None:
    """Plot all molecules as grey background with selected train/test highlighted.

    Parameters
    ----------
    tsne_coords
        ``(n_samples, 2)`` t-SNE coordinates for all molecules.
    train_row_indices
        Row indices of selected training molecules.
    test_row_indices
        Row indices of selected test molecules.
    output_path
        Output image path.
    title
        Plot title.
    """
    sns.set_theme(style="ticks", context="notebook", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(10, 8))

    total = tsne_coords.shape[0]
    selected = set(train_row_indices) | set(test_row_indices)
    background = np.array([i for i in range(total) if i not in selected])

    # Background: all unselected molecules
    ax.scatter(
        tsne_coords[background, 0],
        tsne_coords[background, 1],
        s=3,
        c="#c9c9c9",
        alpha=0.15,
        linewidths=0,
        label=f"Unselected ({len(background):,})",
    )

    # Train: red
    if train_row_indices:
        ax.scatter(
            tsne_coords[train_row_indices, 0],
            tsne_coords[train_row_indices, 1],
            s=28,
            c="#d62728",
            alpha=0.9,
            edgecolors="white",
            linewidths=0.35,
            label=f"Train ({len(train_row_indices):,})",
        )

    # Test: blue
    if test_row_indices:
        ax.scatter(
            tsne_coords[test_row_indices, 0],
            tsne_coords[test_row_indices, 1],
            s=28,
            c="#1f77b4",
            alpha=0.9,
            edgecolors="white",
            linewidths=0.35,
            label=f"Test ({len(test_row_indices):,})",
        )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(frameon=False, loc="best")
    sns.despine(ax=ax)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_tsne_clusters(
    tsne_coords: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
    title: str = "MiniBatchKMeans Clusters (t-SNE)",
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
