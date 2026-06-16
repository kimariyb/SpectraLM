"""Tests for MaxMin clustering of molecular fingerprints."""

from __future__ import annotations

import numpy as np

from src.data.clustering import (
    ClusterResult,
    _labels_to_clusters,
    cluster_samples,
    tsne_project,
)


def make_fingerprints(n_samples: int = 10, n_features: int = 64) -> np.ndarray:
    """Build a small random fingerprint matrix for testing.

    Parameters
    ----------
    n_samples
        Number of rows.
    n_features
        Number of fingerprint bits.

    Returns
    -------
    numpy.ndarray
        Float32 fingerprint matrix.
    """
    rng = np.random.default_rng(42)
    return rng.random((n_samples, n_features)).astype(np.float32)


# ---------------------------------------------------------------------------
# _labels_to_clusters
# ---------------------------------------------------------------------------


def test_labels_to_clusters_maps_every_row_exactly_once() -> None:
    """Every row index should appear in exactly one cluster."""
    labels = np.array([0, 0, 1, 2, 1, 0], dtype=np.int32)
    clusters = _labels_to_clusters(labels)
    all_indices = {idx for cluster in clusters for idx in cluster}
    assert all_indices == set(range(len(labels)))


def test_labels_to_clusters_handles_single_label() -> None:
    """Single-label arrays should produce one cluster with all indices."""
    labels = np.array([0, 0, 0], dtype=np.int32)
    clusters = _labels_to_clusters(labels)
    assert len(clusters) == 1
    assert clusters[0] == (0, 1, 2)


# ---------------------------------------------------------------------------
# cluster_samples (MaxMin)
# ---------------------------------------------------------------------------


def test_cluster_samples_returns_one_label_per_row() -> None:
    """MaxMin should return one label for each fingerprint row."""
    fps = make_fingerprints(8, 32)
    result = cluster_samples(fps, {"distance_threshold": 0.3, "seed": 42})
    assert len(result.labels) == 8
    assert result.method == "maxmin"
    assert result.distance_threshold == 0.3
    assert result.cluster_count >= 1


def test_cluster_samples_lower_threshold_more_clusters() -> None:
    """Lower distance threshold should produce more (or equal) clusters."""
    fps = make_fingerprints(20, 32)
    r_lo = cluster_samples(fps, {"distance_threshold": 0.1, "seed": 1})
    r_hi = cluster_samples(fps, {"distance_threshold": 0.9, "seed": 1})
    assert r_lo.cluster_count >= r_hi.cluster_count


def test_cluster_samples_identical_fps_same_cluster() -> None:
    """Identical fingerprints should be assigned to the same cluster."""
    fps = np.array([
        [1, 0, 1, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
    ], dtype=np.float32)
    result = cluster_samples(fps, {"distance_threshold": 0.5, "seed": 0})
    assert result.labels[0] == result.labels[1]
    assert result.labels[2] != result.labels[0]


def test_cluster_samples_ignores_rows_parameter() -> None:
    """The rows parameter should be accepted but ignored (backward compat)."""
    fps = make_fingerprints(6, 16)
    result = cluster_samples(fps, {"distance_threshold": 0.5}, rows=[{"id": "x"}])
    assert result.cluster_count >= 1


def test_cluster_samples_raises_on_empty() -> None:
    """Empty fingerprint arrays should raise ValueError."""
    import pytest
    with pytest.raises(ValueError):
        cluster_samples(np.empty((0, 64)))


def test_cluster_samples_deterministic_with_seed() -> None:
    """Same seed should produce identical labels."""
    fps = make_fingerprints(12, 32)
    r1 = cluster_samples(fps, {"distance_threshold": 0.3, "seed": 99})
    r2 = cluster_samples(fps, {"distance_threshold": 0.3, "seed": 99})
    assert np.array_equal(r1.labels, r2.labels)


# ---------------------------------------------------------------------------
# tsne_project
# ---------------------------------------------------------------------------


def test_tsne_project_returns_2d() -> None:
    """t-SNE should reduce data to (n_samples, 2)."""
    data = make_fingerprints(20, 32)
    coords = tsne_project(data, random_state=1)
    assert coords.shape == (20, 2)
    assert coords.dtype == np.float32
    assert np.isfinite(coords).all()


def test_tsne_project_clamps_perplexity() -> None:
    """Perplexity larger than n_samples should be clamped to n_samples-1."""
    data = make_fingerprints(5, 16)
    coords = tsne_project(data, perplexity=100.0, random_state=1)
    assert coords.shape == (5, 2)
    assert np.isfinite(coords).all()


# ---------------------------------------------------------------------------
# ClusterResult
# ---------------------------------------------------------------------------


def test_cluster_result_fields() -> None:
    """ClusterResult should expose labels, clusters, method, threshold, count."""
    labels = np.array([0, 0, 1, 1, 2, 3], dtype=np.int32)
    clusters = _labels_to_clusters(labels)
    result = ClusterResult(
        labels=labels,
        clusters=clusters,
        method="maxmin",
        distance_threshold=0.4,
    )
    assert result.cluster_count == 4
    assert result.method == "maxmin"
    assert result.distance_threshold == 0.4
