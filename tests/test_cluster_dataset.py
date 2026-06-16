"""Tests for MiniBatchKMeans clustering of molecular fingerprints."""

from __future__ import annotations

import numpy as np

from src.data.clustering import (
    ClusterResult,
    _labels_to_clusters,
    cluster_samples,
    tsne_project,
)


def make_fingerprints(n_samples: int = 10, n_features: int = 64) -> np.ndarray:
    """Build a small random fingerprint matrix for testing."""
    rng = np.random.default_rng(42)
    return rng.random((n_samples, n_features)).astype(np.float32)


# ---------------------------------------------------------------------------
# _labels_to_clusters
# ---------------------------------------------------------------------------


def test_labels_to_clusters_maps_every_row_exactly_once() -> None:
    labels = np.array([0, 0, 1, 2, 1, 0], dtype=np.int32)
    clusters = _labels_to_clusters(labels)
    all_indices = {idx for cluster in clusters for idx in cluster}
    assert all_indices == set(range(len(labels)))


def test_labels_to_clusters_handles_single_label() -> None:
    labels = np.array([0, 0, 0], dtype=np.int32)
    clusters = _labels_to_clusters(labels)
    assert len(clusters) == 1
    assert clusters[0] == (0, 1, 2)


# ---------------------------------------------------------------------------
# cluster_samples (MiniBatchKMeans)
# ---------------------------------------------------------------------------


def test_cluster_samples_returns_one_label_per_row() -> None:
    fps = make_fingerprints(20, 32)
    result = cluster_samples(fps, {"n_clusters": 3, "pca_components": 5, "seed": 42})
    assert len(result.labels) == 20
    assert result.method == "minibatch_kmeans"
    assert result.n_clusters == 3
    assert result.cluster_count == 3
    assert 0.0 <= result.pca_variance <= 1.0
    assert result.inertia > 0


def test_cluster_samples_respects_n_clusters() -> None:
    fps = make_fingerprints(30, 32)
    for k in [2, 5]:
        result = cluster_samples(fps, {"n_clusters": k, "pca_components": 5, "seed": 1})
        assert result.n_clusters == k


def test_cluster_samples_raises_on_empty() -> None:
    import pytest
    with pytest.raises(ValueError):
        cluster_samples(np.empty((0, 64)))


def test_cluster_samples_deterministic_with_seed() -> None:
    fps = make_fingerprints(15, 32)
    r1 = cluster_samples(fps, {"n_clusters": 3, "seed": 99})
    r2 = cluster_samples(fps, {"n_clusters": 3, "seed": 99})
    assert np.array_equal(r1.labels, r2.labels)


def test_cluster_samples_uses_batch_size() -> None:
    """batch_size config should be accepted without error."""
    fps = make_fingerprints(25, 64)
    result = cluster_samples(
        fps, {"n_clusters": 3, "pca_components": 5, "batch_size": 5, "seed": 1}
    )
    assert result.cluster_count == 3


# ---------------------------------------------------------------------------
# tsne_project
# ---------------------------------------------------------------------------


def test_tsne_project_returns_2d() -> None:
    data = make_fingerprints(20, 32)
    coords = tsne_project(data, random_state=1)
    assert coords.shape == (20, 2)
    assert coords.dtype == np.float32
    assert np.isfinite(coords).all()


def test_tsne_project_clamps_perplexity() -> None:
    data = make_fingerprints(5, 16)
    coords = tsne_project(data, perplexity=100.0, random_state=1)
    assert coords.shape == (5, 2)


# ---------------------------------------------------------------------------
# ClusterResult
# ---------------------------------------------------------------------------


def test_cluster_result_fields() -> None:
    labels = np.array([0, 0, 1, 1, 2, 3], dtype=np.int32)
    clusters = _labels_to_clusters(labels)
    result = ClusterResult(
        labels=labels,
        clusters=clusters,
        method="minibatch_kmeans",
        n_clusters=4,
        pca_variance=0.85,
        inertia=12.3,
    )
    assert result.cluster_count == 4
    assert result.method == "minibatch_kmeans"
    assert result.pca_variance == 0.85
    assert result.inertia == 12.3
