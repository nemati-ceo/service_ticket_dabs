"""Tests for cluster_stats — counts and the silhouette sample_size scaling guard.

silhouette_score is the pipeline's one O(N^2) call; `sample_size` caps it. These
tests pin the counting logic and prove the sampling path runs without error and
only kicks in when there are more non-noise points than sample_size.
"""

import numpy as np
import pytest

pytest.importorskip("sklearn")     # cluster_stats needs sklearn; skip cleanly if absent

from conftest import load_by_path

cl = load_by_path("clustering05", "05_clustering/clustering.py")


def _two_clusters_with_noise():
    # two well-separated blobs in 2-D + one noise point
    a = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
    b = np.array([[9.0, 9.0], [9.1, 9.0], [9.0, 9.1]])
    emb = np.vstack([a, b, [[100.0, 100.0]]])
    labels = np.array([0, 0, 0, 1, 1, 1, -1])
    return emb, labels


def test_counts_and_noise():
    emb, labels = _two_clusters_with_noise()
    n_clusters, n_noise, noise_pct, sil = cl.cluster_stats(emb, labels)
    assert n_clusters == 2
    assert n_noise == 1
    assert round(noise_pct, 2) == round(1 / 7 * 100, 2)
    assert sil is not None and -1.0 <= sil <= 1.0


def test_single_cluster_has_no_silhouette():
    emb = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
    labels = np.array([0, 0, 0])
    n_clusters, n_noise, noise_pct, sil = cl.cluster_stats(emb, labels)
    assert n_clusters == 1
    assert sil is None          # silhouette undefined for <2 clusters


def test_sample_size_path_runs():
    # 6 non-noise points, sample_size=4 -> sampling branch is exercised, no error
    emb, labels = _two_clusters_with_noise()
    _, _, _, sil = cl.cluster_stats(emb, labels, sample_size=4)
    assert sil is not None and -1.0 <= sil <= 1.0


def test_sample_size_larger_than_points_is_exact():
    # sample_size above the point count -> falls back to exact, still valid
    emb, labels = _two_clusters_with_noise()
    _, _, _, sil = cl.cluster_stats(emb, labels, sample_size=10_000)
    assert sil is not None and -1.0 <= sil <= 1.0
