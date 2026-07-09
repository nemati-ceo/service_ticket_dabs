"""Tests for merge.resolve_themes — the edge-case-2 skip guard and the real merge.

resolve_themes decides whether the O(k^2) centroid merge runs. With < 2 clusters
(all noise, or a single cluster) there is nothing to merge, so it must short-circuit
to an identity theme_map with an empty merge_log. With >= 2 clusters it delegates to
the centroid union-find merge, which should fuse near-duplicate clusters.
"""

import numpy as np
import pytest

pytest.importorskip("sklearn")     # merge_clusters needs sklearn.cosine_similarity

from conftest import load_by_path

mg = load_by_path("merge05", "05_clustering/merge.py")


def test_all_noise_skips_merge():
    # every point is noise -> no clusters, identity map is just {-1: -1}, no merges
    labels = np.array([-1, -1, -1])
    emb = np.zeros((3, 4))
    theme_map, merge_log, cluster_ids = mg.resolve_themes(emb, labels, n_clusters=0, threshold=0.9)
    assert cluster_ids == []
    assert merge_log == []
    assert theme_map == {-1: -1}


def test_single_cluster_skips_merge():
    # one cluster (+ noise) -> nothing to merge; cluster maps to itself, noise stays -1
    labels = np.array([0, 0, 0, -1])
    emb = np.random.default_rng(0).normal(size=(4, 5))
    theme_map, merge_log, cluster_ids = mg.resolve_themes(emb, labels, n_clusters=1, threshold=0.9)
    assert cluster_ids == [0]
    assert merge_log == []
    assert theme_map == {0: 0, -1: -1}


def test_two_identical_clusters_merge():
    # two clusters with identical directions -> centroid cosine ~1 >= threshold -> fused
    emb = np.array([[1.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
    labels = np.array([0, 0, 1, 1])
    theme_map, merge_log, cluster_ids = mg.resolve_themes(emb, labels, n_clusters=2, threshold=0.9)
    assert cluster_ids == [0, 1]
    assert len(merge_log) == 1
    # both clusters collapse onto a single theme
    assert theme_map[0] == theme_map[1]
    assert theme_map[-1] == -1


def test_two_orthogonal_clusters_stay_separate():
    # orthogonal centroids -> cosine 0 < threshold -> no merge, distinct themes
    emb = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    labels = np.array([0, 0, 1, 1])
    theme_map, merge_log, cluster_ids = mg.resolve_themes(emb, labels, n_clusters=2, threshold=0.9)
    assert merge_log == []
    assert theme_map[0] != theme_map[1]
