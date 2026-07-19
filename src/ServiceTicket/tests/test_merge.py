"""Tests for merge.resolve_themes — the edge-case-2 skip guard and the real merge.

resolve_themes decides whether the O(k^2) centroid merge runs. With < 2 clusters
(all noise, or a single cluster) there is nothing to merge, so it must short-circuit
to an identity theme_map with an empty merge_log. With >= 2 clusters it delegates to
the centroid union-find merge, which should fuse near-duplicate clusters.
"""

import numpy as np
import pytest

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


# --- parity with the ORIGINAL implementation ---------------------------------------
#
# merge_clusters ran a k^2 Python loop over the centroid cosine matrix (12.5M iterations
# at 5000 clusters) and cluster_centroids masked the whole embedding array once per
# cluster (O(k*n)). Both were rewritten; these pin that the rewrite is behaviour-
# identical, not merely faster. The references below ARE the old code.

def _orig_centroids(embeddings, labels):
    labels = np.asarray(labels)
    cluster_ids = sorted(c for c in set(labels) if c != -1)
    cents = []
    for cid in cluster_ids:
        e = embeddings[labels == cid]
        e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
        c = e.mean(axis=0)
        cents.append(c / max(float(np.linalg.norm(c)), 1e-12))
    centroids = np.vstack(cents) if cents else np.empty((0, embeddings.shape[1]))
    return centroids, cluster_ids


def _orig_merge(centroids, cluster_ids, threshold):
    parent = {cid: cid for cid in cluster_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    n = centroids / np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
    sim = n @ n.T if len(cluster_ids) else np.empty((0, 0))
    merge_log = []
    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            if sim[i, j] >= threshold:
                ra, rb = find(cluster_ids[i]), find(cluster_ids[j])
                if ra != rb:
                    parent[ra] = rb
                merge_log.append((cluster_ids[i], cluster_ids[j], round(float(sim[i, j]), 4)))
    theme_map = {cid: find(cid) for cid in cluster_ids}
    theme_map[-1] = -1
    return theme_map, merge_log


def _data(n, dim, n_clusters, seed, noise_frac=0.2):
    """Clustered embeddings: points near n_clusters random centers, plus noise."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    labels = rng.integers(0, n_clusters, size=n)
    emb = centers[labels] + 0.15 * rng.standard_normal((n, dim)).astype(np.float32)
    labels = labels.astype(np.int64)
    labels[rng.random(n) < noise_frac] = -1
    return emb.astype(np.float32), labels


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("threshold", [0.5, 0.9, 0.99])
def test_rewrite_matches_the_original(seed, threshold):
    emb, labels = _data(400, 12, 8, seed)
    want_c, want_ids = _orig_centroids(emb, labels)
    got_c, got_ids = mg.cluster_centroids(emb, labels)
    assert [int(c) for c in want_ids] == got_ids
    np.testing.assert_allclose(got_c, want_c, atol=1e-5)

    want_map, want_log = _orig_merge(want_c, [int(c) for c in want_ids], threshold)
    got_map, got_log = mg.merge_clusters(got_c, got_ids, threshold)
    assert got_map == want_map
    assert [(a, b) for a, b, _ in got_log] == [(a, b) for a, b, _ in want_log]


def test_centroids_are_unit_norm():
    emb, labels = _data(200, 10, 5, 7)
    cents, _ = mg.cluster_centroids(emb, labels)
    np.testing.assert_allclose(np.linalg.norm(cents, axis=1), 1.0, atol=1e-5)


def test_noise_is_excluded_and_ids_stay_sorted():
    emb, labels = _data(200, 10, 5, 8)
    _, ids = mg.cluster_centroids(emb, labels)
    assert -1 not in ids and ids == sorted(ids)


def test_all_noise_yields_no_centroids():
    cents, ids = mg.cluster_centroids(np.ones((5, 4), dtype=np.float32), np.full(5, -1))
    assert cents.shape == (0, 4) and ids == []


def test_merging_is_transitive():
    """A~B and B~C merge all three even when A and C are far apart — documented behaviour,
    and the reason a too-low threshold collapses unrelated themes into one group."""
    centroids = np.array([[1.0, 0.0], [0.7071, 0.7071], [0.0, 1.0]], dtype=np.float32)
    theme_map, _ = mg.merge_clusters(centroids, [0, 1, 2], threshold=0.7)
    assert len({theme_map[0], theme_map[1], theme_map[2]}) == 1
    assert float(centroids[0] @ centroids[2]) < 0.7


def test_theme_map_keys_are_plain_ints():
    """np.int64 keys leaked out of the merge path and plain ints out of the skip path."""
    emb, labels = _data(80, 6, 3, 12)
    theme_map, _, ids = mg.resolve_themes(emb, labels, n_clusters=3, threshold=0.9)
    assert all(type(k) is int for k in theme_map)
    assert all(type(c) is int for c in ids)
