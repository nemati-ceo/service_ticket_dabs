"""merge.py — merge near-duplicate clusters into themes via centroid cosine + union-find."""

import numpy as np


def cluster_centroids(embeddings, labels):
    """L2-normalized mean embedding per non-noise cluster. Returns (centroids, cluster_ids)."""
    labels = np.asarray(labels)
    emb = np.asarray(embeddings, dtype=np.float32)
    keep = labels != -1
    lab, emb = labels[keep], emb[keep]
    if lab.size == 0:
        return np.empty((0, emb.shape[1]), dtype=np.float32), []

    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    # Group by one sort instead of one full-array mask per cluster: masking was O(k*n)
    # and scanned every embedding once per cluster.
    order = np.argsort(lab, kind="stable")
    lab, emb = lab[order], emb[order]
    starts = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1]])
    sums = np.add.reduceat(emb, starts, axis=0)
    counts = np.diff(np.r_[starts, lab.size])[:, None]
    cents = sums / counts
    cents /= np.maximum(np.linalg.norm(cents, axis=1, keepdims=True), 1e-12)
    return cents.astype(np.float32), lab[starts].tolist()


def resolve_themes(embeddings, labels, n_clusters, threshold):
    """Map clusters to themes, skipping the merge when there is nothing to merge.

    With < 2 clusters (all noise, or a single cluster) no pair can merge, so each
    cluster maps to itself (noise stays -1) and merge_log is empty.
    Returns (theme_map, merge_log, cluster_ids).
    """
    cluster_ids = sorted(int(c) for c in set(np.asarray(labels).tolist()) if c != -1)
    if n_clusters < 2:
        return {**{cid: cid for cid in cluster_ids}, -1: -1}, [], cluster_ids
    centroids, cluster_ids = cluster_centroids(embeddings, labels)
    theme_map, merge_log = merge_clusters(centroids, cluster_ids, threshold)
    return theme_map, merge_log, cluster_ids


def merge_clusters(centroids, cluster_ids, threshold):
    """Union clusters whose centroid cosine >= threshold. Returns (theme_map, merge_log).

    Merging is TRANSITIVE: if A~B and B~C both clear the threshold, A, B and C land in one
    theme even when A·C does not. That is what makes a too-low threshold collapse unrelated
    themes into one giant group — check n_themes against n_clusters before trusting a run.
    """
    cluster_ids = [int(c) for c in cluster_ids]
    parent = {cid: cid for cid in cluster_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    k = len(cluster_ids)
    merge_log = []
    if k > 1:
        # Centroids are already L2-normalized, so the gram matrix IS the cosine matrix.
        # Only pairs at/above the threshold reach Python: the old nested loop ran k^2
        # iterations (12.5M at 5000 clusters) to find a handful of merges.
        sim = centroids @ centroids.T
        rows, cols = np.triu_indices(k, k=1)
        hits = np.flatnonzero(sim[rows, cols] >= threshold)
        for h in hits:
            i, j = int(rows[h]), int(cols[h])
            a, b = cluster_ids[i], cluster_ids[j]
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
            merge_log.append((a, b, round(float(sim[i, j]), 4)))

    theme_map = {cid: find(cid) for cid in cluster_ids}
    theme_map[-1] = -1
    return theme_map, merge_log
