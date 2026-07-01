"""merge.py — merge near-duplicate clusters into themes via centroid cosine + union-find."""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def cluster_centroids(embeddings, labels):
    """L2-normalized mean embedding per non-noise cluster. Returns (centroids, cluster_ids)."""
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


def resolve_themes(embeddings, labels, n_clusters, threshold):
    """Map clusters to themes, skipping the merge when there is nothing to merge.

    With < 2 clusters (all noise, or a single cluster) no pair can merge, so each
    cluster maps to itself (noise stays -1) and merge_log is empty — this avoids the
    O(k^2) centroid cosine pass. Otherwise run the centroid union-find merge.
    Returns (theme_map, merge_log, cluster_ids).
    """
    cluster_ids = sorted(c for c in set(np.asarray(labels).tolist()) if c != -1)
    if n_clusters < 2:
        return {**{cid: cid for cid in cluster_ids}, -1: -1}, [], cluster_ids
    centroids, cluster_ids = cluster_centroids(embeddings, labels)
    theme_map, merge_log = merge_clusters(centroids, cluster_ids, threshold)
    return theme_map, merge_log, cluster_ids


def merge_clusters(centroids, cluster_ids, threshold):
    """Union clusters whose centroid cosine >= threshold. Returns (theme_map, merge_log)."""
    parent = {cid: cid for cid in cluster_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    sim = cosine_similarity(centroids) if len(cluster_ids) else np.empty((0, 0))
    merge_log = []
    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            if sim[i, j] >= threshold:
                union(cluster_ids[i], cluster_ids[j])
                merge_log.append((cluster_ids[i], cluster_ids[j], round(float(sim[i, j]), 4)))

    theme_map = {cid: find(cid) for cid in cluster_ids}
    theme_map[-1] = -1
    return theme_map, merge_log
