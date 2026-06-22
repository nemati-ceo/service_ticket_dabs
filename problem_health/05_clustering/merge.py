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
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        c = e.mean(axis=0)
        cents.append(c / np.linalg.norm(c))
    centroids = np.vstack(cents) if cents else np.empty((0, embeddings.shape[1]))
    return centroids, cluster_ids


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
