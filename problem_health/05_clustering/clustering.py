"""clustering.py — embed text, reduce with UMAP, cluster with HDBSCAN."""

import numpy as np


def embed(texts, model_name, batch_size=64):
    """Encode texts with a sentence-transformer."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(list(texts), show_progress_bar=True, batch_size=batch_size)


def reduce_umap(embeddings, params):
    """UMAP dimensionality reduction."""
    import umap
    return umap.UMAP(**params).fit_transform(embeddings)


def cluster_hdbscan(embeddings_5d, params):
    """HDBSCAN clustering; returns integer labels (-1 = noise)."""
    import hdbscan
    return hdbscan.HDBSCAN(**params).fit(embeddings_5d).labels_


def small_sample_noise(n_rows):
    """Labels + stats for a sample too small to cluster: every point is noise (-1).

    Below a handful of rows HDBSCAN/UMAP can't form meaningful clusters (and UMAP with
    n_neighbors >= n_rows errors), so we short-circuit to an all-noise result. Returns
    (labels, (n_clusters, n_noise, noise_pct, silhouette)) matching cluster_stats().
    """
    labels = np.full(n_rows, -1, dtype=int)
    noise_pct = 100.0 if n_rows else 0.0
    return labels, (0, n_rows, noise_pct, None)


def cluster_stats(embeddings_5d, labels, sample_size=None):
    """Return (n_clusters, n_noise, noise_pct, silhouette) and log them.

    silhouette_score builds the full N x N distance matrix (O(N^2) time + memory).
    `sample_size` caps that by subsampling that many non-noise points (sklearn-native)
    so it scales on large datasets; None keeps the exact (small-N only) computation.
    """
    from sklearn.metrics import silhouette_score
    labels = np.asarray(labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    noise_pct = n_noise / len(labels) * 100 if len(labels) else 0.0
    sil = None
    if n_clusters > 1:
        mask = labels != -1
        kwargs = {"metric": "cosine"}
        if sample_size and int(mask.sum()) > sample_size:
            kwargs.update(sample_size=sample_size, random_state=42)
        try:
            sil = float(silhouette_score(embeddings_5d[mask], labels[mask], **kwargs))
        except Exception as e:
            print(f"[ph05] silhouette skipped ({e})")
    print(f"[ph05] clusters={n_clusters} noise={n_noise} ({noise_pct:.1f}%)"
          + (f" silhouette={sil:.4f}" if sil is not None else ""))
    return n_clusters, n_noise, noise_pct, sil
