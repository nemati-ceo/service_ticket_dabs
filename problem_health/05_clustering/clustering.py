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


def cluster_stats(embeddings_5d, labels):
    """Return (n_clusters, n_noise, noise_pct, silhouette) and log them."""
    from sklearn.metrics import silhouette_score
    labels = np.asarray(labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    noise_pct = n_noise / len(labels) * 100 if len(labels) else 0.0
    sil = None
    if n_clusters > 1:
        mask = labels != -1
        try:
            sil = float(silhouette_score(embeddings_5d[mask], labels[mask], metric="cosine"))
        except Exception as e:
            print(f"[ph05] silhouette skipped ({e})")
    print(f"[ph05] clusters={n_clusters} noise={n_noise} ({noise_pct:.1f}%)"
          + (f" silhouette={sil:.4f}" if sil is not None else ""))
    return n_clusters, n_noise, noise_pct, sil
