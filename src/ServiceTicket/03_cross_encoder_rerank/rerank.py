"""rerank.py — cross-encoder reranking of the top-K candidate problems per incident."""

import os

import numpy as np


def _load_cached(model_name, volume_path, cls, **kwargs):
    """Load from Volume if cached there; else download from HF and cache to Volume."""
    if volume_path and os.path.isdir(volume_path) and os.listdir(volume_path):
        print(f"  loading {cls.__name__} from Volume: {volume_path}")
        return cls(volume_path, **kwargs)
    model = cls(model_name, **kwargs)
    if volume_path:
        try:
            os.makedirs(volume_path, exist_ok=True)
            model.save(volume_path)
            print(f"  {cls.__name__} downloaded and cached to Volume: {volume_path}")
        except Exception as e:
            print(f"  WARNING: could not cache model to Volume ({e})")
    return model


def load_cross_encoder(model_name, max_length=512, volume_path=None):
    """Load the cross-encoder reranker (Volume cache first, else HF download)."""
    from sentence_transformers import CrossEncoder
    model = _load_cached(model_name, volume_path, CrossEncoder, max_length=max_length)
    print(f"  Cross-encoder ready: {model_name} (max_length={max_length})")
    return model


def load_bi_encoder(model_name, volume_path=None):
    """Load the bi-encoder (Volume cache first). Load ONCE and reuse across encode calls."""
    from sentence_transformers import SentenceTransformer
    return _load_cached(model_name, volume_path, SentenceTransformer)


def encode_texts(texts, model_name, batch_size=64, volume_path=None, model=None):
    """Bi-encoder embeddings (L2-normalized). Pass `model` to reuse an already-loaded one."""
    if model is None:
        model = load_bi_encoder(model_name, volume_path)
    return model.encode(list(texts), batch_size=batch_size,
                        normalize_embeddings=True, convert_to_numpy=True,
                        show_progress_bar=True)


def top_k_candidates(similarity_matrix, top_k):
    """Naive top-K from a full incident x problem matrix.

    NOT used by the pipeline — it is the reference implementation
    top_k_candidates_from_embeddings is verified against in tests. Keep it.
    """
    sm = np.asarray(similarity_matrix)
    k = min(top_k, sm.shape[1])
    idx = np.argsort(-sm, axis=1)[:, :k]
    rows = np.arange(sm.shape[0])[:, None]
    return idx, sm[rows, idx]


def top_k_candidates_from_embeddings(incident_embeddings, problem_embeddings,
                                     top_k, chunk_size=1000):
    """Top-K (indices, cosine) per incident WITHOUT materializing the full matrix."""
    inc = np.asarray(incident_embeddings, dtype=np.float32)
    prob = np.asarray(problem_embeddings, dtype=np.float32)
    if prob.shape[0] == 0:
        raise ValueError("problem catalog is empty — nothing to shortlist against")
    k = min(top_k, prob.shape[0])
    out = np.empty((inc.shape[0], k), dtype=np.int64)
    cos = np.empty((inc.shape[0], k), dtype=np.float32)
    for start in range(0, inc.shape[0], chunk_size):
        sims = inc[start:start + chunk_size] @ prob.T
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        rows = np.arange(part.shape[0])[:, None]
        order = np.argsort(-sims[rows, part], axis=1)
        idx = part[rows, order]
        out[start:start + part.shape[0]] = idx
        cos[start:start + part.shape[0]] = sims[rows, idx]
    return out, cos


def rerank(cross_encoder, incident_texts, candidate_texts, candidate_indices,
           chunk_size=5000, batch_size=128):
    """Cross-encoder re-score every (incident, candidate-problem) pair."""
    from tqdm import tqdm

    n, top_k = candidate_indices.shape
    pairs_buffer, scores_buffer = [], []

    for i in tqdm(range(n)):
        incident_text = str(incident_texts[i])
        for j in candidate_indices[i]:
            pairs_buffer.append((incident_text, str(candidate_texts[j])))

        if len(pairs_buffer) >= chunk_size:
            scores_buffer.extend(cross_encoder.predict(
                pairs_buffer, batch_size=batch_size, show_progress_bar=False))
            pairs_buffer = []

    if pairs_buffer:
        scores_buffer.extend(cross_encoder.predict(
            pairs_buffer, batch_size=batch_size, show_progress_bar=False))

    return np.asarray(scores_buffer).reshape(n, top_k)


def to_probabilities(scores):
    """Sigmoid -> [0,1]. Cross-encoder logits are unbounded; sigmoid makes them comparable."""
    from scipy.special import expit
    return expit(np.asarray(scores))
