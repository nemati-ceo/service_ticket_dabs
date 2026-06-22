"""
rerank.py — cross-encoder reranking of the top-K candidate problems per incident.

Stage 01 produces a CHEAP semantic shortlist (bi-encoder cosine); this stage
RE-SCORES only the top-K shortlisted (incident, problem) pairs with a heavier
cross-encoder (ms-marco-MiniLM-L-6-v2) that reads both texts jointly and is far
more accurate than cosine — but too slow to run on every pair, hence the
shortlist.

Pairs are scored in CHUNKS (a flat buffer flushed every `chunk_size` pairs) so
the full incident x top_k pair list never has to materialize at once.
"""

import numpy as np


def load_cross_encoder(model_name, max_length=512):
    """Load the cross-encoder reranker."""
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(model_name, max_length=max_length)
    print(f"  Cross-encoder loaded: {model_name} (max_length={max_length})")
    return model


def encode_texts(texts, model_name, batch_size=64):
    """Bi-encoder embeddings (L2-normalized) for the shortlist step."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(list(texts), batch_size=batch_size,
                        normalize_embeddings=True, convert_to_numpy=True,
                        show_progress_bar=True)


def top_k_candidates(similarity_matrix, top_k):
    """Top-K problem indices per incident from a precomputed incident x problem matrix."""
    sm = np.asarray(similarity_matrix)
    k = min(top_k, sm.shape[1])
    return np.argsort(-sm, axis=1)[:, :k]


def top_k_candidates_from_embeddings(incident_embeddings, problem_embeddings,
                                     top_k, chunk_size=1000):
    """
    Top-K problem indices per incident WITHOUT materializing the full
    incident x problem matrix (bounded memory, like stage-02 eval).
    Embeddings are assumed L2-normalized, so cosine == dot product.
    """
    inc = np.asarray(incident_embeddings, dtype=np.float32)
    prob = np.asarray(problem_embeddings, dtype=np.float32)
    k = min(top_k, prob.shape[0])
    out = np.empty((inc.shape[0], k), dtype=np.int64)
    for start in range(0, inc.shape[0], chunk_size):
        sims = inc[start:start + chunk_size] @ prob.T          # (chunk, n_problems)
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]    # k best (unordered)
        rows = np.arange(part.shape[0])[:, None]
        order = np.argsort(-sims[rows, part], axis=1)          # order those k desc
        out[start:start + part.shape[0]] = part[rows, order]
    return out


def rerank(cross_encoder, incident_texts, candidate_texts, candidate_indices,
           chunk_size=5000, batch_size=128):
    """
    Cross-encoder re-score every (incident, candidate-problem) pair.

      incident_texts    : list of incident summaries, len = n_incidents (positional)
      candidate_texts   : list of problem summaries, indexable by candidate index
      candidate_indices : (n_incidents, top_k) array of problem indices to score

    Returns raw cross-encoder scores as an (n_incidents, top_k) float array.
    Pairs are buffered and flushed every `chunk_size` pairs (bounded memory).
    """
    from tqdm import tqdm

    n, top_k = candidate_indices.shape
    pairs_buffer, scores_buffer = [], []

    for i in tqdm(range(n)):
        incident_text = str(incident_texts[i])
        for j in candidate_indices[i]:
            pairs_buffer.append((incident_text, str(candidate_texts[j])))

        # buffer full -> flush a chunk through the cross-encoder
        if len(pairs_buffer) >= chunk_size:
            scores_buffer.extend(cross_encoder.predict(
                pairs_buffer, batch_size=batch_size, show_progress_bar=False))
            pairs_buffer = []

    # final flush
    if pairs_buffer:
        scores_buffer.extend(cross_encoder.predict(
            pairs_buffer, batch_size=batch_size, show_progress_bar=False))

    return np.asarray(scores_buffer).reshape(n, top_k)


def to_probabilities(scores):
    """Sigmoid -> [0,1]. Cross-encoder logits are unbounded; sigmoid makes them comparable."""
    from scipy.special import expit
    return expit(np.asarray(scores))
