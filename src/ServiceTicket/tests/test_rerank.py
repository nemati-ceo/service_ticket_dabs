"""Stage 03 reranking: chunked scoring must not scramble the score->pair alignment.

rerank() buffers (incident, candidate) pairs, flushes to the cross-encoder whenever the
buffer passes chunk_size, then reshapes the flat scores to (n_incidents, top_k). That
reshape is only correct if scores come back in exactly the order pairs went in — a
chunk-boundary bug would silently attribute one incident's scores to another, and every
downstream number (rerank, GBM, linking) would still look plausible.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE03 = os.path.join(ROOT, "03_cross_encoder_rerank")
sys.path.insert(0, STAGE03)

spec = importlib.util.spec_from_file_location("rerank", os.path.join(STAGE03, "rerank.py"))
rr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rr)


class FakeCrossEncoder:
    """Scores a pair as hash-free arithmetic on its texts, so the expected value is exact."""

    def __init__(self):
        self.calls = 0

    def predict(self, pairs, batch_size=128, show_progress_bar=False):
        self.calls += 1
        # score = incident index * 100 + candidate index, both parsed back out of the text
        return np.array([int(a.split("_")[1]) * 100 + int(b.split("_")[1])
                         for a, b in pairs], dtype=float)


N, K = 7, 5
INCIDENTS = [f"inc_{i}" for i in range(N)]
PROBLEMS = [f"prob_{j}" for j in range(20)]
rng = np.random.default_rng(0)
CANDIDATES = np.stack([rng.choice(len(PROBLEMS), K, replace=False) for _ in range(N)])

EXPECTED = np.array([[i * 100 + CANDIDATES[i][j] for j in range(K)] for i in range(N)],
                    dtype=float)


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 5000])
def test_chunking_preserves_alignment(chunk_size):
    """Any chunk_size must give the identical score matrix — including chunk_size=1."""
    scores = rr.rerank(FakeCrossEncoder(), INCIDENTS, PROBLEMS, CANDIDATES,
                       chunk_size=chunk_size, batch_size=2)
    assert scores.shape == (N, K)
    np.testing.assert_array_equal(scores, EXPECTED)


def test_chunk_size_actually_chunks():
    """Guard the guard: chunk_size=1 must really flush many times, not one big call."""
    ce = FakeCrossEncoder()
    rr.rerank(ce, INCIDENTS, PROBLEMS, CANDIDATES, chunk_size=1, batch_size=2)
    assert ce.calls > 1


def test_sigmoid_is_monotonic_so_ranking_is_unchanged():
    """Stage 03 evals on sigmoid but the GBM consumes raw logits — order must agree."""
    raw = np.array([[-3.0, 0.5, 2.0, -1.0]])
    sig = rr.to_probabilities(raw)
    assert np.array_equal(np.argsort(-raw, axis=1), np.argsort(-sig, axis=1))
    assert ((sig >= 0) & (sig <= 1)).all()


def test_top_k_candidates_from_embeddings_matches_full_matrix():
    """The chunked shortlist must equal argsort over the full cosine matrix."""
    inc = rng.normal(size=(9, 8)).astype(np.float32)
    prob = rng.normal(size=(15, 8)).astype(np.float32)
    inc /= np.linalg.norm(inc, axis=1, keepdims=True)
    prob /= np.linalg.norm(prob, axis=1, keepdims=True)

    idx_chunked, cos_chunked = rr.top_k_candidates_from_embeddings(inc, prob, 5, chunk_size=2)
    idx_full, cos_full = rr.top_k_candidates(inc @ prob.T, 5)

    np.testing.assert_array_equal(idx_chunked, idx_full)
    np.testing.assert_allclose(cos_chunked, cos_full, atol=1e-6)
