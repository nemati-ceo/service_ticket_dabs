"""Stage 03 rerank.py — encode reuse and shortlist edge cases.

Model caching moved to the shared root-level model_cache.py (see test_model_cache.py).
The shortlist must behave when the catalog is smaller than top_k, fail clearly when it
is empty, and never materialize the full incident x problem matrix.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE03 = os.path.join(ROOT, "03_cross_encoder_rerank")
sys.path.insert(0, STAGE03)

spec = importlib.util.spec_from_file_location("ph03_rerank", os.path.join(STAGE03, "rerank.py"))
rr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rr)


class _FakeModel:
    """Records how it was constructed and whether it was saved."""
    instances = []

    def __init__(self, source, **kwargs):
        self.source = source
        self.kwargs = kwargs
        self.saved_to = None
        _FakeModel.instances.append(self)

    def save(self, path):
        self.saved_to = path


@pytest.fixture(autouse=True)
def _reset():
    _FakeModel.instances = []


# --- encode_texts must reuse a preloaded model ---------------------------------

class _FakeEncoder:
    loads = 0

    def __init__(self, *a, **k):
        _FakeEncoder.loads += 1

    def encode(self, texts, **kwargs):
        return np.zeros((len(texts), 3), dtype=np.float32)


def test_encode_texts_reuses_a_passed_model(monkeypatch):
    """Encoding incidents then problems must not load the model twice."""
    _FakeEncoder.loads = 0
    monkeypatch.setattr(rr, "load_bi_encoder", lambda *a, **k: _FakeEncoder())

    model = rr.load_bi_encoder("m", None)          # one deliberate load
    rr.encode_texts(["a", "b"], "m", model=model)
    rr.encode_texts(["c"], "m", model=model)

    assert _FakeEncoder.loads == 1                  # not 3


def test_encode_texts_loads_when_no_model_given(monkeypatch):
    _FakeEncoder.loads = 0
    monkeypatch.setattr(rr, "load_bi_encoder", lambda *a, **k: _FakeEncoder())
    rr.encode_texts(["a"], "m")
    assert _FakeEncoder.loads == 1


# --- shortlist edges -----------------------------------------------------------

def test_top_k_clamps_when_catalog_smaller_than_k():
    inc = np.eye(3, 4, dtype=np.float32)
    prob = np.eye(2, 4, dtype=np.float32)              # only 2 problems, ask for 5
    idx, cos = rr.top_k_candidates_from_embeddings(inc, prob, top_k=5)
    assert idx.shape == (3, 2) and cos.shape == (3, 2)


def test_empty_catalog_raises_clearly():
    inc = np.eye(2, 4, dtype=np.float32)
    with pytest.raises(ValueError, match="catalog is empty"):
        rr.top_k_candidates_from_embeddings(inc, np.empty((0, 4), dtype=np.float32), top_k=5)


def test_candidates_are_ordered_best_first():
    inc = np.array([[1.0, 0.0]], dtype=np.float32)
    prob = np.array([[0.0, 1.0], [1.0, 0.0], [0.7, 0.7]], dtype=np.float32)
    idx, cos = rr.top_k_candidates_from_embeddings(inc, prob, top_k=3)
    assert idx[0].tolist() == [1, 2, 0]                # cosine 1.0 > 0.7 > 0.0
    assert cos[0][0] > cos[0][1] > cos[0][2]


# --- the shortlist must never materialize the full incident x problem matrix ----
#
# The deleted top_k_candidates(similarity_matrix, top_k) required the caller to hold an
# n x m matrix and then FULL-argsorted it (int64 = 2x the width). Measured: 7.4 GB at
# 200k incidents x 2207 problems, versus 0.64 GB chunked — that is the driver collapse.
# These pin the chunked path's behaviour so nobody can quietly reintroduce the old one.

def _normalized(rows, dim, seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((rows, dim), dtype=np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 59, 60, 61, 10_000])
def test_result_is_independent_of_chunk_size(chunk_size):
    """Chunking is a memory strategy, not a behaviour change — including at the boundaries
    where the last chunk is partial (59/60/61 against 60 rows)."""
    inc, prob = _normalized(60, 16, 3), _normalized(40, 16, 4)
    base_idx, base_cos = rr.top_k_candidates_from_embeddings(inc, prob, 10, chunk_size=60)
    idx, cos = rr.top_k_candidates_from_embeddings(inc, prob, 10, chunk_size=chunk_size)
    assert idx.tolist() == base_idx.tolist()
    np.testing.assert_allclose(cos, base_cos, atol=1e-6)


def test_returned_cosines_belong_to_the_returned_indices():
    """A chunk-offset bug would return valid-looking indices with another chunk's scores."""
    inc, prob = _normalized(30, 8, 5), _normalized(25, 8, 6)
    idx, cos = rr.top_k_candidates_from_embeddings(inc, prob, 5, chunk_size=4)
    expected = (inc @ prob.T)[np.arange(len(inc))[:, None], idx]
    np.testing.assert_allclose(cos, expected, atol=1e-6)


def test_the_full_matrix_entrypoint_stays_deleted():
    """top_k_candidates(similarity_matrix, ...) forced an n x m allocation on the caller."""
    assert not hasattr(rr, "top_k_candidates"), (
        "the full-matrix shortlist is back — it OOMs the driver at scale; "
        "use top_k_candidates_from_embeddings")
