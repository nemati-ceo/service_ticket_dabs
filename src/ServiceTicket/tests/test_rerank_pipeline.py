"""Stage 03 pipeline — the reranked table must be keyed by real incident numbers.

_save_table builds the long (incident, candidate) table stage 04 joins to. If the
incident frame has no `number` column, fabricating 0,1,2... would produce a table that
joins to nothing downstream with no error anywhere — so it must raise instead.
"""

import importlib.util
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE03 = os.path.join(ROOT, "03_cross_encoder_rerank")
sys.path.insert(0, STAGE03)


def _load(name):
    spec = importlib.util.spec_from_file_location(f"ph03_{name}", os.path.join(STAGE03, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_load("rerank")
_load("evaluate")
pl = _load("pipeline")

RC = {"number_col": "number", "problem_id_col": "problem_id",
      "output_table": "cat.sch.ph03_output_RerankedScores"}
PROBS = pd.DataFrame({"problem_id": ["P1", "P2", "P3"]})
IDX = np.array([[0, 1], [2, 0]])          # 2 incidents x top-2 candidates
COS = np.array([[0.9, 0.8], [0.7, 0.6]])
RAW = np.array([[2.0, 1.0], [3.0, 0.5]])
SIG = 1 / (1 + np.exp(-RAW))


class _FakeSpark:
    """Captures the frame handed to createDataFrame instead of writing anywhere."""

    def __init__(self):
        self.written = None

    def createDataFrame(self, pdf):
        self.written = pdf
        outer = self

        class _W:
            def format(self, *_a): return self
            def option(self, *_a): return self
            def mode(self, *_a): return self
            def saveAsTable(self, t): outer.table = t
        return types.SimpleNamespace(write=_W())


def test_raises_when_incident_frame_has_no_number_column():
    df_no_number = pd.DataFrame({"incident_summary": ["a", "b"]})
    with pytest.raises(ValueError, match="no 'number' column"):
        pl._save_table(_FakeSpark(), RC, df_no_number, PROBS, IDX, COS, RAW, SIG)


def test_writes_one_row_per_incident_candidate_pair_and_returns_count():
    df = pd.DataFrame({"number": ["INC1", "INC2"]})
    spark = _FakeSpark()
    n = pl._save_table(spark, RC, df, PROBS, IDX, COS, RAW, SIG)

    assert n == 4                                   # 2 incidents x 2 candidates
    out = spark.written
    assert len(out) == 4
    assert out["number"].tolist() == ["INC1", "INC1", "INC2", "INC2"]
    # candidate ids come from the catalog via the index matrix
    assert out["candidate_problem_id"].tolist() == ["P1", "P2", "P3", "P1"]
    assert out["rerank_rank"].tolist() == [1, 2, 1, 2]
    assert out["rerank_score"].tolist() == [2.0, 1.0, 3.0, 0.5]


def test_row_order_keeps_scores_with_their_own_incident():
    """A transposed reshape here would hand INC2's scores to INC1 and still look sane."""
    df = pd.DataFrame({"number": ["INC1", "INC2"]})
    spark = _FakeSpark()
    pl._save_table(spark, RC, df, PROBS, IDX, COS, RAW, SIG)
    out = spark.written
    inc2 = out[out["number"] == "INC2"]
    assert inc2["rerank_score"].tolist() == [3.0, 0.5]
    assert inc2["cosine_sim"].tolist() == [0.7, 0.6]
