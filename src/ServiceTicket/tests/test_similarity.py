"""Stage 01 similarity — core scoring.

similarity.py produces `semantic_similarity` (the column stage-04 train filters on) and
the per-problem health score. Pure numpy/pandas, so pinned directly with hand-computed
expectations.
"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE01 = os.path.join(ROOT, "01_problem_health")
sys.path.insert(0, STAGE01)

spec = importlib.util.spec_from_file_location("ph01_similarity", os.path.join(STAGE01, "similarity.py"))
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)


# --- pairwise_cosine ----------------------------------------------------------

def test_pairwise_cosine_row_aligned():
    a = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    b = np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    got = sim.pairwise_cosine(a, b)
    assert got.tolist() == [1.0, 0.0, -1.0]        # dot per row of unit vectors


def test_pairwise_cosine_clips_to_unit_range():
    # non-normalized inputs whose dot exceeds 1 must clip to 1.0
    a = np.array([[2.0, 0.0]])
    b = np.array([[3.0, 0.0]])
    assert sim.pairwise_cosine(a, b).tolist() == [1.0]


def test_pairwise_cosine_raises_on_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        sim.pairwise_cosine(np.zeros((2, 3)), np.zeros((2, 4)))


# --- add_similarity -----------------------------------------------------------

def test_add_similarity_attaches_named_column():
    df = pd.DataFrame({"number": ["A", "B"]})
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([[1.0, 0.0], [1.0, 0.0]])
    out = sim.add_similarity(df, a, b)
    assert out["semantic_similarity"].tolist() == [1.0, 0.0]


# --- aggregate_problem_health -------------------------------------------------

def test_problem_health_is_mean_similarity_per_problem():
    df = pd.DataFrame({
        "problem_id": ["P1", "P1", "P2"],
        "semantic_similarity": [0.8, 0.6, 0.9],
    })
    agg = sim.aggregate_problem_health(df).set_index("problem_id")
    assert agg.loc["P1", "ProblemHealth_Score"] == pytest.approx(0.7)   # mean(0.8, 0.6)
    assert agg.loc["P2", "ProblemHealth_Score"] == pytest.approx(0.9)
    assert "Last_Incident_Date" not in agg.columns                      # no created_col


def test_problem_health_keeps_latest_incident_date_when_present():
    df = pd.DataFrame({
        "problem_id": ["P1", "P1"],
        "semantic_similarity": [0.5, 0.7],
        "sys_created_on": ["01/02/2024 10:00:00", "03/04/2024 12:00:00"],
    })
    agg = sim.aggregate_problem_health(df).set_index("problem_id")
    assert agg.loc["P1", "ProblemHealth_Score"] == pytest.approx(0.6)
    assert agg.loc["P1", "Last_Incident_Date"] == pd.Timestamp("2024-03-04 12:00:00")
