"""Tests for the pure Top-K accuracy metrics (stages 03 & 04).

These are the numbers the pipeline now logs to MLflow, so they're worth pinning
with hand-computed expected values. Both stages share the basename evaluate.py,
so each is loaded by path under a distinct module name.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import load_by_path

ev03 = load_by_path("evaluate03", "03_cross_encoder_rerank/evaluate.py")
ev04 = load_by_path("evaluate04", "04_gradient_boost_inference/evaluate.py")


# --- stage 03: reranked Top-K -------------------------------------------------

def test_ph03_topk_accuracy_handworked():
    true_ids = ["A", "B"]
    cand_ids = [["X", "Y", "A"],     # gold "A" sits last, behind low score
                ["B", "Z", "W"]]     # gold "B" is the top score
    scores = np.array([[0.9, 0.2, 0.1],
                       [0.5, 0.4, 0.3]])
    res = ev03.topk_accuracy(true_ids, cand_ids, scores, k_values=[1, 2, 3])
    assert res == {1: 0.5, 2: 0.5, 3: 1.0}


def test_ph03_k_larger_than_candidates_is_clamped():
    true_ids = ["A", "B"]
    cand_ids = [["X", "Y", "A"], ["B", "Z", "W"]]
    scores = np.array([[0.9, 0.2, 0.1], [0.5, 0.4, 0.3]])
    res = ev03.topk_accuracy(true_ids, cand_ids, scores, k_values=[5])
    assert res[5] == 1.0             # k=5 with only 3 candidates == Top-3


def test_ph03_ranking_uses_scores_not_position():
    # gold is the LAST candidate but has the HIGHEST score -> Top-1 hit
    res = ev03.topk_accuracy(
        ["A"], [["X", "Y", "A"]], np.array([[0.1, 0.2, 0.9]]), k_values=[1])
    assert res[1] == 1.0


def test_ph03_empty_is_zero_not_error():
    res = ev03.topk_accuracy([], np.empty((0, 0)), np.empty((0, 0)), k_values=[1])
    assert res[1] == 0.0


# --- stage 04: incident-level Top-K -------------------------------------------

def _ph04_frame():
    # rows intentionally out of order to prove rank_candidates sorts them
    return pd.DataFrame([
        {"number": 1, "problem_id": "P1", "candidate_pid": "P3", "gbm_propensity": 0.1},
        {"number": 2, "problem_id": "P2", "candidate_pid": "P7", "gbm_propensity": 0.7},
        {"number": 1, "problem_id": "P1", "candidate_pid": "P9", "gbm_propensity": 0.9},
        {"number": 1, "problem_id": "P1", "candidate_pid": "P1", "gbm_propensity": 0.5},
        {"number": 2, "problem_id": "P2", "candidate_pid": "P2", "gbm_propensity": 0.8},
    ])


def test_ph04_rank_candidates_orders_and_flags():
    ranked = ev04.rank_candidates(_ph04_frame(), number_col="number", problem_id_col="problem_id")
    inc1 = ranked[ranked.number == 1].sort_values("rank_within_incident")
    assert list(inc1["candidate_pid"]) == ["P9", "P1", "P3"]      # by score desc
    assert list(inc1["rank_within_incident"]) == [1, 2, 3]
    assert list(inc1["is_correct"]) == [False, True, False]       # only P1 matches gold


def test_ph04_topk_match_rate_handworked():
    ranked = ev04.rank_candidates(_ph04_frame(), number_col="number", problem_id_col="problem_id")
    res = ev04.topk_match_rate(ranked, k_values=[1, 2], number_col="number")
    # incident 1: correct at rank 2 ; incident 2: correct at rank 1
    assert res == {1: 0.5, 2: 1.0}


def test_ph04_empty_raises_instead_of_reporting_zero():
    """0.0 reads as "the model scored nothing right"; nothing was scored at all. The
    caller catches this and prints the eval as skipped."""
    empty = _ph04_frame().iloc[0:0]
    ranked = ev04.rank_candidates(empty, number_col="number", problem_id_col="problem_id")
    with pytest.raises(ValueError, match="top-k match rate is undefined"):
        ev04.topk_match_rate(ranked, k_values=[1], number_col="number")
