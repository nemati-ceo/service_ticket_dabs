"""Stage 04 features/evaluate/inference — the label and the denominator.

Two silent-wrong bugs are pinned here: a missing gold problem_id must stay missing (it
used to become the string "nan", which survives .notna() and got trained on as a
negative), and top-k accuracy must only count incidents that HAVE a gold problem_id.
"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE04 = os.path.join(ROOT, "04_gradient_boost_inference")
sys.path.insert(0, STAGE04)


def _load(name):
    spec = importlib.util.spec_from_file_location(f"ph04_{name}",
                                                  os.path.join(STAGE04, f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


feat = _load("features")
sys.modules["features"] = feat          # inference.py does `from features import ...`
ev = _load("evaluate")
inf = _load("inference")


COLS = dict(number_col="number", problem_id_col="problem_id",
            candidate_id_col="candidate_problem_id",
            incident_bs_col="business_service", problem_bs_col="business_service",
            cosine_col="cosine_sim", reranker_col="rerank_score")


def _reranked(pairs):
    return pd.DataFrame({"number": [n for n, _ in pairs],
                         "candidate_problem_id": [c for _, c in pairs],
                         "cosine_sim": 0.5, "rerank_score": 0.5})


# --- the gold problem_id must survive as NULL ----------------------------------

def test_missing_gold_problem_id_stays_null_not_the_string_nan():
    reranked = _reranked([("INC1", "P1"), ("INC2", "P1")])
    incidents = pd.DataFrame({"number": ["INC1", "INC2"],
                              "problem_id": ["P1", None],
                              "business_service": ["a", "a"]})
    problems = pd.DataFrame({"problem_id": ["P1"], "business_service": ["a"]})

    fm = feat.build_feature_matrix(reranked, incidents, problems, **COLS)

    assert fm[fm["number"] == "INC2"]["problem_id"].isna().all()      # NOT the string "nan"


def test_unlabeled_rows_are_not_positives():
    reranked = _reranked([("INC1", "P1"), ("INC2", "P2")])
    incidents = pd.DataFrame({"number": ["INC1", "INC2"], "problem_id": ["P1", np.nan]})
    problems = pd.DataFrame({"problem_id": ["P1", "P2"]})

    fm = feat.build_feature_matrix(reranked, incidents, problems, **COLS)
    assert fm.set_index("number")["label"].to_dict() == {"INC1": 1, "INC2": 0}


def test_train_notna_filter_actually_drops_the_unlabeled_row():
    """The regression this guards: .notna() on "nan" is True, so nothing got dropped."""
    reranked = _reranked([("INC1", "P1"), ("INC2", "P1")])
    incidents = pd.DataFrame({"number": ["INC1", "INC2"], "problem_id": ["P1", None]})
    problems = pd.DataFrame({"problem_id": ["P1"]})

    fm = feat.build_feature_matrix(reranked, incidents, problems, **COLS)
    assert len(fm[fm["problem_id"].notna()]) == 1


def test_numeric_problem_ids_are_still_stringified():
    reranked = _reranked([("INC1", "77")])
    incidents = pd.DataFrame({"number": ["INC1"], "problem_id": [77]})
    problems = pd.DataFrame({"problem_id": ["77"]})

    fm = feat.build_feature_matrix(reranked, incidents, problems, **COLS)
    assert fm.loc[0, "problem_id"] == "77" and fm.loc[0, "label"] == 1


# --- top-k denominator ---------------------------------------------------------

def _ranked(rows):
    df = pd.DataFrame(rows, columns=["number", "problem_id", "candidate_pid", "gbm_propensity"])
    return ev.rank_candidates(df, number_col="number", problem_id_col="problem_id")


def test_unlabeled_incidents_are_excluded_from_the_denominator():
    ranked = _ranked([("INC1", "P1", "P1", 0.9), ("INC1", "P1", "P2", 0.1),
                      ("INC2", None, "P1", 0.9), ("INC2", None, "P2", 0.1)])
    # 1 labeled incident, ranked correctly -> 1.0, not 0.5
    assert ev.topk_accuracy(ranked, [1], number_col="number") == {1: 1.0}


def test_all_unlabeled_raises_instead_of_reporting_zero():
    ranked = _ranked([("INC1", None, "P1", 0.9)])
    with pytest.raises(ValueError, match="no incidents with a gold problem_id"):
        ev.topk_accuracy(ranked, [1], number_col="number")


def test_topk_is_cumulative_over_rank():
    ranked = _ranked([("INC1", "P2", "P1", 0.9), ("INC1", "P2", "P2", 0.1)])
    assert ev.topk_accuracy(ranked, [1, 2], number_col="number") == {1: 0.0, 2: 1.0}


def test_excluded_count_is_reported(capsys):
    ranked = _ranked([("INC1", "P1", "P1", 0.9), ("INC2", None, "P1", 0.9)])
    ev.topk_accuracy(ranked, [1], number_col="number")
    assert "1 incident(s) with no gold problem_id excluded" in capsys.readouterr().out


# --- inference guards ----------------------------------------------------------

class _Model:
    n_features_in_ = len(feat.FEATURE_COLS)

    def predict_proba(self, X):
        return np.column_stack([np.zeros(len(X)), np.full(len(X), 0.7)])


def _features(n=3):
    return pd.DataFrame({c: np.linspace(0, 1, n) for c in feat.FEATURE_COLS})


def test_score_batches_cover_every_row():
    out = inf.score(_Model(), _features(5), batch_size=2)
    assert out["gbm_propensity"].tolist() == [0.7] * 5


def test_missing_feature_column_raises():
    df = _features().drop(columns=[feat.FEATURE_COLS[0]])
    with pytest.raises(ValueError, match="missing"):
        inf.score(_Model(), df)


def test_model_feature_count_mismatch_raises():
    class _Wrong(_Model):
        n_features_in_ = len(feat.FEATURE_COLS) + 1

    with pytest.raises(ValueError, match="out of sync"):
        inf.score(_Wrong(), _features())


def test_missing_model_file_says_what_to_do(tmp_path):
    with pytest.raises(FileNotFoundError, match="mode: train"):
        inf.load_model(str(tmp_path / "nope.pkl"))
