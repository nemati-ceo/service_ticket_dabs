"""Stage 04 TRAIN-ONLY weak-link filter (min_semantic_similarity).

The filter drops incidents whose cosine to their GOLD problem is below the threshold —
a bad label teaches the model to reproduce bad links. It MUST run only in the train
branch: `similarity_col` is a passthrough column that inference never reads, so a
production run can never drop a row on it. These tests pin both halves of that contract.
"""

import importlib.util
import os
import sys

import pandas as pd
import pytest

pytest.importorskip("sklearn")     # train.py imports GradientBoostingClassifier at module load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE04 = os.path.join(ROOT, "04_gradient_boost_inference")
sys.path.insert(0, STAGE04)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(STAGE04, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


feat = _load("features")
tr = _load("train")


def _labeled():
    # 4 incidents, gold cosine spread across the 0.35 threshold.
    return pd.DataFrame({
        "number": ["INC1", "INC2", "INC3", "INC4"],
        "candidate_pid": ["PRB_A", "PRB_B", "PRB_C", "PRB_D"],
        "problem_id": ["PRB_A", "PRB_B", "PRB_C", "PRB_D"],
        "cosine_sim": [0.9, 0.8, 0.7, 0.6],
        "reranker_score": [5.0, 4.0, 3.0, 2.0],
        "bs_match": [1, 1, 0, 0],
        "label": [1, 1, 1, 1],
        "semantic_similarity": [0.90, 0.36, 0.35, 0.10],   # INC4 below 0.35
    })


def test_filter_drops_below_threshold_keeps_at_or_above():
    kept = tr.filter_weak_links(_labeled(), "semantic_similarity", 0.35)
    assert set(kept["number"]) == {"INC1", "INC2", "INC3"}   # 0.35 kept (>=), 0.10 dropped
    assert "INC4" not in set(kept["number"])


def test_none_threshold_is_a_no_op():
    """Production passes no threshold on this frame — nothing may be dropped."""
    df = _labeled()
    kept = tr.filter_weak_links(df, "semantic_similarity", None)
    assert len(kept) == len(df)


def test_missing_sim_column_skips_and_warns(capsys):
    df = _labeled().drop(columns=["semantic_similarity"])
    kept = tr.filter_weak_links(df, "semantic_similarity", 0.35)
    assert len(kept) == len(df)                              # nothing dropped
    assert "SKIPPED" in capsys.readouterr().out


def test_threshold_too_high_raises():
    with pytest.raises(ValueError, match="threshold too high"):
        tr.filter_weak_links(_labeled(), "semantic_similarity", 0.99)


def test_similarity_col_is_never_a_model_feature():
    """Inference scores on FEATURE_COLS only, so the passthrough can never affect prod."""
    assert "semantic_similarity" not in feat.FEATURE_COLS


def test_build_feature_matrix_carries_sim_col_as_passthrough():
    reranked = pd.DataFrame({
        "number": ["INC1", "INC2"],
        "candidate_problem_id": ["PRB_A", "PRB_B"],
        "cosine_sim": [0.9, 0.8],
        "rerank_score": [5.0, 4.0],
    })
    incidents = pd.DataFrame({
        "number": ["INC1", "INC2"],
        "problem_id": ["PRB_A", "PRB_B"],
        "business_service": ["Email", "Network"],
        "semantic_similarity": [0.9, 0.2],
    })
    problems = pd.DataFrame({"problem_id": ["PRB_A", "PRB_B"],
                             "business_service": ["Email", "Network"]})
    fm = feat.build_feature_matrix(
        reranked, incidents, problems,
        number_col="number", problem_id_col="problem_id",
        candidate_id_col="candidate_problem_id",
        incident_bs_col="business_service", problem_bs_col="business_service",
        cosine_col="cosine_sim", reranker_col="rerank_score",
        sim_col="semantic_similarity")
    assert "semantic_similarity" in fm.columns                    # carried through
    assert "semantic_similarity" not in feat.FEATURE_COLS         # but not a feature
