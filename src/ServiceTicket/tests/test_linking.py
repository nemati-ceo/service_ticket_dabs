"""Top-N linking sheet: the score columns and the two similarity columns next to them.

Three numbers with three different meanings end up side by side on this sheet, and the
last review confused two of them. `linked_problem_similarity` is stage 01's cosine
against the problem the incident is ALREADY linked to; `linked_problem_similarity_summarized`
is that same pair scored off the LLM summaries; `top_N_score` is the GBM propensity that
actually produced the ranking. These tests pin which value lands in which column.
"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE03 = os.path.join(ROOT, "03_cross_encoder_rerank")
STAGE04 = os.path.join(ROOT, "04_gradient_boost_inference")
for _d in (STAGE03, STAGE04, ROOT):
    sys.path.insert(0, _d)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lk = _load("linking", os.path.join(STAGE04, "linking.py"))

# INC1 ranks P_A > P_B, INC2 ranks P_B > P_C. summary_similarity is incident-grain, so it
# repeats on every candidate row of the same incident — exactly how stage 03 writes it.
RANKED = pd.DataFrame({
    "number": ["INC1", "INC1", "INC2", "INC2"],
    "candidate_pid": ["P_A", "P_B", "P_B", "P_C"],
    "rank_within_incident": [1, 2, 1, 2],
    "gbm_propensity": [0.91, 0.42, 0.77, 0.10],
    "cosine_sim": [0.72, 0.55, 0.61, 0.13],
    "summary_similarity": [0.65, 0.65, 0.31, 0.31],
})
PROBLEMS = pd.DataFrame({
    "problem_id": ["P_A", "P_B", "P_C"],
    "problem_summary": ["auth service outage", "vpn tunnel drops", "printer queue stuck"],
})
INCIDENTS = pd.DataFrame({
    "number": ["INC1", "INC2"],
    "semantic_similarity": [0.58, 0.24],
})


def _build(ranked=RANKED, incidents=INCIDENTS, top_n=2):
    return lk.build_top10_linking(
        ranked, PROBLEMS, incidents,
        number_col="number", problem_id_col="problem_id",
        problem_desc_col="problem_summary", top_n=top_n)


def test_top_n_scores_land_on_the_matching_rank():
    """top_N_score must carry the propensity of the problem in top_N_pid, not a reorder."""
    out = _build().set_index("number")
    assert out.loc["INC1", "top_1_pid"] == "P_A"
    assert out.loc["INC1", "top_1_score"] == 0.91
    assert out.loc["INC1", "top_2_pid"] == "P_B"
    assert out.loc["INC1", "top_2_score"] == 0.42
    assert out.loc["INC2", "top_1_pid"] == "P_B"
    assert out.loc["INC2", "top_1_score"] == 0.77


def test_similarity_and_score_are_different_columns():
    """The rank-deciding propensity and the summary cosine must not be conflated."""
    out = _build().set_index("number")
    assert out.loc["INC1", "top_1_similarity"] == 0.72
    assert out.loc["INC1", "top_2_similarity"] == 0.55
    assert out.loc["INC2", "top_1_similarity"] == 0.61
    assert out.loc["INC1", "top_1_score"] != out.loc["INC1", "top_1_similarity"]


def test_health_score_is_renamed_not_dropped():
    """The stage-01 passthrough keeps its value under a name that says what it scores."""
    out = _build()
    assert "semantic_similarity" not in out.columns
    assert out.set_index("number").loc["INC1", "linked_problem_similarity"] == 0.58


def test_summarized_twin_is_one_value_per_incident():
    """summary_similarity is incident-grain; it must not be pivoted per rank."""
    out = _build().set_index("number")
    assert out.loc["INC1", "linked_problem_similarity_summarized"] == 0.65
    assert out.loc["INC2", "linked_problem_similarity_summarized"] == 0.31
    assert not any(c.endswith("_summary_similarity") for c in out.columns)


def test_no_summary_column_upstream_is_not_fatal():
    """Stage 03 can be re-run without the new column; the sheet still builds."""
    out = _build(ranked=RANKED.drop(columns=["summary_similarity"]))
    assert "linked_problem_similarity_summarized" not in out.columns
    assert out.loc[0, "top_1_score"] == 0.91


# --- stage 03: the gold-pair summary similarity itself -------------------------------

ph03 = _load("ph03_pipeline", os.path.join(STAGE03, "pipeline.py"))

# Row-normalized so a dot product IS the cosine, which is what the real embeddings are.
INC_EMB = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
PROB_EMB = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
CATALOG = pd.DataFrame({"problem_id": ["P_A", "P_B"]})


def test_gold_similarity_scores_the_linked_pair_by_id():
    """Scores incident i against ITS problem — id lookup, never row position."""
    incidents = pd.DataFrame({"problem_id": ["P_A", "P_B", "P_B"]})
    got = ph03._gold_summary_similarity(incidents, CATALOG, INC_EMB, PROB_EMB, "problem_id")
    # row 2 is inc [1,0] against P_B [0,1] -> orthogonal, 0.0. A positional pairing would
    # have matched it to P_A and returned 1.0.
    assert np.allclose(got, [1.0, 1.0, 0.0])


def test_unknown_or_missing_problem_is_null_not_zero():
    """0.0 reads as 'no match'; these pairs were never scored at all."""
    incidents = pd.DataFrame({"problem_id": ["P_A", "P_NOT_IN_CATALOG", None]})
    got = ph03._gold_summary_similarity(incidents, CATALOG, INC_EMB, PROB_EMB, "problem_id")
    assert got[0] == 1.0
    assert np.isnan(got[1]) and np.isnan(got[2])


def test_missing_id_column_skips_instead_of_raising():
    incidents = pd.DataFrame({"number": ["INC1", "INC2", "INC3"]})
    assert ph03._gold_summary_similarity(
        incidents, CATALOG, INC_EMB, PROB_EMB, "problem_id") is None
