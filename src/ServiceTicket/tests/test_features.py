"""Stage 04 feature matrix — especially bs_match, which fails SILENTLY.

If the business_service column is absent from either side, bs_match is 0 for every row.
Nothing raises, the GBM just runs on 2 of its 3 features and every downstream number
still looks plausible. These tests make that state visible.
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

spec = importlib.util.spec_from_file_location("features", os.path.join(STAGE04, "features.py"))
feat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feat)

RERANKED = pd.DataFrame({
    "number": ["INC1", "INC1", "INC2", "INC2"],
    "candidate_problem_id": ["PRB_A", "PRB_B", "PRB_A", "PRB_B"],
    "cosine_sim": [0.9, 0.4, 0.3, 0.8],
    "rerank_score": [5.0, -2.0, -1.0, 4.0],
})
INCIDENTS = pd.DataFrame({
    "number": ["INC1", "INC2"],
    "problem_id": ["PRB_A", "PRB_B"],          # gold
    "business_service": ["Email", "Network"],
})


def _build(problems_df, **kw):
    return feat.build_feature_matrix(
        RERANKED.copy(), INCIDENTS.copy(), problems_df,
        number_col="number", problem_id_col="problem_id",
        candidate_id_col="candidate_problem_id",
        incident_bs_col=kw.get("incident_bs_col", "business_service"),
        problem_bs_col=kw.get("problem_bs_col", "business_service"),
        cosine_col="cosine_sim", reranker_col="rerank_score")


def test_label_marks_the_gold_problem():
    problems = pd.DataFrame({"problem_id": ["PRB_A", "PRB_B"],
                             "business_service": ["Email", "Network"]})
    fm = _build(problems)
    gold = fm[fm["label"] == 1]
    assert set(zip(gold["number"], gold["candidate_pid"])) == {("INC1", "PRB_A"),
                                                               ("INC2", "PRB_B")}
    assert fm["label"].sum() == 2


def test_bs_match_is_live_when_the_column_is_present():
    problems = pd.DataFrame({"problem_id": ["PRB_A", "PRB_B"],
                             "business_service": ["Email", "Network"]})
    fm = _build(problems)
    # INC1 (Email) vs PRB_A (Email) -> 1 ; INC1 vs PRB_B (Network) -> 0
    got = {(r.number, r.candidate_pid): r.bs_match for r in fm.itertuples()}
    assert got[("INC1", "PRB_A")] == 1
    assert got[("INC1", "PRB_B")] == 0
    assert got[("INC2", "PRB_B")] == 1
    assert fm["bs_match"].sum() > 0


def test_bs_match_dies_silently_when_the_problem_table_lacks_the_column(capsys):
    """This is the real production config bug: ph02_output_ProblemSummaries has no
    business_service, so bs_match is 0 everywhere and the GBM loses a feature."""
    problems_without_bs = pd.DataFrame({"problem_id": ["PRB_A", "PRB_B"],
                                        "problem_summary": ["a", "b"]})
    fm = _build(problems_without_bs)

    assert fm["bs_match"].sum() == 0                  # the silent failure
    out = capsys.readouterr().out
    assert "bs_match is DEAD" in out                  # but it must SAY so


def test_feature_cols_are_exactly_what_the_model_expects():
    """Column order is load-bearing: the .pkl was fitted on this order."""
    assert feat.FEATURE_COLS == ["cosine_sim", "reranker_score", "bs_match"]
    problems = pd.DataFrame({"problem_id": ["PRB_A", "PRB_B"],
                             "business_service": ["Email", "Network"]})
    fm = _build(problems)
    for col in feat.FEATURE_COLS:
        assert col in fm.columns
        assert fm[col].notna().all()


@pytest.mark.parametrize("bad", ["", "  ", None, np.nan, "None", "nan", "NULL", "<NA>"])
def test_blank_business_service_never_counts_as_a_match(bad):
    """Two blanks are not a match — otherwise every unmapped pair scores bs_match=1.

    The stringified spellings are not paranoia: `astype(str)` renders None as the literal
    "None" on some pandas versions and as NA on others, so this passed locally and failed
    in CI with bs_match=1 for two blank services.
    """
    incidents = INCIDENTS.copy()
    incidents["business_service"] = [bad, "Network"]
    problems = pd.DataFrame({"problem_id": ["PRB_A", "PRB_B"],
                             "business_service": [bad, "Network"]})
    fm = feat.build_feature_matrix(
        RERANKED.copy(), incidents, problems,
        number_col="number", problem_id_col="problem_id",
        candidate_id_col="candidate_problem_id",
        incident_bs_col="business_service", problem_bs_col="business_service",
        cosine_col="cosine_sim", reranker_col="rerank_score")
    inc1_a = fm[(fm["number"] == "INC1") & (fm["candidate_pid"] == "PRB_A")]
    assert int(inc1_a["bs_match"].iloc[0]) == 0
