"""features.py — assemble GBM features by joining stage-03 reranked scores with incident + problem attributes (id-based, no positional alignment)."""

import numpy as np
import pandas as pd

# Column order MUST match how the GradientBoostingClassifier was trained.
FEATURE_COLS = ["cosine_sim", "reranker_score", "bs_match"]


def build_feature_matrix(reranked_df, incidents_df, problems_df, *,
                         number_col, problem_id_col, candidate_id_col,
                         incident_bs_col, problem_bs_col, cosine_col, reranker_col):
    """One row per (incident, candidate): cosine_sim, reranker_score, bs_match, label.

    Joins the stage-03 reranked table (number, candidate_problem_id, cosine, rerank score)
    to the incident gold problem_id/business_service and the problem business_service — all
    by id, so there is no positional-index alignment assumption.
    """
    fm = reranked_df.rename(columns={candidate_id_col: "candidate_pid",
                                     cosine_col: "cosine_sim",
                                     reranker_col: "reranker_score"}).copy()
    fm[number_col] = fm[number_col].astype(str)
    fm["candidate_pid"] = fm["candidate_pid"].astype(str)

    inc_cols = [number_col, problem_id_col]
    if incident_bs_col in incidents_df.columns:
        inc_cols.append(incident_bs_col)
    inc = incidents_df[inc_cols].drop_duplicates(number_col).copy()
    inc[number_col] = inc[number_col].astype(str)
    inc[problem_id_col] = inc[problem_id_col].astype(str)
    fm = fm.merge(inc, on=number_col, how="left")

    if problem_bs_col in problems_df.columns:
        prob = (problems_df[[problem_id_col, problem_bs_col]]
                .drop_duplicates(problem_id_col)
                .rename(columns={problem_id_col: "candidate_pid", problem_bs_col: "_cand_bs"}))
        prob["candidate_pid"] = prob["candidate_pid"].astype(str)
        fm = fm.merge(prob, on="candidate_pid", how="left")

    inc_bs = (fm[incident_bs_col].astype(str).str.strip().to_numpy()
              if incident_bs_col in fm.columns else np.full(len(fm), ""))
    cand_bs = (fm["_cand_bs"].astype(str).str.strip().to_numpy()
               if "_cand_bs" in fm.columns else np.full(len(fm), ""))
    fm["bs_match"] = ((inc_bs != "") & (cand_bs != "") & (inc_bs == cand_bs)).astype(int)

    fm["label"] = (fm["candidate_pid"] == fm[problem_id_col].astype(str)).astype(int)
    fm["cosine_sim"] = pd.to_numeric(fm["cosine_sim"], errors="coerce").fillna(0.0)
    fm["reranker_score"] = pd.to_numeric(fm["reranker_score"], errors="coerce").fillna(0.0)
    return fm
