"""features.py — assemble GBM features by joining stage-03 reranked scores with incident + problem attributes (id-based, no positional alignment)."""

import numpy as np
import pandas as pd

# Column order MUST match how the GradientBoostingClassifier was trained.
FEATURE_COLS = ["cosine_sim", "reranker_score", "bs_match"]


def build_feature_matrix(reranked_df, incidents_df, problems_df, *,
                         number_col, problem_id_col, candidate_id_col,
                         incident_bs_col, problem_bs_col, cosine_col, reranker_col,
                         sim_col=None):
    """One row per (incident, candidate): cosine_sim, reranker_score, bs_match, label.

    Joins the stage-03 reranked table (number, candidate_problem_id, cosine, rerank score)
    to the incident gold problem_id/business_service and the problem business_service — all
    by id, so there is no positional-index alignment assumption.

    sim_col is a train-only passthrough (not in FEATURE_COLS, so inference ignores it).
    """
    fm = reranked_df.rename(columns={candidate_id_col: "candidate_pid",
                                     cosine_col: "cosine_sim",
                                     reranker_col: "reranker_score"}).copy()
    fm[number_col] = fm[number_col].astype(str)
    fm["candidate_pid"] = fm["candidate_pid"].astype(str)

    inc_cols = [number_col, problem_id_col]
    if incident_bs_col in incidents_df.columns:
        inc_cols.append(incident_bs_col)
    if sim_col and sim_col in incidents_df.columns and sim_col not in inc_cols:
        inc_cols.append(sim_col)                          # train-only passthrough
    inc = incidents_df[inc_cols].drop_duplicates(number_col).copy()
    inc[number_col] = inc[number_col].astype(str)
    # NOT .astype(str): that turns a missing gold problem_id into the string "nan", which
    # passes .notna() — train.py's "drop rows with no gold problem_id" filter then keeps
    # every unlabeled row and trains on them as negatives.
    inc[problem_id_col] = inc[problem_id_col].map(lambda v: None if pd.isna(v) else str(v))
    fm = fm.merge(inc, on=number_col, how="left")

    if problem_bs_col in problems_df.columns:
        prob = (problems_df[[problem_id_col, problem_bs_col]]
                .drop_duplicates(problem_id_col)
                .rename(columns={problem_id_col: "candidate_pid", problem_bs_col: "_cand_bs"}))
        prob["candidate_pid"] = prob["candidate_pid"].astype(str)
        fm = fm.merge(prob, on="candidate_pid", how="left")

    inc_missing = incident_bs_col not in fm.columns
    cand_missing = "_cand_bs" not in fm.columns
    inc_bs = (np.full(len(fm), "") if inc_missing
              else fm[incident_bs_col].astype(str).str.strip().to_numpy())
    cand_bs = (np.full(len(fm), "") if cand_missing
               else fm["_cand_bs"].astype(str).str.strip().to_numpy())
    fm["bs_match"] = ((inc_bs != "") & (cand_bs != "") & (inc_bs == cand_bs)).astype(int)

    # A missing business_service column does not raise — it quietly makes bs_match 0 for
    # every row, so the model runs on 2 of its 3 features and nothing looks wrong. Say so
    # loudly: the source of a dead feature is always a config/table mismatch upstream.
    if inc_missing or cand_missing:
        missing = []
        if inc_missing:
            missing.append(f"incident '{incident_bs_col}' (not in the incidents table)")
        if cand_missing:
            missing.append(f"problem '{problem_bs_col}' (not in the problem table)")
        print(f"[ph04] WARNING: bs_match is DEAD (always 0) — missing {', '.join(missing)}. "
              f"The GBM is running on {len(FEATURE_COLS) - 1} of {len(FEATURE_COLS)} features.")
    elif fm["bs_match"].sum() == 0:
        print("[ph04] WARNING: bs_match is 0 for every row — the business_service values "
              "never match. Check that incident and problem business_service use the same "
              "vocabulary.")
    else:
        print(f"[ph04] bs_match: {int(fm['bs_match'].sum())}/{len(fm)} rows matched "
              f"({fm['bs_match'].mean() * 100:.1f}%)")

    gold = fm[problem_id_col]
    fm["label"] = (gold.notna() & (fm["candidate_pid"] == gold)).astype(int)
    fm["cosine_sim"] = pd.to_numeric(fm["cosine_sim"], errors="coerce").fillna(0.0)
    fm["reranker_score"] = pd.to_numeric(fm["reranker_score"], errors="coerce").fillna(0.0)
    return fm
