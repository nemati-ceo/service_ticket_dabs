"""
features.py — build the per-(incident, candidate) feature matrix for the GBM.

For each incident and each of its top-K shortlisted candidate problems, assemble
the three model features — cosine similarity (stage 01 bi-encoder), cross-encoder
rerank score (stage 03), and a business-service match flag — plus the gold label
and the id columns used for evaluation and linking.

Vectorized (no Python row loop). The arrays are row-aligned with `df_full`.
"""

import numpy as np
import pandas as pd

# Column order MUST match how the GradientBoostingClassifier was trained.
FEATURE_COLS = ["cosine_sim", "reranker_score", "bs_match"]


def build_feature_matrix(df_full, prob_summary_pd, top50_indices,
                         similarity_matrix, reranked_scores, *,
                         number_col, problem_id_col, incident_bs_col,
                         problem_bs_col, top_k):
    n = len(df_full)
    k = min(top_k, top50_indices.shape[1])
    cand_idx = np.asarray(top50_indices[:n, :k])                 # (n, k) problem indices
    flat = cand_idx.reshape(-1)                                  # (n*k,)
    rows_i = np.repeat(np.arange(n), k)

    prob_pid = prob_summary_pd[problem_id_col].astype(str).to_numpy()
    prob_bs = (prob_summary_pd[problem_bs_col].astype(str).to_numpy()
               if problem_bs_col in prob_summary_pd.columns
               else np.full(len(prob_summary_pd), ""))

    cand_pid = prob_pid[flat]
    cand_bs = prob_bs[flat]
    cosine = np.asarray(similarity_matrix)[rows_i, flat].astype(float)
    reranker = np.asarray(reranked_scores[:n, :k]).reshape(-1).astype(float)

    inc_bs = (df_full[incident_bs_col].astype(str).to_numpy()
              if incident_bs_col in df_full.columns else np.full(n, ""))
    inc_bs_rep = np.repeat(inc_bs, k)
    gt_rep = np.repeat(df_full[problem_id_col].astype(str).to_numpy(), k)
    number_rep = np.repeat(df_full[number_col].astype(str).to_numpy(), k)

    # bs_match: 1 only when both services are present and equal (after strip)
    inc_s = pd.Series(inc_bs_rep, dtype="object").str.strip()
    cand_s = pd.Series(cand_bs, dtype="object").str.strip()
    bs_match = (inc_s.ne("") & cand_s.ne("") & (inc_s.values == cand_s.values)).astype(int)

    return pd.DataFrame({
        number_col: number_rep,
        "candidate_pid": cand_pid,
        problem_id_col: gt_rep,
        "cosine_sim": cosine,
        "reranker_score": reranker,
        "bs_match": bs_match.to_numpy(),
        "label": (cand_pid == gt_rep).astype(int),
    })
