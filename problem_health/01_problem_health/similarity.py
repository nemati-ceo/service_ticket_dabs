"""
similarity.py — semantic similarity scoring for incident<->linked-problem pairs.

NOTE: this is ELEMENT-WISE (each incident vs its OWN linked problem), matching
the original PH01 script. It is NOT all-vs-all, so there is no large-matrix /
OOM risk. normalize_embeddings=True means cosine == dot product.
"""

import numpy as np
import pandas as pd


def pairwise_cosine(combined_embeddings, problem_embeddings):
    """
    Row-aligned cosine similarity: score[i] = cos(incident_i, problem_i).
    Embeddings are assumed L2-normalized, so cosine = row-wise dot product.
    """
    a = np.asarray(combined_embeddings, dtype=np.float32)
    b = np.asarray(problem_embeddings, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    scores = np.einsum("ij,ij->i", a, b)          # row-wise dot product
    return np.clip(scores, -1.0, 1.0).astype(float)


def add_similarity(df, combined_embeddings, problem_embeddings,
                   col_name="semantic_similarity"):
    """Attach the similarity score column to df (row-aligned)."""
    df[col_name] = pairwise_cosine(combined_embeddings, problem_embeddings)
    lo, hi = df[col_name].min(), df[col_name].max()
    print(f"  similarity range: {lo:.4f} .. {hi:.4f}")
    return df


def aggregate_problem_health(df_incidentscore, problem_key="problem_id",
                             sim_col="semantic_similarity",
                             created_col="sys_created_on"):
    """
    Problem-level health = mean incident similarity per problem.
    Returns a dataframe: problem_id, ProblemHealth_Score, Last_Incident_Date.
    """
    df = df_incidentscore.copy()
    if created_col in df.columns:
        df[created_col] = pd.to_datetime(
            df[created_col], format="%m/%d/%Y %H:%M:%S", errors="coerce"
        )
        agg = df.groupby(problem_key).agg(
            {sim_col: "mean", created_col: "max"}
        ).reset_index().rename(columns={
            sim_col: "ProblemHealth_Score",
            created_col: "Last_Incident_Date",
        })
    else:
        agg = df.groupby(problem_key).agg(
            {sim_col: "mean"}
        ).reset_index().rename(columns={sim_col: "ProblemHealth_Score"})
    return agg
