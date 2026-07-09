"""evaluate.py — incident-level Top-K accuracy for the GBM-ranked candidates."""


def rank_candidates(feature_df, *, number_col, problem_id_col, score_col="gbm_propensity"):
    df = feature_df.sort_values([number_col, score_col], ascending=[True, False]).copy()
    df["rank_within_incident"] = df.groupby(number_col).cumcount() + 1
    df["is_correct"] = df[problem_id_col] == df["candidate_pid"]
    return df


def topk_accuracy(ranked_df, k_values, *, number_col):
    n_incidents = ranked_df[number_col].nunique()
    print("=== Top-K Evaluation (Incident-level) ===")
    results = {}
    for k in k_values:
        correct = int(ranked_df.loc[ranked_df["rank_within_incident"] <= k]
                      .groupby(number_col)["is_correct"].any().sum())
        acc = correct / n_incidents if n_incidents else 0.0
        results[k] = acc
        print(f"Top-{k}: {correct}/{n_incidents} = {acc:.4f}")
    return results
