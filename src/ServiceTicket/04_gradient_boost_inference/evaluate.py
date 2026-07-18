"""evaluate.py — incident-level Top-K accuracy for the GBM-ranked candidates."""


def rank_candidates(feature_df, *, number_col, problem_id_col, score_col="gbm_propensity"):
    df = feature_df.sort_values([number_col, score_col], ascending=[True, False]).copy()
    df["rank_within_incident"] = df.groupby(number_col).cumcount() + 1
    df["is_correct"] = df[problem_id_col] == df["candidate_pid"]
    return df


def topk_accuracy(ranked_df, k_values, *, number_col, problem_id_col="problem_id"):
    # Score only incidents that HAVE a gold problem_id. In production most incidents are
    # unlinked; counting them in the denominator makes every accuracy look worse than it is
    # (they can never be correct, so they only ever divide).
    labeled = ranked_df[ranked_df[problem_id_col].notna()]
    n_incidents = labeled[number_col].nunique()
    unlabeled = ranked_df[number_col].nunique() - n_incidents
    print("=== Top-K Evaluation (Incident-level) ===")
    if unlabeled:
        print(f"  {unlabeled} incident(s) with no gold problem_id excluded from the denominator")
    if not n_incidents:
        raise ValueError("no incidents with a gold problem_id — top-k accuracy is undefined")
    results = {}
    for k in k_values:
        correct = int(labeled.loc[labeled["rank_within_incident"] <= k]
                      .groupby(number_col)["is_correct"].any().sum())
        acc = correct / n_incidents
        results[k] = acc
        print(f"Top-{k}: {correct}/{n_incidents} = {acc:.4f}")
    return results
