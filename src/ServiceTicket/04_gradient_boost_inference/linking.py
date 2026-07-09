"""linking.py — build the Top-N incident->problem linking table (wide format)."""


def build_top10_linking(ranked_df, prob_summary_pd, df_full, *,
                        number_col, problem_id_col, problem_desc_col, top_n=10):
    desc_map = (dict(zip(prob_summary_pd[problem_id_col].astype(str),
                         prob_summary_pd[problem_desc_col].astype(str)))
                if problem_desc_col in prob_summary_pd.columns else {})

    ranked_df = ranked_df.copy()
    ranked_df["problem_description"] = ranked_df["candidate_pid"].map(desc_map).fillna("")

    top = ranked_df[ranked_df["rank_within_incident"] <= top_n]
    pid_wide = top.pivot(index=number_col, columns="rank_within_incident", values="candidate_pid")
    pid_wide.columns = [f"top_{r}_pid" for r in pid_wide.columns]
    desc_wide = top.pivot(index=number_col, columns="rank_within_incident", values="problem_description")
    desc_wide.columns = [f"top_{r}_problem_description" for r in desc_wide.columns]

    info = df_full.drop_duplicates(subset=[number_col]).copy()
    info[number_col] = info[number_col].astype(str)
    info = info.drop(columns=[c for c in ("combined_cleaned_desc_embedding", "problem_embedding")
                              if c in info.columns])

    return (info.merge(pid_wide, on=number_col, how="left")
                .merge(desc_wide, on=number_col, how="left"))
