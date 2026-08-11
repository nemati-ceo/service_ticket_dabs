"""linking.py — build the Top-N incident->problem linking table (wide format)."""


def build_top10_linking(ranked_df, prob_summary_pd, df_full, *,
                        number_col, problem_id_col, problem_desc_col, top_n=10,
                        score_col="gbm_propensity", similarity_col="cosine_sim"):
    desc_map = (dict(zip(prob_summary_pd[problem_id_col].astype(str),
                         prob_summary_pd[problem_desc_col].astype(str)))
                if problem_desc_col in prob_summary_pd.columns else {})

    ranked_df = ranked_df.copy()
    ranked_df["problem_description"] = ranked_df["candidate_pid"].map(desc_map).fillna("")

    top = ranked_df[ranked_df["rank_within_incident"] <= top_n]

    def _wide(values, suffix):
        w = top.pivot(index=number_col, columns="rank_within_incident", values=values)
        w.columns = [f"top_{r}_{suffix}" for r in w.columns]
        return w.reset_index()

    info = df_full.drop_duplicates(subset=[number_col]).copy()
    info[number_col] = info[number_col].astype(str)
    info = info.drop(columns=[c for c in ("combined_cleaned_desc_embedding", "problem_embedding")
                              if c in info.columns])
    # Stage 01's score rides in on df_full. It scores the incident against the problem it
    # is ALREADY linked to, NOT against any of the top-N — left as bare
    # "semantic_similarity" next to ten top_N_score columns it reads as "the" similarity,
    # which is exactly how the two got confused when this sheet was reviewed.
    info = info.rename(columns={"semantic_similarity": "linked_problem_similarity"})

    out = info
    for wide in (_wide("candidate_pid", "pid"),
                 _wide("problem_description", "problem_description"),
                 # Two different numbers per candidate, on purpose. `score` is the GBM
                 # propensity that DECIDED the rank (a classifier output over cosine +
                 # reranker + business-service match). `similarity` is the plain cosine
                 # between the two LLM summaries — the only column on this sheet that is
                 # the same KIND of number as linked_problem_similarity_summarized, so it
                 # is the one to compare against when asking what summarization is worth.
                 _wide(score_col, "score"),
                 _wide(similarity_col, "similarity")):
        out = out.merge(wide, on=number_col, how="left")

    # Summarized twin of linked_problem_similarity: stage 03 scores the SAME linked pair
    # off the LLM summaries. Incident grain (identical on every candidate row), so it is
    # taken once per incident rather than pivoted per rank.
    if "summary_similarity" in ranked_df.columns:
        twin = (ranked_df.groupby(number_col)["summary_similarity"].first()
                .rename("linked_problem_similarity_summarized").reset_index())
        out = out.merge(twin, on=number_col, how="left")
    return out
