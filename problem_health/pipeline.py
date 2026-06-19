"""
pipeline.py — ProblemHealth 01 main pipeline (8 steps).

Reconstructed from the original ProblemHealth 01 script + modularized.
Incremental processing now uses incremental.py (key + content-hash + timestamp),
so only NEW or CHANGED incidents are cleaned/scored; unchanged scores are reused.

>>> VERIFY against your original where marked  # CHECK
"""

import pandas as pd

import incremental as inc
import embeddings as emb
import similarity as sim
from preprocessing import (
    clean_text, clean_shortDescription_text,
    clean_description_text, removeGeneralProblemText,
)


# ---- text cleaning helpers (apply per column) -----------------------------
def _clean_inc_short(s): return clean_text(clean_shortDescription_text(str(s)))
def _clean_inc_desc(s):  return clean_text(clean_description_text(str(s)))
def _clean_prob(s):      return removeGeneralProblemText(str(s))


def apply_cleaning(df):
    """Add the 6 cleaned/combined columns. Only call on rows to process."""
    df["cleaned_short_description"] = df["short_description"].astype(str).apply(_clean_inc_short)
    df["cleaned_description"]       = df["description"].astype(str).apply(_clean_inc_desc)
    df["combined_cleaned_desc"]     = df["cleaned_short_description"] + " " + df["cleaned_description"]
    df["cleaned_prob_short_desc"]   = df["problem_id.short_description"].astype(str).apply(_clean_prob)
    df["cleaned_problem_desc"]      = df["problem_id.description"].astype(str).apply(_clean_prob)
    df["combined_prob_desc"]        = df["cleaned_prob_short_desc"] + " " + df["cleaned_problem_desc"]
    return df


def get_existing_scores(spark, table):
    """Load previously scored incidents from the output table, or None."""
    try:
        existing = spark.table(table).toPandas()
        print(f"  Loaded {len(existing)} existing scored incidents.")
        return existing
    except Exception:
        print("  No existing scores found. Will score all incidents.")
        return None


def run_problem_health(spark, cfg):
    """Run the full pipeline. cfg = parsed config dict."""
    t = cfg["tables"]
    inc_cfg = cfg["incremental"]
    key = inc_cfg["key_column"]
    update_col = inc_cfg["update_column"]
    hash_cols = inc_cfg["hash_columns"]
    limit = cfg.get("run", {}).get("limit")

    # 1. load input
    print("[1/8] Loading input data...")
    df_all = spark.table(t["input"]).toPandas()
    if limit:
        df_all = df_all.head(limit)
        print(f"  TEST MODE: limited to {len(df_all)} rows")
    print(f"[1/8] Done. {df_all.shape[0]} rows, {df_all.shape[1]} cols")

    # 2. existing scores + incremental change detection
    print("[2/8] Checking for existing scores...")
    df_existing = get_existing_scores(spark, t["output_incident"])
    deleted = inc.find_deleted_keys(df_all, df_existing, key)
    if df_existing is not None and deleted:
        df_existing = df_existing[~df_existing[key].astype(str).isin(deleted)].copy()
    df_to_score, df_unchanged = inc.identify_changes(
        df_all, df_existing, key_col=key,
        hash_cols=hash_cols, update_col=update_col,
    )
    print(f"[2/8] Done. {len(df_to_score)} incidents to score.")

    if df_to_score.empty:
        print("No new or updated incidents. Reusing existing scores.")
        df_incidentscore = df_unchanged
    else:
        # 3. clean (only new/updated)
        print("[3/8] Cleaning text...")
        df = apply_cleaning(df_to_score)
        print("[3/8] Done. Text cleaning complete.")

        # 4. model + embeddings
        print("[4/8] Loading model...")
        model = emb.load_or_save_model(
            cfg["model"]["name"], cfg["model"]["registry_name"],
            backend=cfg["model"].get("backend", "onnx"),
        )
        bs = cfg["model"].get("batch_size", 256)
        print("[4/8] Encoding incident embeddings...")
        combined_embeddings = emb.encode_texts(model, df["combined_cleaned_desc"], bs)
        print("[4/8] Encoding problem embeddings...")
        problem_embeddings = emb.encode_texts(model, df["combined_prob_desc"], bs)

        # 5. similarity (element-wise)
        print("[5/8] Computing cosine similarity...")
        df = sim.add_similarity(df, combined_embeddings, problem_embeddings)

        # 6. merge new + unchanged
        print("[6/8] Merging new scores with existing...")
        if df_unchanged is not None and not df_unchanged.empty:
            common = [c for c in df.columns if c in df_unchanged.columns]
            df_incidentscore = pd.concat(
                [df[common], df_unchanged[common]], ignore_index=True
            )
        else:
            df_incidentscore = df
        print(f"[6/8] Done. {len(df_incidentscore)} total scored.")

        # 7. save incident scores
        print("[7/8] Saving incident-level scores...")
        _save_delta(spark, df_incidentscore, t["output_incident"])

    # 8. aggregate to problem-level health
    print("[8/8] Aggregating problem-level health scores...")
    problem_health = sim.aggregate_problem_health(df_incidentscore)
    _save_delta(spark, problem_health, t["output_problem"])

    print("=" * 60)
    print(f"  Incidents scored: {len(df_incidentscore)}")
    print(f"  Problems scored:  {len(problem_health)}")
    print("=" * 60)
    return df_incidentscore, problem_health


def _save_delta(spark, pdf, table):
    """Overwrite a Delta table from a pandas dataframe."""
    (spark.createDataFrame(pdf)
        .write.format("delta")
        .option("overwriteSchema", "true")
        .mode("overwrite")
        .saveAsTable(table))
    print(f"  saved -> {table} ({pdf.shape})")
