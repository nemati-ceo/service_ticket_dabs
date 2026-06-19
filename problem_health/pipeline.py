"""
pipeline.py — ProblemHealth 01 main pipeline (8 steps).
Modularized + incremental (key + content-hash + timestamp) + try/except guards.
Restores volume saves (embeddings / incident scores / problem health).
"""

import pandas as pd

import incremental as inc
import embeddings as emb
import similarity as sim
from preprocessing import (
    clean_text, clean_shortDescription_text,
    clean_description_text, removeGeneralProblemText,
)


def _clean_inc_short(s): return clean_text(clean_shortDescription_text(str(s)))
def _clean_inc_desc(s):  return clean_text(clean_description_text(str(s)))
def _clean_prob(s):      return removeGeneralProblemText(str(s))


def apply_cleaning(df):
    """Add the 6 cleaned/combined columns. Only called on rows to process."""
    df["cleaned_short_description"] = df["short_description"].astype(str).apply(_clean_inc_short)
    print("[3/8]   - short_description cleaned")
    df["cleaned_description"] = df["description"].astype(str).apply(_clean_inc_desc)
    print("[3/8]   - description cleaned")
    df["combined_cleaned_desc"] = df["cleaned_short_description"] + " " + df["cleaned_description"]
    df["cleaned_prob_short_desc"] = df["problem_id.short_description"].astype(str).apply(_clean_prob)
    print("[3/8]   - problem short_description cleaned")
    df["cleaned_problem_desc"] = df["problem_id.description"].astype(str).apply(_clean_prob)
    print("[3/8]   - problem description cleaned")
    df["combined_prob_desc"] = df["cleaned_prob_short_desc"] + " " + df["cleaned_problem_desc"]
    return df


def get_existing_scores(spark, table):
    try:
        existing = spark.table(table).toPandas()
        print(f"  Loaded {len(existing)} existing scored incidents.")
        return existing
    except Exception:
        print("  No existing scores found. Will score all incidents.")
        return None


def run_problem_health(spark, cfg):
    t = cfg["tables"]
    vol = cfg["volume"]
    base_path = vol["base_path"]
    ic = cfg["incremental"]
    key, update_col, hash_cols = ic["key_column"], ic["update_column"], ic["hash_columns"]
    limit = cfg.get("run", {}).get("limit")

    # 1. load
    print("[1/8] Loading input data...")
    df_all = spark.table(t["input"]).toPandas()
    if limit:
        df_all = df_all.head(limit)
        print(f"  TEST MODE: limited to {len(df_all)} rows")
    print(f"[1/8] Done. {df_all.shape[0]} rows, {df_all.shape[1]} cols")

    # 2. existing + incremental
    print("[2/8] Checking for existing scores...")
    df_existing = get_existing_scores(spark, t["output_incident"])
    deleted = inc.find_deleted_keys(df_all, df_existing, key)
    if df_existing is not None and deleted:
        df_existing = df_existing[~df_existing[key].astype(str).isin(deleted)].copy()
    df_to_score, df_unchanged = inc.identify_changes(
        df_all, df_existing, key_col=key, hash_cols=hash_cols, update_col=update_col)
    print(f"[2/8] Done. {len(df_to_score)} incidents to score.")

    if df_to_score.empty:
        print("No new or updated incidents. Reusing existing scores.")
        df_incidentscore = df_unchanged if df_unchanged is not None else pd.DataFrame()
    else:
        # 3. clean
        print("[3/8] Cleaning text...")
        df = apply_cleaning(df_to_score)
        print("[3/8] Done. Text cleaning complete.")

        # 4. model + embeddings
        print("[4/8] Loading model...")
        model = emb.load_or_save_model(
            cfg["model"]["name"], cfg["model"]["registry_name"],
            backend=cfg["model"].get("backend", "onnx"))
        bs = cfg["model"].get("batch_size", 256)
        print("[4/8] Encoding incident embeddings...")
        combined_embeddings = emb.encode_texts(model, df["combined_cleaned_desc"], bs)
        print("[4/8] Encoding problem embeddings...")
        problem_embeddings = emb.encode_texts(model, df["combined_prob_desc"], bs)
        print(f"[4/8] Done. Encoded {len(combined_embeddings)} embeddings.")

        # save embeddings to volume (guarded)
        if vol.get("save_embeddings"):
            try:
                pd.DataFrame(combined_embeddings).to_parquet(f"{base_path}/combined_embeddings.parquet")
                pd.DataFrame(problem_embeddings).to_parquet(f"{base_path}/problem_embeddings.parquet")
                print(f"  Embeddings saved to: {base_path}")
            except Exception as e:
                print(f"  WARNING: could not save embeddings to volume ({e})")

        # 5. similarity (element-wise)
        print("[5/8] Computing cosine similarity...")
        df = sim.add_similarity(df, combined_embeddings, problem_embeddings)

        # 6. merge new + unchanged
        print("[6/8] Merging new scores with existing...")
        scored_cols = df.columns.tolist()
        if df_unchanged is not None and not df_unchanged.empty:
            common = [c for c in scored_cols if c in df_unchanged.columns]
            df_incidentscore = pd.concat(
                [df[common], df_unchanged[common]], ignore_index=True)
            print(f"[6/8] Done. Merged: {len(df)} new + {len(df_unchanged)} unchanged = {len(df_incidentscore)} total")
        else:
            df_incidentscore = df
            print(f"[6/8] Done. All {len(df_incidentscore)} are newly scored.")

        # 7. save incident scores -> Delta + volume (guarded)
        print("[7/8] Saving incident-level scores to Delta...")
        _save_delta(spark, df_incidentscore, t["output_incident"])
        if vol.get("save_incident_scores"):
            try:
                df_incidentscore.to_parquet(f"{base_path}/IncidentScore_SemanticSimilarity.parquet")
                print(f"  Incident scores saved to volume: {base_path}")
            except Exception as e:
                print(f"  WARNING: could not save incident scores to volume ({e})")

    # 8. aggregate problem-level health -> Delta + volume (guarded)
    print("[8/8] Aggregating problem-level health scores...")
    problem_health = sim.aggregate_problem_health(df_incidentscore)
    _save_delta(spark, problem_health, t["output_problem"])
    if vol.get("save_problem_health"):
        try:
            problem_health.to_parquet(f"{base_path}/ProblemHealth.parquet")
            problem_health.to_csv(f"{base_path}/ProblemHealth.csv", index=False)
            print(f"  Problem health saved to volume: {base_path}")
        except Exception as e:
            print(f"  WARNING: could not save problem health to volume ({e})")

    print("=" * 60)
    print("Pipeline complete!")
    print(f"  Incidents scored: {len(df_incidentscore)}")
    print(f"  Problems scored:  {len(problem_health)}")
    print("=" * 60)
    return df_incidentscore, problem_health


def _save_delta(spark, pdf, table):
    try:
        (spark.createDataFrame(pdf)
            .write.format("delta")
            .option("overwriteSchema", "true")
            .mode("overwrite")
            .saveAsTable(table))
        print(f"  saved -> {table} ({pdf.shape})")
    except Exception as e:
        print(f"  ERROR saving to {table}: {e}")
        raise
