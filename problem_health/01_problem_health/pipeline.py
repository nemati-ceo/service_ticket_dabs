"""pipeline.py — ProblemHealth 01 main pipeline (8 steps), orchestration only."""

import numpy as np
import pandas as pd

import incremental as inc
import embeddings as emb
import similarity as sim
from timing import Timer, ts
from cleaning import clean_text_step
from storage import get_existing_scores, save_incident_scores, save_delta


def run_problem_health(spark, cfg):
    t = cfg["tables"]
    vol = cfg["volume"]
    base_path = vol["base_path"]
    ic = cfg["incremental"]
    key, update_col, hash_cols = ic["key_column"], ic["update_column"], ic["hash_columns"]
    limit = cfg.get("run", {}).get("limit")
    timer = Timer()

    print("[1/8] Loading input data...")
    df_all = spark.table(t["input"]).toPandas()
    if limit:
        df_all = df_all.head(limit)
        print(f"  TEST MODE: limited to {len(df_all)} rows")
    print(f"[1/8] Done. {df_all.shape[0]} rows, {df_all.shape[1]} cols")
    timer.lap("[1/8] load")

    print("[2/8] Checking for existing scores...")
    df_existing = get_existing_scores(spark, t["output_incident"])
    deleted = inc.find_deleted_keys(df_all, df_existing, key)
    if df_existing is not None and deleted:
        df_existing = df_existing[~df_existing[key].astype(str).isin(deleted)].copy()
    df_to_score, df_unchanged = inc.identify_changes(
        df_all, df_existing, key_col=key, hash_cols=hash_cols, update_col=update_col)
    print(f"[2/8] Done. {len(df_to_score)} incidents to score.")
    timer.lap("[2/8] incremental")

    if df_to_score.empty:
        df_incidentscore = df_unchanged if df_unchanged is not None else pd.DataFrame()
        if deleted:
            print(f"[7/8] No new/updated incidents; persisting {len(deleted)} deletion(s)...")
            save_incident_scores(spark, df_incidentscore, t["output_incident"], vol, base_path)
        else:
            print("No new, updated, or deleted incidents. Reusing existing scores.")
    else:
        print("[3/8] Cleaning text...")
        df = clean_text_step(spark, df_to_score, cfg)
        print("[3/8] Done. Text cleaning complete.")
        timer.lap("[3/8] clean")

        print("[4/8] Loading model...")
        model = emb.load_or_save_model(
            cfg["model"]["name"], cfg["model"]["registry_name"],
            backend=cfg["model"].get("backend", "onnx"),
            volume_path=cfg["model"].get("volume_path"))
        timer.lap("[4/8] load model")
        bs = cfg["model"].get("batch_size", 256)
        print("[4/8] Encoding incident embeddings...")
        combined_embeddings = emb.encode_texts(model, df["combined_cleaned_desc"], bs)

        prob_key = cfg.get("aggregation", {}).get("problem_key", "problem_id")
        print(f"[4/8] Encoding problem embeddings (deduplicated by {prob_key})...")
        uniq = df.drop_duplicates(subset=[prob_key]).reset_index(drop=True)
        uniq_emb = emb.encode_texts(model, uniq["combined_prob_desc"], bs)
        pe_by_problem = pd.Series(list(uniq_emb), index=uniq[prob_key])
        problem_embeddings = np.vstack(df[prob_key].map(pe_by_problem).to_numpy())
        print(f"[4/8]   encoded {len(uniq)} unique problems for {len(df)} incidents")
        print(f"[4/8] Done. Encoded {len(combined_embeddings)} incident + "
              f"{len(uniq)} unique problem embeddings.")
        timer.lap("[4/8] encode")

        if vol.get("save_embeddings"):
            try:
                pd.DataFrame(combined_embeddings).to_parquet(f"{base_path}/combined_embeddings.parquet")
                pd.DataFrame(problem_embeddings).to_parquet(f"{base_path}/problem_embeddings.parquet")
                print(f"  Embeddings saved to: {base_path}")
            except Exception as e:
                print(f"  WARNING: could not save embeddings to volume ({e})")

        print("[5/8] Computing cosine similarity...")
        df = sim.add_similarity(df, combined_embeddings, problem_embeddings)
        timer.lap("[5/8] similarity")

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
        timer.lap("[6/8] merge")

        print("[7/8] Saving incident-level scores to Delta...")
        save_incident_scores(spark, df_incidentscore, t["output_incident"], vol, base_path)
        timer.lap("[7/8] save incidents")

    print("[8/8] Aggregating problem-level health scores...")
    problem_health = sim.aggregate_problem_health(df_incidentscore)
    save_delta(spark, problem_health, t["output_problem"])
    if vol.get("save_problem_health"):
        try:
            problem_health.to_parquet(f"{base_path}/ProblemHealth.parquet")
            print(f"  Problem health saved to volume: {base_path}")
        except Exception as e:
            print(f"  WARNING: could not save problem health to volume ({e})")
    timer.lap("[8/8] problem health")

    total = timer.summary()
    print("=" * 60)
    print("Pipeline complete!")
    print(f"  Incidents scored: {len(df_incidentscore)}")
    print(f"  Problems scored:  {len(problem_health)}")
    print(f"  Total wall-clock: {total:.2f}s  (finished {ts()})")
    print("=" * 60)
    return df_incidentscore, problem_health
