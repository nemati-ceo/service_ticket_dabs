"""
pipeline.py — ProblemHealth 01 main pipeline (8 steps).
Modularized + incremental (key + content-hash + timestamp) + try/except guards.
Restores volume saves (embeddings / incident scores / problem health).
"""

import time
from datetime import datetime

import numpy as np
import pandas as pd

import incremental as inc
import embeddings as emb
import similarity as sim
from preprocessing import (
    clean_text, clean_shortDescription_text,
    clean_description_text, removeGeneralProblemText,
)


def _ts():
    """Wall-clock timestamp for log lines."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class _Timer:
    """Per-step + total timing. Call lap() after each step; summary() at the end."""

    def __init__(self):
        self.start = time.perf_counter()
        self.mark = self.start
        self.laps = []  # list of (label, seconds)
        print(f"[time] pipeline started at {_ts()}")

    def lap(self, label):
        now = time.perf_counter()
        dt = now - self.mark
        self.mark = now
        self.laps.append((label, dt))
        print(f"[time] {label}: {dt:.2f}s  (elapsed {now - self.start:.2f}s)  @ {_ts()}")
        return dt

    def summary(self):
        total = time.perf_counter() - self.start
        print("-" * 60)
        print(f"[time] STEP TIMINGS (finished {_ts()})")
        for label, dt in self.laps:
            pct = (dt / total * 100) if total else 0
            print(f"[time]   {label:<28} {dt:8.2f}s  {pct:5.1f}%")
        print(f"[time]   {'TOTAL':<28} {total:8.2f}s  100.0%")
        print("-" * 60)
        return total


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


def _clean_with_spark(spark, df_to_score):
    """
    Distributed cleaning: pandas -> Spark DataFrame -> pandas_udf clean -> pandas.
    Runs the per-row cleaning across all executor cores instead of the driver.
    Output columns are identical to apply_cleaning().
    """
    import cleaning_spark as cs
    # Arrow makes the pandas<->Spark round trip fast; harmless if already on.
    try:
        spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    except Exception:
        pass
    sdf = spark.createDataFrame(df_to_score)
    sdf_clean = cs.apply_cleaning_spark(sdf)
    return sdf_clean.toPandas()


def clean_text_step(spark, df_to_score, cfg):
    """
    Dispatch cleaning by config: cleaning.engine = "spark" (distributed) or
    "pandas" (single-thread driver). Spark path falls back to pandas on error
    so a job never dies just because of the cleaning engine.
    """
    engine = cfg.get("cleaning", {}).get("engine", "pandas").lower()
    if engine == "spark":
        print("[3/8]   engine=spark (distributed pandas_udf)")
        try:
            return _clean_with_spark(spark, df_to_score)
        except Exception as e:
            print(f"[3/8]   WARNING: Spark cleaning failed ({e}); falling back to pandas.")
    else:
        print("[3/8]   engine=pandas (single-thread driver)")
    return apply_cleaning(df_to_score)


def get_existing_scores(spark, table):
    try:
        existing = spark.table(table).toPandas()
        print(f"  Loaded {len(existing)} existing scored incidents.")
        return existing
    except Exception as e:
        msg = str(e).lower()
        # Only treat a genuinely-missing table as "first run". Any other error
        # (permissions, transient read failure) must NOT silently trigger a full
        # rescore + overwrite of the target table.
        if any(s in msg for s in (
            "table_or_view_not_found", "not found", "cannot be found",
            "does not exist", "no such table",
        )):
            print("  No existing scores table found. Will score all incidents.")
            return None
        print(f"  ERROR reading existing scores from {table}: {e}")
        raise


def run_problem_health(spark, cfg):
    t = cfg["tables"]
    vol = cfg["volume"]
    base_path = vol["base_path"]
    ic = cfg["incremental"]
    key, update_col, hash_cols = ic["key_column"], ic["update_column"], ic["hash_columns"]
    limit = cfg.get("run", {}).get("limit")
    timer = _Timer()

    # 1. load
    print("[1/8] Loading input data...")
    df_all = spark.table(t["input"]).toPandas()
    if limit:
        df_all = df_all.head(limit)
        print(f"  TEST MODE: limited to {len(df_all)} rows")
    print(f"[1/8] Done. {df_all.shape[0]} rows, {df_all.shape[1]} cols")
    timer.lap("[1/8] load")

    # 2. existing + incremental
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
            # No rescore needed, but deletions must still be persisted to the
            # incident table — df_unchanged already excludes the deleted keys.
            print(f"[7/8] No new/updated incidents; persisting {len(deleted)} deletion(s)...")
            _save_incident_scores(spark, df_incidentscore, t["output_incident"], vol, base_path)
        else:
            print("No new, updated, or deleted incidents. Reusing existing scores.")
    else:
        # 3. clean
        print("[3/8] Cleaning text...")
        df = clean_text_step(spark, df_to_score, cfg)
        print("[3/8] Done. Text cleaning complete.")
        timer.lap("[3/8] clean")

        # 4. model + embeddings
        print("[4/8] Loading model...")
        model = emb.load_or_save_model(
            cfg["model"]["name"], cfg["model"]["registry_name"],
            backend=cfg["model"].get("backend", "onnx"),
            volume_path=cfg["model"].get("volume_path"))
        timer.lap("[4/8] load model")
        bs = cfg["model"].get("batch_size", 256)
        print("[4/8] Encoding incident embeddings...")
        combined_embeddings = emb.encode_texts(model, df["combined_cleaned_desc"], bs)

        # Encode each UNIQUE problem ONCE, then map back to every incident row.
        # Many incidents share the same problem_id, so this avoids re-encoding the
        # same problem text thousands of times (restores Nancy's original L27-31).
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
        timer.lap("[5/8] similarity")

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
        timer.lap("[6/8] merge")

        # 7. save incident scores -> Delta + volume (guarded)
        print("[7/8] Saving incident-level scores to Delta...")
        _save_incident_scores(spark, df_incidentscore, t["output_incident"], vol, base_path)
        timer.lap("[7/8] save incidents")

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
    timer.lap("[8/8] problem health")

    total = timer.summary()
    print("=" * 60)
    print("Pipeline complete!")
    print(f"  Incidents scored: {len(df_incidentscore)}")
    print(f"  Problems scored:  {len(problem_health)}")
    print(f"  Total wall-clock: {total:.2f}s  (finished {_ts()})")
    print("=" * 60)
    return df_incidentscore, problem_health


def _save_incident_scores(spark, df_incidentscore, table, vol, base_path):
    """Persist incident-level scores to Delta (+ volume parquet if enabled)."""
    _save_delta(spark, df_incidentscore, table)
    if vol.get("save_incident_scores"):
        try:
            df_incidentscore.to_parquet(f"{base_path}/IncidentScore_SemanticSimilarity.parquet")
            print(f"  Incident scores saved to volume: {base_path}")
        except Exception as e:
            print(f"  WARNING: could not save incident scores to volume ({e})")


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
