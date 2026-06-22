"""pipeline.py — stage 04 orchestrator (Gradient Boosting inference), orchestration only."""

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

import features as feat
import inference as inf
import evaluate as ev
import linking as lk


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_gbm_inference(spark, cfg):
    gc = cfg["gbm_inference"]
    base = gc.get("volume_base_path")
    top_k = gc.get("top_k", 50)
    num_col = gc.get("number_col", "number")
    pid_col = gc.get("problem_id_col", "problem_id")

    t0 = time.perf_counter()
    print(f"[ph04] started {_ts()} | model={os.path.basename(gc['model_path'])} | top_k={top_k}")

    df_full = _load_frame(spark, gc.get("incident_sql"), gc.get("incident_table"),
                          gc.get("incident_parquet"), what="incidents")
    if gc.get("limit"):
        df_full = df_full.head(gc["limit"]).reset_index(drop=True)
        print(f"[ph04] TEST MODE: limited to {len(df_full)} incidents")
    prob_summary_pd = _load_frame(spark, None, gc.get("problem_table"),
                                  gc.get("problem_parquet"), what="problem catalog")

    n = len(df_full)
    top50_indices = _load_array(gc["top50_indices_path"])[:n]
    similarity_matrix = _load_array(gc["similarity_matrix_path"])[:n]
    reranked_scores = _load_array(gc["reranked_scores_path"])[:n]

    feature_df = feat.build_feature_matrix(
        df_full, prob_summary_pd, top50_indices, similarity_matrix, reranked_scores,
        number_col=num_col, problem_id_col=pid_col,
        incident_bs_col=gc.get("incident_bs_col", "business_service"),
        problem_bs_col=gc.get("problem_bs_col", "problem_id.business_service"),
        top_k=top_k)
    print(f"[ph04] feature matrix: {feature_df.shape} | positives: {int(feature_df['label'].sum())}")

    model = inf.load_model(gc["model_path"])
    feature_df = inf.score(model, feature_df, batch_size=gc.get("batch_size", 500_000))

    ranked = ev.rank_candidates(feature_df, number_col=num_col, problem_id_col=pid_col)
    if gc.get("eval", {}).get("enabled", True):
        try:
            ev.topk_accuracy(ranked, gc["eval"].get("k_values", [1, 5, 7, 10]), number_col=num_col)
        except Exception as e:
            print(f"[ph04:eval] skipped ({e})")

    linked = lk.build_top10_linking(
        ranked, prob_summary_pd, df_full,
        number_col=num_col, problem_id_col=pid_col,
        problem_desc_col=gc.get("problem_desc_col", "combined_prob_desc"),
        top_n=gc.get("top_n", 10))
    _save(spark, linked, gc, base)

    total = time.perf_counter() - t0
    print("=" * 60)
    print("Stage 04 complete!")
    print(f"  Incidents linked: {linked.shape[0]}")
    print(f"  Total wall-clock: {total:.2f}s  (finished {_ts()})")
    print("=" * 60)
    return linked


def _load_frame(spark, sql, table, parquet_path, what):
    """Load a frame from (in order) a Spark SQL query, a Delta table, or parquet."""
    if sql:
        print(f"  loading {what} via SQL")
        return spark.sql(sql).toPandas()
    if table:
        try:
            print(f"  loading {what} from table {table}")
            return spark.table(table).toPandas()
        except Exception as e:
            print(f"  could not read table {table} ({e}); trying parquet...")
    if parquet_path:
        print(f"  loading {what} from parquet {parquet_path}")
        return pd.read_parquet(parquet_path)
    raise ValueError(f"no input source for {what}: set a sql / table / parquet path in config")


def _load_array(path):
    """Read a 2-D array from .npy or .parquet."""
    return pd.read_parquet(path).to_numpy() if path.endswith(".parquet") else np.load(path)


def _save(spark, pdf, gc, base):
    """Persist the linking table to a UC Delta table (+ Volume parquet). Never CSV."""
    table = gc.get("output_table")
    if table:
        try:
            (spark.createDataFrame(pdf).write.format("delta")
                .option("overwriteSchema", "true").mode("overwrite").saveAsTable(table))
            print(f"[ph04] saved -> {table} ({pdf.shape})")
        except Exception as e:
            print(f"[ph04] ERROR saving to {table}: {e}")
            raise
    if gc.get("save_to_volume") and base:
        try:
            os.makedirs(base, exist_ok=True)
            pdf.to_parquet(f"{base}/Incident_Problem_Linking_Top10.parquet", index=False)
            print(f"[ph04] linking table saved to volume: {base}")
        except Exception as e:
            print(f"[ph04] WARNING: could not save to volume ({e})")
