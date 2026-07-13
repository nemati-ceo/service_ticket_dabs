"""pipeline.py — stage 04 orchestrator (Gradient Boosting inference), orchestration only."""

import os
import time
from datetime import datetime

import pandas as pd

import features as feat
import inference as inf
import evaluate as ev
import linking as lk


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _mlflow_utils():
    """Load the shared root-level mlflow_utils.py (best-effort logging helpers)."""
    import importlib.util
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "mlflow_utils", os.path.join(root, "mlflow_utils.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build_features(spark, cfg):
    """Assemble the feature matrix. SHARED by train and production — identical columns."""
    gc = cfg["gbm_inference"]
    num_col = gc.get("number_col", "number")
    pid_col = gc.get("problem_id_col", "problem_id")
    cand_col = gc.get("candidate_id_col", "candidate_problem_id")

    # stage-03 reranked scores (number, candidate_problem_id, cosine_sim, rerank_score)
    reranked_df = _load_frame(spark, gc.get("reranked_sql"), gc.get("reranked_table"),
                              gc.get("reranked_parquet"), what="reranked scores")
    df_full = _load_frame(spark, gc.get("incident_sql"), gc.get("incident_table"),
                          gc.get("incident_parquet"), what="incidents")
    # problem_sql (not just problem_table): the problem catalog must carry
    # business_service, and ph02_output_ProblemSummaries does not have it — it holds only
    # problem_id + problem_summary. Without a join, bs_match is 0 for every row and the
    # GBM silently runs on 2 of its 3 features.
    prob_summary_pd = _load_frame(spark, gc.get("problem_sql"), gc.get("problem_table"),
                                  gc.get("problem_parquet"), what="problem catalog")

    feature_df = feat.build_feature_matrix(
        reranked_df, df_full, prob_summary_pd,
        number_col=num_col, problem_id_col=pid_col, candidate_id_col=cand_col,
        incident_bs_col=gc.get("incident_bs_col", "business_service"),
        problem_bs_col=gc.get("problem_bs_col", "business_service"),
        cosine_col=gc.get("cosine_col", "cosine_sim"),
        reranker_col=gc.get("reranker_col", "rerank_score"))
    print(f"[ph04] feature matrix: {feature_df.shape} | positives: {int(feature_df['label'].sum())}")
    return feature_df, df_full, prob_summary_pd


def run_gbm(spark, cfg):
    """Stage 04 entrypoint. mode: train -> fit a model. mode: production -> score."""
    mode = (cfg.get("mode") or "production").lower()
    if mode == "train":
        import train as tr
        feature_df, _, _ = build_features(spark, cfg)
        return tr.run_gbm_train(spark, cfg, feature_df)
    return run_gbm_inference(spark, cfg)


def run_gbm_inference(spark, cfg):
    gc = cfg["gbm_inference"]
    base = gc.get("volume_base_path")
    num_col = gc.get("number_col", "number")
    pid_col = gc.get("problem_id_col", "problem_id")

    t0 = time.perf_counter()
    print(f"[ph04] started {_ts()} | model={os.path.basename(gc['model_path'])}")

    feature_df, df_full, prob_summary_pd = build_features(spark, cfg)

    model = inf.load_model(gc["model_path"])
    feature_df = inf.score(model, feature_df, batch_size=gc.get("batch_size", 500_000))

    ranked = ev.rank_candidates(feature_df, number_col=num_col, problem_id_col=pid_col)
    topk = None
    if gc.get("eval", {}).get("enabled", True):
        try:
            topk = ev.topk_accuracy(ranked, gc["eval"].get("k_values", [1, 5, 7, 10]), number_col=num_col)
        except Exception as e:
            print(f"[ph04:eval] skipped ({e})")

    linked = lk.build_top10_linking(
        ranked, prob_summary_pd, df_full,
        number_col=num_col, problem_id_col=pid_col,
        problem_desc_col=gc.get("problem_desc_col", "combined_prob_desc"),
        top_n=gc.get("top_n", 10))
    _save(spark, linked, gc, base)

    total = time.perf_counter() - t0

    mu = _mlflow_utils()
    with mu.stage_run(cfg, "ph04_gbm_inference") as ml:
        ml.log_params({"model": os.path.basename(gc["model_path"]),
                       "top_n": gc.get("top_n", 10),
                       "batch_size": gc.get("batch_size", 500_000)})
        ml.set_tags({"output_table": gc.get("output_table")})
        pos = int(feature_df["label"].sum())
        ml.log_metrics({"incidents_linked": linked.shape[0],
                        "feature_rows": feature_df.shape[0],
                        "positives": pos,
                        "positive_rate": (pos / feature_df.shape[0]) if feature_df.shape[0] else 0,
                        "wall_clock_s": total, **mu.topk_metrics(topk)})
        if topk:
            ml.log_dict({str(k): v for k, v in topk.items()}, "topk_accuracy.json")

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
