"""pipeline.py — stage 04 orchestrator (Gradient Boosting inference), orchestration only."""

import os
import time
from datetime import datetime

from timing import Timer

import device_log as dev
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
                              what="reranked scores")
    df_full = _load_frame(spark, gc.get("incident_sql"), gc.get("incident_table"),
                          what="incidents")
    # problem_sql (not just problem_table): the problem catalog must carry
    # business_service, and ph02_output_ProblemSummaries does not have it — it holds only
    # problem_id + problem_summary. Without a join, bs_match is 0 for every row and the
    # GBM silently runs on 2 of its 3 features.
    prob_summary_pd = _load_frame(spark, gc.get("problem_sql"), gc.get("problem_table"),
                                  what="problem catalog")

    # train-only passthrough (weak-link filter); carried in both modes, inference ignores it.
    sim_col = cfg.get("gbm_train", {}).get("similarity_col", "semantic_similarity")

    feature_df = feat.build_feature_matrix(
        reranked_df, df_full, prob_summary_pd,
        number_col=num_col, problem_id_col=pid_col, candidate_id_col=cand_col,
        incident_bs_col=gc.get("incident_bs_col", "business_service"),
        problem_bs_col=gc.get("problem_bs_col", "business_service"),
        cosine_col=gc.get("cosine_col", "cosine_sim"),
        reranker_col=gc.get("reranker_col", "rerank_score"),
        sim_col=sim_col)
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
    num_col = gc.get("number_col", "number")
    pid_col = gc.get("problem_id_col", "problem_id")

    t0 = time.perf_counter()
    print(f"[ph04] started {_ts()} | model={os.path.basename(gc['model_path'])}")

    # MLflow run wraps ALL the work so a crash mid-scoring lands as a FAILED run.
    mu = _mlflow_utils()
    with mu.stage_run(cfg, "ph04_gbm_inference") as ml:
        ml.log_params({"model": os.path.basename(gc["model_path"]),
                       "top_n": gc.get("top_n", 10),
                       "batch_size": gc.get("batch_size", 500_000)})
        ml.set_tags({"output_table": gc.get("output_table")})

        timer = Timer()
        feature_df, df_full, prob_summary_pd = build_features(spark, cfg)
        timer.lap("build features")

        # Volume .pkl, loaded once per run — nothing is downloaded and nothing re-fetches
        # the model per batch.
        model = inf.load_model(gc["model_path"])
        timer.lap("load model")
        # sklearn's GradientBoostingClassifier has no CUDA path: scoring stays on the
        # driver CPU even on the GPU runtime. Logged so the GPU-vs-CPU picture is
        # complete rather than silent for this stage.
        ml.log_params({"model_device":
                       dev.cpu_only("[ph04] GBM score", "sklearn GradientBoostingClassifier")})
        with dev.probe("[ph04] score") as p:
            feature_df = inf.score(model, feature_df, batch_size=gc.get("batch_size", 500_000))
        ml.log_metrics(p.metrics())
        timer.lap(f"score {feature_df.shape[0]:,} candidates")

        ranked = ev.rank_candidates(feature_df, number_col=num_col, problem_id_col=pid_col)
        timer.lap("rank")
        # No top-k eval here — it is TRAIN-MODE monitoring only (train.py). Production
        # incidents have no gold problem_id, so the rate would be computed over whatever
        # labeled rows happen to be in the batch and read as pipeline quality.

        linked = lk.build_top10_linking(
            ranked, prob_summary_pd, df_full,
            number_col=num_col, problem_id_col=pid_col,
            problem_desc_col=gc.get("problem_desc_col", "combined_prob_desc"),
            top_n=gc.get("top_n", 10))
        timer.lap("build linking table")
        print(f"[ph04] linking -> live Delta table {gc.get('output_table')} ...")
        out_rows = _save(spark, linked, gc)
        timer.lap("save")
        timer.summary()

        total = time.perf_counter() - t0
        pos = int(feature_df["label"].sum())
        scored = feature_df["gbm_propensity"].astype(float)
        ml.log_metrics({"incidents_linked": linked.shape[0], "output_rows": out_rows,
                        # score spread: a collapsed range means the GBM stopped discriminating
                        "propensity_mean": float(scored.mean()) if len(scored) else 0.0,
                        "propensity_min": float(scored.min()) if len(scored) else 0.0,
                        "propensity_max": float(scored.max()) if len(scored) else 0.0,
                        **mu.step_timings(timer.laps),
                        "feature_rows": feature_df.shape[0],
                        "candidates_scored": feature_df.shape[0],
                        "positives": pos,
                        "positive_rate": (pos / feature_df.shape[0]) if feature_df.shape[0] else 0,
                        "wall_clock_s": total})

    print("=" * 60)
    print("Stage 04 complete! Live Delta table written:")
    print(f"  {gc.get('output_table')}  ({out_rows} rows)")
    print(f"  Incidents linked: {linked.shape[0]}")
    print(f"  Candidates scored: {feature_df.shape[0]}")
    print(f"  Total wall-clock: {total:.2f}s  (finished {_ts()})")
    print("=" * 60)
    return linked


def _load_frame(spark, sql, table, what):
    """Load a frame from a Spark SQL query or a live Delta table. No parquet fallback:
    swallowing the table error and silently reading a stale file is how a run reports
    success on yesterday's data."""
    if sql:
        print(f"  loading {what} via SQL")
        return spark.sql(sql).toPandas()
    if table:
        print(f"  loading {what} from table {table}")
        return spark.table(table).toPandas()
    raise ValueError(f"no input source for {what}: set a sql or table in config")


def _save(spark, pdf, gc):
    """Persist the linking table to its live Delta table."""
    table = gc.get("output_table")
    if not table:
        return 0
    try:
        (spark.createDataFrame(pdf).write.format("delta")
            .option("overwriteSchema", "true").mode("overwrite").saveAsTable(table))
        print(f"[ph04] saved -> {table}  ({pdf.shape[0]} rows, {pdf.shape[1]} cols) [live Delta table]")
        return len(pdf)
    except Exception as e:
        print(f"[ph04] ERROR saving to {table}: {e}")
        raise
