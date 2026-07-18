"""pipeline.py — stage 03 orchestrator (cross-encoder reranking), orchestration only."""

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

import rerank as rr
import evaluate as ev


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


def run_reranking(spark, cfg):
    rc = cfg["reranking"]
    top_k = rc.get("top_k", 50)

    t0 = time.perf_counter()
    print(f"[ph03] started {_ts()} | model={rc['model']} | top_k={top_k}")

    # MLflow run wraps ALL the work so a crash mid-rerank lands as a FAILED run.
    mu = _mlflow_utils()
    with mu.stage_run(cfg, "ph03_reranking") as ml:
        ml.log_params({"model": rc["model"], "top_k": top_k,
                       "max_length": rc.get("max_length", 512),
                       "chunk_size": rc.get("chunk_size"),
                       "batch_size": rc.get("batch_size"),
                       "bi_encoder_model": rc.get("bi_encoder_model"),
                       "limit": rc.get("limit")})
        ml.set_tags({"output_table": rc.get("output_table")})

        df_full = _load_frame(spark, rc.get("input_sql"), rc.get("input_table"),
                              rc.get("input_parquet"), what="incidents")
        if rc.get("limit"):
            df_full = df_full.head(rc["limit"]).reset_index(drop=True)
            print(f"[ph03] TEST MODE: limited to {len(df_full)} incidents")
        prob_summary_pd = _load_frame(spark, None, rc.get("problem_table"),
                                      rc.get("problem_parquet"), what="problem catalog")
        incident_texts = df_full[rc.get("incident_text_col", "incident_summary")].astype(str).tolist()
        candidate_texts = prob_summary_pd[rc.get("problem_text_col", "problem_summary")].astype(str).tolist()
        n_pairs = len(incident_texts) * top_k
        print(f"[ph03] {len(incident_texts)} incidents x top_k={top_k} "
              f"= {n_pairs:,} pairs to rerank | {len(candidate_texts)} problems in catalog")

        candidate_indices, candidate_cosine = _candidate_indices(rc, top_k, incident_texts, candidate_texts)
        if candidate_indices.shape[0] != len(incident_texts):
            raise ValueError(
                f"candidate rows {candidate_indices.shape[0]} != incidents {len(incident_texts)}")
        if candidate_indices.size and int(candidate_indices.max()) >= len(candidate_texts):
            raise ValueError(
                f"candidate index {int(candidate_indices.max())} out of range for "
                f"{len(candidate_texts)} problems — embeddings/catalog are misaligned")

        model = rr.load_cross_encoder(rc["model"], rc.get("max_length", 512),
                                      volume_path=rc.get("model_volume_path"))
        raw_scores = rr.rerank(
            model, incident_texts, candidate_texts, candidate_indices,
            chunk_size=rc.get("chunk_size", 5000), batch_size=rc.get("batch_size", 128))
        sigmoid_scores = rr.to_probabilities(raw_scores)

        out_rows = 0
        if rc.get("output_table"):
            out_rows = _save_table(spark, rc, df_full, prob_summary_pd, candidate_indices,
                                   candidate_cosine, raw_scores, sigmoid_scores)

        ec = rc.get("eval", {})
        topk = None
        if ec.get("enabled"):
            try:
                id_col = rc.get("problem_id_col", "problem_id")
                prob_ids = prob_summary_pd[id_col].to_numpy()
                candidate_pids = prob_ids[candidate_indices]
                true_ids = df_full[id_col].to_numpy()
                topk = ev.topk_accuracy(true_ids, candidate_pids, sigmoid_scores,
                                        ec.get("k_values", [5, 10]))
                ev.print_baselines(ec.get("baselines"))
            except Exception as e:
                print(f"[ph03:eval] skipped ({e})")

        total = time.perf_counter() - t0
        sig = np.asarray(sigmoid_scores, dtype=float)

        ml.log_metrics({"n_incidents": len(incident_texts),
                        "n_problems_catalog": len(candidate_texts),
                        "pairs_reranked": n_pairs, "output_rows": out_rows,
                        # score spread: a collapsed range means the reranker stopped discriminating
                        "rerank_score_mean": float(sig.mean()) if sig.size else 0.0,
                        "rerank_score_min": float(sig.min()) if sig.size else 0.0,
                        "rerank_score_max": float(sig.max()) if sig.size else 0.0,
                        "wall_clock_s": total,
                        **mu.topk_metrics(topk),
                        **mu.baseline_delta_metrics(topk, ec.get("baselines"))})
        if topk:
            ml.log_dict({str(k): v for k, v in topk.items()}, "topk_accuracy.json")

    print("=" * 60)
    print("Stage 03 complete! Live Delta table written:")
    print(f"  {rc.get('output_table')}  ({out_rows} rows)")
    print(f"  Incidents reranked:  {len(incident_texts)}")
    print(f"  Candidates/incident: {top_k}  ({n_pairs:,} pairs)")
    print(f"  Total wall-clock: {total:.2f}s  (finished {_ts()})")
    print("=" * 60)
    return raw_scores, sigmoid_scores


def _save_table(spark, rc, df_full, prob_summary_pd, candidate_indices,
                candidate_cosine, raw_scores, sigmoid_scores):
    """Write reranked scores as a long Delta table: one row per (incident, candidate)."""
    n, k = candidate_indices.shape
    num_col = rc.get("number_col", "number")
    id_col = rc.get("problem_id_col", "problem_id")
    prob_ids = prob_summary_pd[id_col].astype(str).to_numpy()
    # Fail loudly: fabricating 0,1,2... here would silently produce a table that joins
    # to nothing in stage 04, with no error anywhere.
    if num_col not in df_full.columns:
        raise ValueError(
            f"incident frame has no '{num_col}' column — cannot key the reranked table "
            f"(got {list(df_full.columns)})")
    numbers = df_full[num_col].astype(str).to_numpy()
    long = pd.DataFrame({
        num_col: np.repeat(numbers, k),
        "candidate_problem_id": prob_ids[candidate_indices].reshape(-1),
        "rerank_rank": np.tile(np.arange(1, k + 1), n),
        "cosine_sim": np.asarray(candidate_cosine).reshape(-1).astype(float),
        "rerank_score": np.asarray(raw_scores).reshape(-1),
        "rerank_score_sigmoid": np.asarray(sigmoid_scores).reshape(-1),
    })
    table = rc["output_table"]
    try:
        (spark.createDataFrame(long).write.format("delta")
            .option("overwriteSchema", "true").mode("overwrite").saveAsTable(table))
        print(f"[ph03] saved -> {table}  ({long.shape[0]} rows, {long.shape[1]} cols) [live Delta table]")
        return len(long)
    except Exception as e:
        print(f"[ph03] ERROR saving to {table}: {e}")
        raise


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


def _candidate_indices(rc, top_k, incident_texts, candidate_texts):
    """Top-K candidate problem indices per incident (indices into `candidate_texts`)."""
    bi_model = rc.get("bi_encoder_model", "all-MiniLM-L6-v2")
    bs = rc.get("bi_encoder_batch_size", 64)
    print(f"  candidates by encoding with bi-encoder '{bi_model}' "
          f"({len(incident_texts)} incidents x {len(candidate_texts)} problems)")
    vp = rc.get("bi_encoder_volume_path")
    inc_emb = rr.encode_texts(incident_texts, bi_model, batch_size=bs, volume_path=vp)
    prob_emb = rr.encode_texts(candidate_texts, bi_model, batch_size=bs, volume_path=vp)
    return rr.top_k_candidates_from_embeddings(
        inc_emb, prob_emb, top_k, chunk_size=rc.get("candidate_chunk_size", 1000))
