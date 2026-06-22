"""
pipeline.py — stage 03 orchestrator (cross-encoder reranking), orchestration only.

  rerank.py   -> load_cross_encoder, top_k_candidates*, rerank, to_probabilities
  evaluate.py -> topk_accuracy, print_baselines

Flow:
  1. load summarized incidents (incident_summary + gold problem_id) and the
     problem-summary catalog (one row per problem, the candidate pool)
  2. shortlist: top-K candidate problems per incident
     (precomputed similarity matrix preferred; else chunked top-K from embeddings)
  3. cross-encoder re-score the shortlisted pairs (chunked, bounded memory)
  4. save raw + sigmoid reranked scores to Volume
  5. optional Top-K accuracy vs prior-stage baselines

ALIGNMENT: the shortlist must be row-aligned with the problem catalog — candidate
index j must mean the same problem in both. The DEFAULT (encode here) guarantees
this by construction; precomputed *_path artifacts are the caller's responsibility.
A `max(index) < len(catalog)` backstop assert guards against gross misalignment.
"""

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

import rerank as rr
import evaluate as ev


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_reranking(spark, cfg):
    rc = cfg["reranking"]
    base = rc.get("volume_base_path")
    top_k = rc.get("top_k", 50)

    t0 = time.perf_counter()
    print(f"[ph03] started {_ts()} | model={rc['model']} | top_k={top_k}")

    # 1. inputs
    df_full = _load_frame(spark, rc.get("input_sql"), rc.get("input_table"),
                          rc.get("input_parquet"), what="incidents")
    prob_summary_pd = _load_frame(spark, None, rc.get("problem_table"),
                                  rc.get("problem_parquet"), what="problem catalog")
    incident_texts = df_full[rc.get("incident_text_col", "incident_summary")].astype(str).tolist()
    candidate_texts = prob_summary_pd[rc.get("problem_text_col", "problem_summary")].astype(str).tolist()
    print(f"[ph03] {len(incident_texts)} incidents x top_k={top_k} "
          f"= {len(incident_texts) * top_k:,} pairs to rerank")

    # 2. shortlist candidates (precomputed artifacts if aligned; else encode here)
    candidate_indices = _candidate_indices(rc, top_k, incident_texts, candidate_texts)
    if candidate_indices.shape[0] != len(incident_texts):
        raise ValueError(
            f"candidate rows {candidate_indices.shape[0]} != incidents {len(incident_texts)}")
    if int(candidate_indices.max()) >= len(candidate_texts):
        raise ValueError(
            f"candidate index {int(candidate_indices.max())} out of range for "
            f"{len(candidate_texts)} problems — embeddings/catalog are misaligned")

    # 3. cross-encoder rerank (chunked)
    model = rr.load_cross_encoder(rc["model"], rc.get("max_length", 512))
    raw_scores = rr.rerank(
        model, incident_texts, candidate_texts, candidate_indices,
        chunk_size=rc.get("chunk_size", 5000), batch_size=rc.get("batch_size", 128))
    sigmoid_scores = rr.to_probabilities(raw_scores)

    # 4. persist
    if rc.get("save_to_volume") and base:
        _save_scores(base, raw_scores, sigmoid_scores)

    # 5. optional eval — a metric failure must NOT discard the saved scores
    ec = rc.get("eval", {})
    if ec.get("enabled"):
        try:
            id_col = rc.get("problem_id_col", "problem_id")
            prob_ids = prob_summary_pd[id_col].to_numpy()
            candidate_pids = prob_ids[candidate_indices]           # (n, top_k)
            true_ids = df_full[id_col].to_numpy()
            ev.topk_accuracy(true_ids, candidate_pids, sigmoid_scores,
                             ec.get("k_values", [5, 10]))
            ev.print_baselines(ec.get("baselines"))
        except Exception as e:
            print(f"[ph03:eval] skipped ({e})")

    total = time.perf_counter() - t0
    print("=" * 60)
    print("Stage 03 complete!")
    print(f"  Incidents reranked:  {len(incident_texts)}")
    print(f"  Candidates/incident: {top_k}")
    print(f"  Total wall-clock: {total:.2f}s  (finished {_ts()})")
    print("=" * 60)
    return raw_scores, sigmoid_scores


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
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
    """Read a 2-D array from .npy or .parquet (stage 01 saves embeddings as parquet)."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path).to_numpy()
    return np.load(path)


def _candidate_indices(rc, top_k, incident_texts, candidate_texts):
    """
    Top-K candidate problem indices per incident (indices into `candidate_texts`).

    Preference order:
      1. precomputed similarity_matrix (.npy/.parquet) -> argsort  [opt-in]
      2. precomputed incident + problem embeddings     -> chunked top-K  [opt-in,
         caller guarantees both are row-aligned with candidate_texts]
      3. DEFAULT: encode incidents + the problem catalog here with the bi-encoder,
         so the shortlist is aligned with `candidate_texts` BY CONSTRUCTION.
    """
    cchunk = rc.get("candidate_chunk_size", 1000)

    sim_path = rc.get("similarity_matrix_path")
    if sim_path and os.path.exists(sim_path):
        print(f"  candidates from precomputed similarity matrix: {sim_path}")
        return rr.top_k_candidates(_load_array(sim_path), top_k)

    inc_path = rc.get("incident_embeddings_path")
    prob_path = rc.get("problem_embeddings_path")
    if inc_path and prob_path and os.path.exists(inc_path) and os.path.exists(prob_path):
        print(f"  candidates from precomputed embeddings (chunked top-K): {inc_path}, {prob_path}")
        print("  NOTE: assuming these are aligned with the problem catalog row order.")
        return rr.top_k_candidates_from_embeddings(
            _load_array(inc_path), _load_array(prob_path), top_k, chunk_size=cchunk)

    # default — self-contained, alignment-safe
    bi_model = rc.get("bi_encoder_model", "all-MiniLM-L6-v2")
    bs = rc.get("bi_encoder_batch_size", 64)
    print(f"  candidates by encoding here with bi-encoder '{bi_model}' "
          f"({len(incident_texts)} incidents x {len(candidate_texts)} problems)")
    inc_emb = rr.encode_texts(incident_texts, bi_model, batch_size=bs)
    prob_emb = rr.encode_texts(candidate_texts, bi_model, batch_size=bs)
    return rr.top_k_candidates_from_embeddings(inc_emb, prob_emb, top_k, chunk_size=cchunk)


def _save_scores(base, raw_scores, sigmoid_scores):
    try:
        os.makedirs(base, exist_ok=True)
        np.save(f"{base}/reranked_scores.npy", raw_scores)
        np.save(f"{base}/reranked_scores_sigmoid.npy", sigmoid_scores)
        print(f"[ph03] reranked scores saved to volume: {base}")
    except Exception as e:
        print(f"[ph03] WARNING: could not save scores to volume ({e})")
