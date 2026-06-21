"""
evaluate.py — OFFLINE quality metric for stage 02 (Top-K retrieval accuracy).

Sampled + chunked so it NEVER builds the full incident x problem matrix.
OFF by default (summarization.eval.enabled=false). This is a sanity signal,
not ground truth: it scores summary-embedding matches against the EXISTING
incident -> problem links.
"""

import numpy as np


def run(spark, cfg):
    from sentence_transformers import SentenceTransformer

    sc = cfg["summarization"]
    ev = sc.get("eval", {})
    sample_size = ev.get("sample_size", 5000)
    top_k = ev.get("top_k", 10)
    model_name = ev.get("embedding_model", "all-MiniLM-L6-v2")

    problems = spark.sql(
        f"SELECT problem_id, problem_summary FROM {sc['output_problem']}"
    ).toPandas()

    incidents = spark.sql(f"""
        SELECT s.number, s.incident_summary, CAST(i.problem_id AS STRING) AS true_problem_id
        FROM {sc['output_incident']} s
        JOIN {sc['input_table']} i ON CAST(i.number AS STRING) = s.number
        WHERE i.problem_id IS NOT NULL
    """).toPandas()

    if len(incidents) > sample_size:
        incidents = incidents.sample(n=sample_size, random_state=42).reset_index(drop=True)
    print(f"[ph02:eval] {len(incidents)} sampled incidents vs {len(problems)} problems, top_k={top_k}")

    model = SentenceTransformer(model_name)
    prob_emb = model.encode(problems["problem_summary"].fillna("").tolist(),
                            batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    inc_emb = model.encode(incidents["incident_summary"].fillna("").tolist(),
                           batch_size=64, normalize_embeddings=True, convert_to_numpy=True)

    prob_ids = problems["problem_id"].astype(str).to_numpy()
    true_ids = incidents["true_problem_id"].astype(str).to_numpy()

    # chunked top-k (bounded memory); normalized embeddings -> cosine = dot product
    correct, chunk = 0, 1000
    for start in range(0, len(inc_emb), chunk):
        sims = inc_emb[start:start + chunk] @ prob_emb.T        # (chunk, n_problems)
        k = min(top_k, sims.shape[1])
        topk_idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        for row_i, idxs in enumerate(topk_idx):
            if true_ids[start + row_i] in prob_ids[idxs]:
                correct += 1

    acc = correct / len(incidents) if len(incidents) else 0.0
    print(f"[ph02:eval] Top-{top_k} accuracy: {correct}/{len(incidents)} = {acc:.4f}")
    return acc
