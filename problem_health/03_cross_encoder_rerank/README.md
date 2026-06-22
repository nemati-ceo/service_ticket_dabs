# 03 — Cross-encoder Reranking

Stage 3 of the pipeline. Takes the cheap bi-encoder shortlist (top-K candidate
problems per incident) and **re-scores only those pairs** with a heavier
cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) that reads the incident
and problem text *jointly* — far more accurate than cosine, but too slow to run
on every pair, which is why it only runs on the shortlist.

## Modules
| File | Role |
|---|---|
| `run.py` | shared entry point (`run.stage03()`) — loads root `config.yml` |
| `pipeline.py` | orchestrator: load inputs → shortlist → rerank → save → eval |
| `rerank.py` | cross-encoder load, top-K candidate selection, chunked predict, sigmoid |
| `evaluate.py` | optional Top-K hit-rate vs prior-stage baselines |

## Flow
1. **Load** summarized incidents (`incident_summary` + gold `problem_id`) and the
   problem-summary catalog (the candidate pool).
2. **Shortlist** the top-K candidate problems per incident. By default stage 03
   **encodes the incidents + the problem catalog itself** (bi-encoder
   `bi_encoder_model`) and takes a **chunked top-K** (never materializes the full
   matrix) — so the shortlist is aligned with the catalog by construction.
   Precomputed `similarity_matrix_path` / embedding paths are opt-in overrides.
3. **Rerank** every `(incident, candidate-problem)` pair with the cross-encoder.
   Pairs are buffered and flushed every `chunk_size` pairs (bounded memory).
4. **Save** raw logits + sigmoid scores to the Volume.
5. **Eval** (optional): re-order candidates by cross-encoder score and report
   the Top-K hit-rate against the existing incident→problem links.

## Outputs
| Target | Shape / grain | Meaning |
|---|---|---|
| Delta `ph03_output_RerankedScores` | one row per `(incident, candidate)` | `number`, `candidate_problem_id`, `rerank_rank`, `rerank_score`, `rerank_score_sigmoid` |
| Volume `reranked_scores.npy` | `(n_incidents, top_k)` | raw cross-encoder logits |
| Volume `reranked_scores_sigmoid.npy` | `(n_incidents, top_k)` | sigmoid → comparable `[0,1]` scores |

Both are column-aligned with the candidate indices, so column `c` of incident
`i` corresponds to candidate problem `candidate_indices[i, c]`.

## ⚠️ Alignment requirement
Candidate index `j` must mean the same problem in the shortlist source **and** in
the problem catalog (`problem_*`). The **default** (encode here) satisfies this by
construction. If you instead point `*_path` at precomputed artifacts, **you** must
guarantee they are row-aligned with the catalog — note that stage 01's
`combined_embeddings.parquet` / `problem_embeddings.parquet` are **per-incident**
(not a per-problem catalog) and therefore are *not* valid here; leave those null.
The pipeline still asserts `max(index) < len(catalog)` as a backstop.

Config: `reranking:` section in the shared root `../config.yml`.
