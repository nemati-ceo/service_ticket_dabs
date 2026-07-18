# 03 — Cross-encoder Reranking

Takes a cheap bi-encoder shortlist (top-K candidate problems per incident) and
**re-scores only those pairs** with a heavier cross-encoder
(`cross-encoder/ms-marco-MiniLM-L-6-v2`) that reads the incident and problem text
*jointly* — far more accurate than cosine, but too slow to run on every pair, which is
why it only runs on the shortlist.

## Modules
| File | Role |
|---|---|
| `pipeline.py` | orchestrator: load → shortlist → rerank → save → eval, mlflow |
| `rerank.py` | cross-encoder load, top-K candidate selection, chunked predict, sigmoid |
| `evaluate.py` | optional Top-K hit-rate vs prior-stage baselines |

## Flow
1. **Load** summarized incidents (`incident_summary` + gold `problem_id`) and the
   problem-summary catalog (the candidate pool) — both from stage-02 Delta tables.
2. **Shortlist** the top-K candidates per incident: stage 03 encodes the incidents and the
   catalog with the bi-encoder and takes a **chunked top-K** (never materializes the full
   similarity matrix), so the shortlist is aligned with the catalog by construction.
3. **Rerank** every `(incident, candidate)` pair with the cross-encoder. Pairs are buffered
   and flushed every `chunk_size` pairs, so memory stays bounded.
4. **Save** the long table (below).
5. **Eval** (optional): re-order candidates by cross-encoder score and report the Top-K
   hit-rate against the existing incident→problem links, plus the delta vs frozen baselines.

## Models — loaded from the Volume, never re-downloaded
Both the cross-encoder (`model_volume_path`) and the bi-encoder (`bi_encoder_volume_path`)
load from the Volume when present, and are downloaded + cached there only on first use.

## Output — one live Delta table (no .npy / parquet)
`ph03_output_RerankedScores`, one row per `(incident, candidate)`:

| Column | Meaning |
|---|---|
| `number` | incident key (**must exist** — see below) |
| `candidate_problem_id` | candidate from the catalog |
| `rerank_rank` | 1..top_k, shortlist order |
| `cosine_sim` | bi-encoder score (stage-04 feature) |
| `rerank_score` | raw cross-encoder logit (stage-04 feature) |
| `rerank_score_sigmoid` | logit → comparable `[0,1]` |

Stage 04 joins this **by id**, so the row order does not matter — but the key does:
if the incident frame has no `number` column the stage **raises**. Fabricating `0,1,2…`
there would produce a table that joins to nothing downstream, with no error anywhere.

## Alignment
Candidate index `j` must mean the same problem in the shortlist and in the catalog. Since
the shortlist is built from the catalog in the same call, this holds by construction. The
pipeline still asserts `max(index) < len(catalog)` and `rows == n_incidents` as backstops.

## MLflow (`ph03_reranking`)
Wraps the work, so a crash mid-rerank lands as a FAILED run. Params: model, `top_k`,
`max_length`, `chunk_size`, `batch_size`, `bi_encoder_model`, `limit`. Metrics:
`n_incidents`, `n_problems_catalog`, `pairs_reranked`, `output_rows`,
`rerank_score_mean/min/max` (a collapsed range means the reranker stopped discriminating),
`top_<k>_accuracy`, baseline deltas, `wall_clock_s`, plus a `topk_accuracy.json` artifact.
Full key list in `../MLFLOW.md`.

## Note
`eval.baselines` carry k = 1, 5, 7, 10 but `eval.k_values` computes only 5 and 10, so
there is no head-to-head at Top-1/Top-7. Align them if you want the full comparison.

Config: `reranking:` in the shared root `../config.yml`.
