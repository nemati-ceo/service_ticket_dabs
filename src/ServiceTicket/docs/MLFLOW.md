# MLflow — what every stage logs

One **parent run** (`problem_health_pipeline`) wraps the whole pipeline; each stage opens a
**nested child run**. Run a stage alone and it opens its own top-level run instead. All
logging is **best-effort** — if MLflow is down or unreachable the stages still run and just
skip logging.

Each stage run **wraps that stage's work**, so a crash lands as a FAILED run with a
traceback and the correct duration (not a clean-looking run that logged nothing).

- **Experiment:** `mlflow.experiment` in `config.yml` — currently `/Shared/ProblemHealth`.
  It must be a folder the **run-as service principal** can write. A personal
  `/Users/<someone>/…` folder gives `PERMISSION_DENIED` and silently drops all logging.
- **Every run** is also stamped with runtime tags (git commit/branch, cluster, user,
  test-vs-full) and a full config snapshot artifact.

## Run names
| Stage | Run name |
|---|---|
| pipeline (parent) | `problem_health_pipeline` |
| 00 Input Sync | `ph00_input_sync` |
| 01 Problem Health | `ph01_problem_health` |
| 01b PII Redaction | `ph01b_pii_redaction` |
| 02 LLM Summarization | `ph02_summarization` |
| 03 Cross-encoder Rerank | `ph03_reranking` |
| 04 Gradient Boosting | `ph04_gbm_inference` |
| 05 Clustering | `ph05_clustering` |

---

## 00 — Input Sync
| Kind | Key | Meaning |
|---|---|---|
| param | `tables_synced` | number of refine→consume pairs |
| tag | `sources` | the refine tables mirrored |
| metric | `rows_total` | rows copied across all tables |
| metric | `rows__<table>` | rows per mirrored table |
| metric | `wall_clock_s` | stage duration |

## 01 — Problem Health
| Kind | Key | Meaning |
|---|---|---|
| param | `embed_model`, `backend`, `batch_size`, `limit` | encoder config |
| tag | `input_table`, `output_table` | lineage |
| metric | `incidents_scored`, `problems_scored` | output sizes |
| metric | `step_<label>_s` | per-step duration (load / clean / model / encode / similarity / save) |
| metric | `input_rows` | rows read |
| metric | `dup_key_pct` | duplicate incident keys — a spike means upstream rot |
| metric | `null_short_description_pct`, `null_description_pct` | empty text rate |
| metric | `wall_clock_s` | stage duration |

## 01b — PII Redaction
| Kind | Key | Meaning |
|---|---|---|
| param | `spacy_model`, `model_path` | which model, loaded from the Volume |
| param | `entities` | entity types redacted |
| param | `score_threshold` | **must stay < 0.4** or phone numbers leak |
| tag | `output_tables` | redacted tables written |
| metric | `rows_redacted`, `tables_redacted` | coverage |
| metric | `wall_clock_s` | stage duration |

## 02 — LLM Summarization
| Kind | Key | Meaning |
|---|---|---|
| param | `model`, `input_table`, `drop_deleted`, `limit` | config |
| param | `problem_prompt_fingerprint`, `incident_prompt_fingerprint` | **which prompt version** produced these summaries; a prompt edit is otherwise untraceable |
| tag | `output_incident`, `output_problem` | lineage |
| metric | `llm_calls_total` | rows actually sent to the LLM — the **cost** driver |
| metric | `problems_cache_hit_pct`, `incidents_cache_hit_pct` | hash-skip effectiveness; **0% = cache defeated** |
| metric | `problems_fallback_pct`, `incidents_fallback_pct` | NO_CONTENT rate; a spike = summaries degraded to raw ticket text |
| metric | `problem_summary_len_avg`, `incident_summary_len_avg` | a drop = truncation |
| metric | `problems_total/summarized`, `incidents_total/summarized` | counts |
| metric | `problem_rows_out`, `incident_rows_out` | live-table row counts |
| metric | `topk_accuracy` | optional offline eval (off by default) |
| metric | `wall_clock_s` | stage duration |

## 03 — Cross-encoder Rerank
| Kind | Key | Meaning |
|---|---|---|
| param | `model`, `top_k`, `max_length`, `chunk_size`, `batch_size`, `bi_encoder_model`, `limit` | config |
| tag | `output_table` | lineage |
| metric | `n_incidents`, `n_problems_catalog` | input sizes |
| metric | `pairs_reranked` | incidents × top_k — the work done |
| metric | `output_rows` | rows in the live Delta table |
| metric | `rerank_score_mean/min/max` | score spread; a collapsed range means the reranker stopped discriminating |
| metric | `top_<k>_accuracy` | Top-K retrieval accuracy |
| metric | `<baseline>_top_<k>_delta` | gain vs the frozen PH02/PH05 baselines |
| artifact | `topk_accuracy.json` | per-k table, browsable |
| metric | `wall_clock_s` | stage duration |

## 04 — Gradient Boosting
| Kind | Key | Meaning |
|---|---|---|
| param | `model`, `top_n`, `batch_size` | config |
| tag | `output_table` | lineage |
| metric | `incidents_linked`, `output_rows` | live-table row counts |
| metric | `feature_rows`, `candidates_scored` | per-(incident, candidate) rows scored |
| metric | `positives`, `positive_rate` | label balance — a drop means the gold links thinned out |
| metric | `propensity_mean/min/max` | score spread — a collapsed range means the GBM stopped discriminating |
| metric | `secs_<step>` | per-step duration (build features, load model, score, rank, save) |
| metric | `wall_clock_s` | stage duration |

Production logs NO Top-K numbers: the incident-level match rate needs a gold `problem_id`,
which production incidents do not have, so it is train-mode monitoring only.

TRAIN mode (`mode: train`) logs the fitted model's train/test metrics instead — including
`train_top<k>_match_rate` / `test_top<k>_match_rate`, `holdout_top_<k>_match_rate`, and a
`holdout_topk_match_rate.json` artifact. It never writes the production linking table.

## 05 — Clustering
| Kind | Key | Meaning |
|---|---|---|
| param | UMAP / HDBSCAN params, `embed_model`, `merge_threshold`, `n_rows`, `n_groups`, `group_col`, `min_cluster_rows` | config |
| tag | `output_table`, `overlay_table` | lineage |
| metric | `n_groups`, `groups_clustered`, `groups_too_small` | how the assignment groups split — a jump in `groups_too_small` means the input thinned out |
| metric | `total_clusters`, `total_themes`, `n_merges` | summed across groups; `total_themes` far below `total_clusters` = aggressive merging |
| metric | `n_noise`, `noise_pct`, `silhouette` | cluster quality. Both are **row-weighted** across groups, so one tiny all-noise group cannot swing them |
| metric | `rows_clustered`, `output_rows`, `overlay_rows` | live-table row counts |
| metric | `secs_<step>` | per-step duration (summaries, load, embed, cluster, merge, save) |
| metric | `wall_clock_s` | stage duration |
| artifact | `per_group_stats.json` | per-group rows/status/clusters/noise/silhouette/themes — which group moved a rollup |
| artifact | `clusters_2d.html` / `.png`, `merge_log.json`, `input.sql` | visual + audit trail |

Per-group numbers are an artifact, not metrics: one metric key per assignment group would
grow the experiment every time a new group appears. See `docs/05-clustering.md` for how to
read the output tables.

---

## How to read a run
1. **Did it cost anything?** `ph02.llm_calls_total` — 0 means everything was cached.
2. **Is the cache working?** `ph02.*_cache_hit_pct` — should be high on a re-run over
   unchanged data. 0% means something is invalidating hashes (prompt/model change, or a
   stage deleting another stage's rows).
3. **Is quality drifting?** `ph02.*_fallback_pct` up, `*_summary_len_avg` down, or
   `ph03.rerank_score_*` collapsing to a narrow band.
4. **Is the input rotting?** `ph01.dup_key_pct` / `null_*_pct` climbing.
5. **Did anything silently shrink?** compare `*_rows_out` / `output_rows` against the
   stage-00 `rows_total` for the run.
