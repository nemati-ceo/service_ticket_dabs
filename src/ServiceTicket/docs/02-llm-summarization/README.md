# 02 — LLM Summarization

Normalizes incident and problem text with an LLM (`ai_query`, Databricks Claude) so
downstream stages match on clean, consistent language. Reads the **redacted** stage-01b
output — this is the only stage that sends text off-cluster, so its input must be scrubbed.

## Modules
| File | Role |
|---|---|
| `pipeline.py` | orchestrator: summarize problems, then incidents, optional eval, mlflow |
| `summarize.py` | prompts + `ai_query` + Spark-native incremental MERGE |
| `evaluate.py` | optional sampled Top-K retrieval metric (off by default) |

## Inputs
- `summarization.input_table` — `ph01b_output_Redacted` (redacted; never a raw table)
- `summarization.problem_source_sql` — UNION of linked problems + zero-incident problems.
  Without the union a zero-incident problem is never summarized and the GBM can never
  propose it as a link. The zero-incident side concatenates `short_description` +
  `description` (both redacted by stage 01b): the short description alone is a title —
  too thin to summarize or to match on. Changing this text changes the cache key, so the
  affected problems are re-summarized once (re-billed) on the next run.

## Outputs — live Delta tables only (no parquet)
| Table | Grain | Columns |
|---|---|---|
| `ph02_output_ProblemSummaries` | problem | `problem_id`, `problem_summary`, `summary_input_hash`, `model_name`, `summarized_at` |
| `ph02_output_IncidentSummaries` | incident | `number`, `incident_summary`, `summary_input_hash`, `model_name`, `summarized_at` |

Stage 05's gap-fill writes its **own** table (`ph05_output_UnlinkedSummaries`), not this
one. Sharing it let stage 02's `drop_deleted` wipe stage 05's rows every run, re-billing
the LLM for the whole unlinked population.

## No duplicates, no re-billing
- **Cache key** = `md5(text + '||' + prompt + '||' + model)`. Text alone is not enough: a
  prompt edit or model swap must re-summarize the rows it affects, or stale summaries
  live forever.
- A `LEFT ANTI JOIN` on `(key, hash)` selects **only new/changed rows** — unchanged rows
  are never re-sent to the LLM.
- `ROW_NUMBER() … _rn = 1` enforces **one row per key** before the LLM. The source key is
  `(number, problem_id)`, so an incident on N problems would otherwise be billed N times
  and break the MERGE with `DELTA_MULTIPLE_SOURCE_ROW_MATCHING_TARGET_ROW`.
- Results are **MERGE-upserted** by key; `drop_deleted` removes keys gone from the source.
- LLM output is **materialized to a staging table**, not left as a view — a lazy view over
  `ai_query()` re-executes on every action and would bill twice.

## MLflow (`ph02_summarization`)
The run wraps the work, so a crash mid-summarization lands as a FAILED run.

- **Params:** `model`, `input_table`, `drop_deleted`, `limit`, `problem/incident_prompt_fingerprint`
- **Metrics:** `llm_calls_total` (cost), `problems/incidents_cache_hit_pct` (0% = cache
  defeated), `problems/incidents_fallback_pct` (spike = summaries degraded to raw text),
  `problem/incident_summary_len_avg` (drop = truncation), `*_rows_out`, `topk_accuracy`,
  `wall_clock_s`, and the token counts below

`prompt_fingerprint` identifies **which prompt version** produced a run's summaries — a
prompt edit silently changes output and is otherwise untraceable between runs.

## Token accounting

`ai_query()` returns no usage struct and Spark SQL has no tokenizer, so the run estimates
spend from character counts at **~4 chars/token** (`summarize._CHARS_PER_TOKEN`). Treat it
as a **cost signal** — "is this run 10x yesterday?" — not a billing figure. The
authoritative per-request numbers live in `system.serving.endpoint_usage`.

| Metric | Counts |
|---|---|
| `problem_input_tokens` / `incident_input_tokens` | prompt prefix (once per row sent) + source text |
| `problem_output_tokens` / `incident_output_tokens` | generated summaries |
| `input_tokens_total` / `output_tokens_total` | the two entities added together |
| `tokens_total` | input + output — the single number to watch |

Two things the estimate does on purpose:

- **Only rows actually sent this run are counted.** A fully cached run reports `0`, which
  is correct: the metric tracks spend, not corpus size. A jump means the cache was
  defeated — usually a prompt or model edit — and `*_cache_hit_pct` confirms it.
- **Fallback rows are excluded from output tokens.** A `NO_CONTENT` row keeps its original
  text, which the LLM did not generate; counting it would inflate output on exactly the
  runs where the model produced the least.

Stage 05's gap-fill calls the same summarizer and logs its own spend as
`gapfill_input_tokens` / `gapfill_output_tokens` / `gapfill_tokens_total` on the
`ph05_clustering` run. **Total LLM cost of a pipeline run = stage 02 + that.**

## Test runs (`run.limit`)
When `run.limit` is set, both source queries are capped (wrapped in a subquery so the
UNION/GROUP BY still applies) and **`drop_deleted` is forced off**. Without the cap a
"limited" run still sent the entire zero-incident problem catalog to the LLM — those rows
come from ProblemsZero, which no upstream stage limits. Without forcing `drop_deleted`
off, a capped source would delete every summary beyond the cap from the live table.

Config: `summarization:` in the shared root `../config.yml`.
