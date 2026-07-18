# Stage 01 — Problem Health

**Dir:** `src/ServiceTicket/01_problem_health/`
**File:** `pipeline.py` (orchestration, 6 steps)

## What it does (short)
Reads the cleaned incident snapshot, embeds incident + problem text, scores how close each incident is to its problem (cosine similarity), then rolls that up into a per-problem health score. Full run every time — no diffing.

## Flow
```
run_problem_health(spark, cfg)
        |
   MLflow stage_run("ph01_problem_health")  <── wraps ALL steps (crash = FAILED run)
        |
 [1/6] LOAD    table -> _load_input_table (Arrow-safe projection)
        |       servicenow -> snow.fetch_incidents
        |       (limit -> TEST MODE head(n))
        |
 [2/6] CLEAN   clean_text_step -> combined_cleaned_desc, combined_prob_desc
        |
 [3/6] MODEL   emb.load_or_save_model (all-MiniLM-L6-v2, onnx, from Volume)
        |
 [4/6] ENCODE  incident emb = encode(combined_cleaned_desc)
        |       problem emb  = encode(unique combined_prob_desc)  <- dedup by problem_id
        |       map problem emb back to each incident
        |       (opt) save embeddings -> parquet (_save_parquet)
        |
 [5/6] SCORE   sim.add_similarity -> cosine per incident -> save_incident_scores (Delta)
        |
 [6/6] HEALTH  sim.aggregate_problem_health -> per-problem -> save_delta
        |       (opt) save -> parquet (_save_parquet)
        |
   log metrics (incidents_scored, problems_scored, wall_clock, step timings, data-quality)
        |
   return df_incidentscore, problem_health
```

## Helpers
- `_mlflow_utils()` — loads root `mlflow_utils.py` by path (best-effort logging).
- `_data_quality(df, key)` — row count, dup-key %, null-text % -> MLflow. Never raises. Catches silent upstream rot.
- `_load_input_table` — ServiceNow schema is messy (dotted col names, VOID cols, Databricks internal cols); Arrow `toPandas()` chokes, so it projects only real cols, aliases dots out, disables Arrow, restores names.
- `_save_parquet(df, base_path, filename, label)` — **new**: best-effort parquet dump, never raises. Dedupes the two copy/pasted save blocks.

## Best-practice changes applied
1. **MLflow run now wraps ALL work** (was opened at the very end). Before: a crash in steps 1–6 logged nothing to MLflow and the run duration didn't cover the compute. Now a failing step lands as a FAILED run with its traceback + correct duration. Matches the stage-05 pattern.
2. **Deduped save-to-volume.** Two near-identical `try/except os.makedirs + to_parquet` blocks folded into `_save_parquet(...)`.

Verified: `python3 -m py_compile` OK. Values (`df_incidentscore`, `problem_health`, `total`) assigned inside the `with` remain in scope for the final summary print.

## Cleanup verdict (pipeline.py)
Clean after the two changes.

---

## cleaning.py — step-2 text cleaner

### What it does (short)
Cleans incident + problem text and builds the 6 derived columns the encoder needs. Dispatches to a distributed Spark engine or single-thread pandas by `cleaning.engine`; Spark failures fall back to pandas.

### Flow
```
clean_text_step(spark, df, cfg)
        |
   engine = cfg.cleaning.engine
        |
   "spark" ─▶ _clean_with_spark ─▶ cleaning_spark.apply_cleaning_spark (pandas_udf)
        |         (on failure: WARNING, fall back to pandas)
   "pandas" ─▶ apply_cleaning (driver)
        |
   apply_cleaning adds columns:
     cleaned_short_description   = clean(short_description)
     cleaned_description         = clean(description)
     combined_cleaned_desc       = short + " " + desc      <- incident text for encoder
     cleaned_prob_short_desc     = clean(problem_id.short_description)
     cleaned_problem_desc        = clean(problem_id.description)
     combined_prob_desc          = prob_short + " " + prob_desc  <- problem text for encoder
```

### Best-practice changes applied
1. **Fixed truncated docstring** on `clean_text_step` — sentence was cut mid-word (`... "spark" (distributed) or"""`). Now states both engines + fallback.
2. **Fixed stale step labels** `[3/8]` -> `[2/6]` (7 prints). Pipeline calls cleaning at step `[2/6]`; the `/8` was leftover from an old 8-step pipeline and mismatched the logs.

Verified: `py_compile` OK. No other `/8]` labels remain in stage 01.

### Cleanup verdict
Clean after the two fixes. Cleaning logic itself unchanged.

---

## cleaning_spark.py — distributed (Spark) cleaner

### What it does (short)
Spark-native twin of `cleaning.apply_cleaning`. Same 6 columns, built with `pandas_udf` so cleaning runs on executors instead of the driver. Called by `cleaning._clean_with_spark` when `cleaning.engine = "spark"`.

### Column parity
Exact match with the pandas `apply_cleaning`: same 6 cols, same concat order. No behavioral drift between engines.

### Cleanup verdict — clean, 1 DRY note (deferred)
The 3 helpers `_clean_inc_short / _clean_inc_desc / _clean_prob` (lines 14-16) are byte-identical copies of the same trio in `cleaning.py` (lines 9-11).
- **Decision:** leave as-is for now. Dedup by moving the trio into `preprocessing.py` (the shared module both files already import) when `preprocessing.py` is reviewed. Not touched half-way.
- No code removed. No other issues.

---

## Remaining stage-01 files to review
`preprocessing.py` (do the helper dedup here), `benchmark_cleaning.py`, `embeddings.py`, `servicenow_source.py`, `similarity.py`, `storage.py`, `timing.py`.
