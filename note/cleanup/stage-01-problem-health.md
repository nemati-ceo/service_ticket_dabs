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

## Remaining stage-01 files — full review

| File | Verdict |
|---|---|
| `preprocessing.py` | Real clean funcs (transcribed). Now also home of the shared composed trio. |
| `embeddings.py` | Clean. Loads model from Volume if present -> **no redownload** (answers note #3). |
| `similarity.py` | Clean. `add_similarity` writes `semantic_similarity` (the train-filter column). |
| `storage.py` | Clean after consolidating the save helper. |
| `timing.py` | Clean — no issues. |
| `servicenow_source.py` | Clean (`dbutils` global is caught; `apikey` header is the load-bearing fix). |
| `benchmark_cleaning.py` | Dev utility (not in prod). Truncated docstring fixed. |

### Best-practice changes applied
1. **DRY dedup (the deferred item).** The composed trio `clean_inc_short / clean_inc_desc / clean_prob` was byte-duplicated in `cleaning.py` and `cleaning_spark.py`. Moved to `preprocessing.py` (single definition); both engines now `import ... as _clean_*` from there and can't drift. Confirmed: `cleaning._clean_inc_short.__module__ == "preprocessing"`.
2. **Consolidated the save-to-volume helper.** The best-effort `os.makedirs + to_parquet + try/except` pattern existed in **3 places** (pipeline.py ×2 region, storage.py). Now one `storage.save_parquet(df, base_path, filename, label)`; `pipeline.py` imports it, `save_incident_scores` uses it. Test updated to target `storage.save_parquet`.
3. **Fixed truncated docstring** in `benchmark_cleaning.py` (line 1 was cut mid-sentence).

### Flagged, NOT changed (need a decision)
- **`preprocessing.clean_close_notes` is DEAD** — 0 references anywhere. Candidate for removal, but relates to `close_notes` which stage 05 redacts/uses elsewhere; left in pending a call on whether it will be wired.
- **`preprocessing.py` calls `nltk.download` at import time** — same firewall risk as the spaCy model in stage 01b: in the redzone a runtime download can hang instead of failing fast. Should be staged to a Volume / bundled like `en_core_web_lg`. Not fixed here (behavior-preserving pass).
- **Email regex `[A-Z|a-z]` (line 56)** — the `|` is a literal pipe inside the char class (transcription quirk), not an alternation. Harmless in practice; left as-is to avoid changing transcribed behavior.

### Verification
`py_compile` all 6 files OK. Full suite: **94 passed, 8 skipped**.

## preprocessing.py — refactor

### What changed (behavior-preserving)
Transcribed cleaning logic, cleaned up without moving any output:
1. **Hoisted all literal remove-lists to module constants** (`DESC_REMOVE_TEXT`, `SHORT_PREFIX_REMOVE`, `GENERAL_PROBLEM_REMOVE`, `URL_GLOBS`, …) — declared once instead of rebuilt inside every call.
2. **Compiled the regexes once** at module load (`EMAIL_RE`, `FIELDID_RE`, `HOID_RE`, `TECHID_RE`, `GA_RE`, `SHORT_ALNUM_RE`, `PUNCT_RE`).
3. **Unified the repeated loop bodies** into three helpers: `_replace_all` (str.replace each), `_sub_prefixed` (re.sub of escaped phrase + a tail like `/\S*`, `.*$`, `\s+\n.*`), `_strip_leading` (sequential prefix strip).
4. **nltk is now LAZY** — `_ensure_nltk()` runs on the first `clean_text` call, not at import. Importing the module no longer hits the network — the redzone firewall risk (download hangs) is gone. Proven: module imports with nltk absent.
5. **Removed dead `clean_close_notes`** (0 references anywhere).
6. **Fixed the email regex** `[A-Z|a-z]` -> `[A-Za-z]` (the `|` was a literal pipe in the char class, a transcription bug; no behavior change for real emails).

### Safety: golden characterization tests
`tests/test_preprocessing.py` pins the EXACT output of every live function
(`clean_description_text`, `clean_shortDescription_text`, `clean_text`,
`removeURL`, `removeGeneralProblemText`, composed trio) on representative inputs.
The goldens were captured from the ORIGINAL code, then the refactor was made to
match them byte-for-byte. nltk is stubbed deterministically so `clean_text` is
reproducible off-cluster.

### Verification
`py_compile` OK. Goldens: 10 passed against BOTH old and new code. Full suite: **104 passed, 8 skipped**.

## Stage 01 — DONE. Remaining stages for later: 02, 03, 05.
