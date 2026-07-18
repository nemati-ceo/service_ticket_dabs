# Stage 01b — PII Redaction

**Dir:** `src/ServiceTicket/01b_pii_redaction/` — `pipeline.py`, `redact.py`, `stage_model.py`

## What it does (short)
Sits between stage 01 and stage 02. Stage 02 is the only stage that sends text off-cluster, so every text column it can reach is scrubbed here first: each configured table's text columns have their PII spans replaced with `<ENTITY>` placeholders (irreversible), and the result is written to a redacted mirror table. Downstream stages read only the redacted tables.

## Flow
```
run_pii_redaction(spark, cfg)
        |
   pii_redaction.enabled ? no -> skip
        |
   _tables(pc)  -> list of {input_table, output_table, text_columns}
        |
   addPyFile(redact.py)      # ship module to executors
   udf = _redact_udf(pc)     # engines built ONCE per executor, cached on _ENGINES
        |
   MLflow stage_run("ph01b_pii_redaction")   <── wraps the redaction work (PII boundary)
        |
   for each table:
     _redact_table -> spaCy+Presidio scrub each text col -> overwrite redacted mirror
        |
   log rows_redacted / tables_redacted / wall_clock
```

## Model loading (no redownload)
`redact.build_engines` loads spaCy by **Volume PATH**, never by name — and **raises FileNotFoundError** if the path is missing (the redzone firewalls spacy.io, so a runtime `spacy.load(name)` would hang). `stage_model.py` is the one-time offline stager (`SKIPS if already staged`, `--force` to re-stage). So stage 01b never downloads.

## Best-practice changes applied
1. **MLflow run now wraps the redaction work** (was opened only at the end to log). This stage is the PII boundary — a crash mid-redaction must land as a FAILED run with its traceback + correct duration. Matches stages 01/05.
2. **`redact_text` default `score_threshold` 0.5 -> 0.35.** Production always passes 0.35 from config, but the bare default was a footgun: Presidio scores PHONE_NUMBER at exactly 0.4, so a 0.5 default silently leaks every phone number. Now the default itself is safe.

## try/catch verdict
- `build_engines` **raises** on a missing model — correct: you must NOT redact-with-nothing and let raw PII flow downstream. Fail hard.
- `redact_text` guards None/blank, otherwise lets analyzer errors propagate (a redaction failure must stop the stage, not silently pass text through).
- `stage_model.stage` guards `spacy.load` (OSError -> download). Correct.
- No best-effort swallowing added anywhere text could leak.

## Tests
- `test_pii_redaction.py` (existing) — config guards (score-threshold trap, redacted-table wiring). Engine tests skip without presidio/spaCy.
- `test_pii_pipeline.py` (new) — `_tables` parsing (list / single / error), `redact_text` None+blank short-circuits, and a guard that the default threshold stays < 0.4.

## Verification
`py_compile` OK. Full suite: **124 passed, 8 skipped**.

---

## Run gate (temporary)
`run.py:_run_all_stages` has a **TEMP VERIFY GATE**: it stops after stage 01b and returns, so a Databricks run executes only 00 → 01 → 01b (production mode). Stages 02-05 are reviewed one at a time and re-enabled by deleting the gate block. Each stage's output is printed (row counts + summary) for eyeball verification in Databricks before moving on.
