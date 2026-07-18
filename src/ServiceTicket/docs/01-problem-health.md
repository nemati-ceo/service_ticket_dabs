# 01 — Problem Health

Cleans the incident/problem text, embeds it, scores how close each incident is to its
linked problem (cosine similarity), then rolls that up into a per-problem health score.
Full run every time — the snapshot is complete, so there is nothing to diff.

## Modules
| File | Role |
|---|---|
| `pipeline.py` | orchestrator, 6 steps, mlflow |
| `cleaning.py` | dispatches cleaning: `spark` (distributed) or `pandas` (driver) |
| `cleaning_spark.py` | Spark-native twin of the pandas cleaner (`pandas_udf`) |
| `preprocessing.py` | the actual text cleaners + the composed trio both engines share |
| `embeddings.py` | load the sentence-transformer (Volume → download → registry) + encode |
| `similarity.py` | row-aligned cosine + per-problem aggregation |
| `storage.py` | write helpers for the live Delta output tables |
| `timing.py` | per-step + total timing (feeds the mlflow step metrics) |
| `servicenow_source.py` | optional live REST input instead of the table |
| `benchmark_cleaning.py` | dev-only: pandas vs Spark cleaning speed + output parity |

## Flow
```
[1/6] LOAD    table (Arrow-safe projection) or ServiceNow REST
[2/6] CLEAN   -> combined_cleaned_desc, combined_prob_desc
[3/6] MODEL   all-MiniLM-L6-v2 (onnx) — from Volume, no redownload
[4/6] ENCODE  incidents; problems encoded once per unique problem_id, mapped back
[5/6] SCORE   cosine per incident      -> ph01_output_IncidentScore_SemanticSimilarity
[6/6] HEALTH  mean similarity/problem  -> ph01_output_ProblemHealth
```

## Outputs — live Delta tables only (no parquet)
| Table | Grain |
|---|---|
| `ph01_output_IncidentScore_SemanticSimilarity` | incident (adds `semantic_similarity`) |
| `ph01_output_ProblemHealth` | problem (`ProblemHealth_Score`, `Last_Incident_Date`) |

`semantic_similarity` is the column stage 04's **train-only** weak-link filter reads.

## Invariants
- **No incident rows are dropped.** The dedup at step 4 builds a separate `uniq` frame to
  encode each problem once; the output keeps every incident.
- **Cleaning must fail hard.** The pure text transforms carry no try/except — a swallowed
  error there would silently corrupt output. Only best-effort I/O (mlflow, model registry)
  is guarded.
- **The model loads from the Volume**; it is only downloaded when the Volume is empty.
- The Volume load path does **not** touch mlflow, so a cached model still loads when
  mlflow is down.

## MLflow (`ph01_problem_health`)
Wraps all 6 steps, so a crash in any step lands as a FAILED run with the right duration.
Params: model/backend/batch_size/limit. Metrics: `incidents_scored`, `problems_scored`,
`wall_clock_s`, per-step timings, and input data-quality (`input_rows`, `dup_key_pct`,
`null_<col>_pct`) which surfaces silent upstream rot.

Config: `tables:`, `model:`, `cleaning:`, `keys:` in the shared root `../config.yml`.
