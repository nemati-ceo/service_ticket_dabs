# Tests

Fast, offline unit tests for the **dependency-free** parts of the pipeline.

## Run

```bash
cd problem_health
pip install -r requirements/requirements-dev.txt
pytest tests/ -v
```

## What's covered

| File | Module under test | What it checks |
|------|-------------------|----------------|
| `test_mlflow_utils.py` | `mlflow_utils.py` | enabled toggle, `topk_metrics` flattening, key/artifact namespacing, no-op when MLflow is disabled or missing, best-effort error swallowing, and the core contract: the **full pipeline logs to exactly ONE run** while a stage run alone opens its own run. Uses a fake `mlflow` that records calls. |
| `test_evaluate.py` | `03_.../evaluate.py`, `04_.../evaluate.py` | Hand-computed Top-K accuracy (the metrics now logged to MLflow): score-based ranking, `k > n_candidates` clamping, empty-input safety. |

## What's *not* covered, and why

Stages 01–05's `run_*` orchestration needs Spark, Unity Catalog tables, an
embedding model download, and the Databricks Claude `ai_query` endpoint — none of
which exist off-cluster. Those paths are validated by running the pipeline on
Databricks, not here. The tests deliberately target only logic that runs anywhere
with `python3` + `numpy`/`pandas`.
