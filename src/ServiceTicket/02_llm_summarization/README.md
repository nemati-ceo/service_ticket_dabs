# 02 — LLM Summarization

Stage 2 of the pipeline. Reads stage-01 output
(`ph01_output_IncidentScore_SemanticSimilarity`), normalizes incident & problem
text with an LLM (`ai_query`, Databricks Claude), and upserts the summaries.

## Modules
| File | Role |
|---|---|
| `run.py` | entry point (`from run import main`) — loads shared root `config.yml` |
| `pipeline.py` | orchestrator: summarize problems + incidents, optional eval |
| `summarize.py` | `ai_query` prompts + Spark-native incremental MERGE |
| `evaluate.py` | optional, sampled Top-K retrieval metric (off by default) |

## No-duplicate summarization
For each entity a `summary_input_hash = md5(text)` is computed. A `LEFT ANTI JOIN`
against the output table on `(key, hash)` selects **only new/changed rows**, so
unchanged rows are never re-sent to the LLM (no re-billing), and results are
**MERGE-upserted** by key (no duplicate rows). Deleted keys are removed.

## Outputs (`redzone_consume.model_governance`)
| Table | Grain | Columns |
|---|---|---|
| `ph02_output_ProblemSummaries` | problem | `problem_id`, `problem_summary`, `summary_input_hash`, `model_name`, `summarized_at` |
| `ph02_output_IncidentSummaries` | incident | `number`, `incident_summary`, `summary_input_hash`, `model_name`, `summarized_at` |

Config: `summarization:` section in the shared root `../config.yml`.
