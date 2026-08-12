# ProblemHealth

Incident → problem intelligence pipeline for ServiceNow tickets. Six stages turn
raw incidents into semantic-similarity scores, LLM summaries, reranked problem
matches, a gradient-boosted linking table, and clustered themes — starting from a
full-snapshot input sync, orchestrated from a single entry point and tracked in MLflow.

## Stages at a glance

| # | Stage | Does | Writes |
|---|-------|------|--------|
| 00 | Input Sync | Full-snapshot MERGE of refine into a consume mirror (INSERT/UPDATE/DELETE-by-absence) | `input_sync.target` (consume mirror) |
| 01 | Problem Health | Embeds incidents/problems, scores cosine similarity (incremental) | `ph01_output_IncidentScore_SemanticSimilarity`, `ph01_output_ProblemHealth` |
| 02 | LLM Summarization | Summarizes incidents & problems (hash-MERGE reuse) | `ph02_output_IncidentSummaries`, `ph02_output_ProblemSummaries` |
| 03 | Cross-Encoder Rerank | Reranks top-K candidate problems per incident | `ph03_output_RerankedScores` |
| 04 | Gradient Boost Inference | Scores cosine+reranker features, emits top-10 links | `ph04_output_Incident_Problem_Linking_Top10` |
| 05 | Clustering | Embeds summaries, UMAP+HDBSCAN, merges near-duplicate clusters into themes | `ph05_output_ClusterThemes`, `ph05_output_ThemeOverlay` |

📖 **See [`Pipeline.md`](Pipeline.md) for the full stage-by-stage architecture
(ASCII data-flow diagrams, gating rules, and table lineage).**

## Running

```python
import run

run.main()        # full pipeline: stage 00 → 01 → 02 → 03 → 04 → 05 (one MLflow run)

run.stage00()     # or run a single stage
run.stage05()
```

One entry point (`run.py`), one shared `config.yml`. Each stage reads the previous
stage's Delta output; stage 00 runs first and raises on failure (stops everything),
then later stages are gated on the earlier ones producing output
(02 needs 01, 03 needs 02, 04 needs 03; 05 depends on 01 + 02). All stages log into
a single MLflow run with stage-namespaced keys (`ph01_*`, `ph03_top_5_accuracy`, …).

## Configuration

- [`../config.yml`](../config.yml) — all wiring: input source, table names, model names, secrets scope.
- [`PARAMETERS.md`](PARAMETERS.md) — every tunable knob per stage, with defaults and what each one affects.
- [`config-reference.md`](config-reference.md) — what every config block means, and the traps.

## Stage docs
| Stage | Doc |
|---|---|
| 00 Input Sync | [`00-input-sync/`](00-input-sync/README.md) |
| 01 Problem Health | [`01-problem-health/`](01-problem-health/README.md) |
| 01b PII Redaction | [`01b-pii-redaction/`](01b-pii-redaction/README.md) |
| 02 LLM Summarization | [`02-llm-summarization/`](02-llm-summarization/README.md) |
| 03 Cross-encoder Rerank | [`03-cross-encoder-rerank/`](03-cross-encoder-rerank/README.md) |
| 04 Gradient Boosting | [`04-gradient-boost-inference/`](04-gradient-boost-inference/README.md) |
| 05 Clustering | [`05-clustering/`](05-clustering/README.md) |

Each stage is a folder: `README.md` is the stage doc, `diagram.md` is its flow diagram
(added per stage — `01-problem-health/` has one so far).

Cross-cutting: [`Pipeline.md`](Pipeline.md) (architecture + the whole-pipeline diagram),
[`MLFLOW.md`](MLFLOW.md) (every metric each stage logs), [`tests.md`](tests.md).

## Layout

```
ServiceTicket/
├── run.py                    # single entry point for all stages
├── config.yml                # shared config (tables, models, knobs)
├── mlflow_utils.py           # shared MLflow logging helpers
├── docs/                     # one folder per stage + cross-cutting markdown
├── 00_input_sync/            # refine → consume full-snapshot overwrite (sync.py)
├── 01_problem_health/        # each stage: pipeline.py (orchestration) + helpers
├── 01b_pii_redaction/
├── 02_llm_summarization/
├── 03_cross_encoder_rerank/
├── 04_gradient_boost_inference/
├── 05_clustering/
└── tests/                    # off-cluster unit tests (dependency-free logic)
```

## Tests

Stages themselves need Spark/Databricks/an LLM endpoint, so the tests cover the
dependency-free logic that runs off-cluster (metrics, merge/cluster helpers,
MLflow utils):

```bash
python3 -m pytest
```
