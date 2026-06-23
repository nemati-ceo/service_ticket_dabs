# 04 — Gradient Boosting Inference

Stage 4 (final) of the pipeline. Combines the signals from the earlier stages —
**cosine similarity** (stage 01 bi-encoder), **cross-encoder rerank score**
(stage 03), and a **business-service match** flag — into a pre-trained
`GradientBoostingClassifier` (`PH04_gradient_boosting_model.pkl`) that produces a
final **propensity** per `(incident, candidate-problem)` pair, then writes the
Top-N linked problems per incident to Unity Catalog.

## Modules
| File | Role |
|---|---|
| `run.py` | shared entry point (`run.stage04()`) — loads root `config.yml` |
| `pipeline.py` | orchestrator: load → features → score → rank/eval → save |
| `features.py` | id-based join → per-pair feature matrix (`cosine_sim`, `reranker_score`, `bs_match`) |
| `inference.py` | load the GBM, batched `predict_proba` → `gbm_propensity` |
| `evaluate.py` | incident-level Top-K accuracy |
| `linking.py` | Top-N wide linking table (`top_<r>_pid` / `top_<r>_problem_description`) |

## Inputs
- **reranked scores** — stage-03 Delta table `ph03_output_RerankedScores`
  (`number`, `candidate_problem_id`, `cosine_sim`, `rerank_score`). This is the
  producer; stage 04 joins to it **by id** (no positional alignment).
- **incident frame** — `number`, gold `problem_id` (+ `business_service`).
- **problem catalog** — `problem_id` (+ `business_service` + description).
- **model** — `PH04_gradient_boosting_model.pkl` (joblib) on a Volume.

## Output (Unity Catalog — never CSV)
| Target | Content |
|---|---|
| Delta table `ph04_output_Incident_Problem_Linking_Top10` | one row per incident + `top_1..N` problem ids and descriptions |
| Volume `Incident_Problem_Linking_Top10.parquet` | same frame, parquet only |

## Notes
- **Scope** is inherited from the stage-03 reranked table (no separate limit).
- **Feature order** (`cosine_sim`, `reranker_score`, `bs_match`) must match how the
  model was trained — defined once in `features.FEATURE_COLS`.
- **No positional alignment:** everything joins by `number` / `problem_id`, so
  row order across sources no longer matters.

Config: `gbm_inference:` section in the shared root `../config.yml`.
