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

## Output — one live Delta table (no parquet)
| Target | Content |
|---|---|
| `ph04_output_Incident_Problem_Linking_Top10` | one row per incident + `top_1..N` problem ids and descriptions |

## Guarantees this stage enforces
| Check | Fails how |
|---|---|
| a missing gold `problem_id` stays NULL (never the string `"nan"`) | otherwise train's `.notna()` filter keeps unlabeled rows and fits on them as negatives |
| Top-K denominator counts only incidents that **have** a gold `problem_id` | otherwise unlinked incidents divide every accuracy down |
| input comes from a live Delta table / SQL — **no parquet fallback** | a swallowed table error would silently score yesterday's file |
| model file missing → `FileNotFoundError` naming `mode: train` | joblib's own error says nothing about what to do |
| model `n_features_in_` vs `FEATURE_COLS` | a model fitted on a different column set scores garbage silently |
| `bs_match` dead (feature always 0) | loud warning — always a config/table mismatch upstream |

## Notes
- **Scope** is inherited from the stage-03 reranked table (no separate limit).
- **Model load:** one `joblib.load` of the Volume `.pkl` per run. Nothing is downloaded,
  and no batch re-reads the model.
- **Timing:** per-step laps (`build features`, `load model`, `score`, `rank`,
  `build linking table`, `save`) print a breakdown and land in MLflow as `secs_*`.
- **Feature order** (`cosine_sim`, `reranker_score`, `bs_match`) must match how the
  model was trained — defined once in `features.FEATURE_COLS`.
- **No positional alignment:** everything joins by `number` / `problem_id`, so
  row order across sources no longer matters.

## TRAIN mode (`mode: train`) — weak-link filter
`gbm_train.min_semantic_similarity` drops incidents whose cosine to their **gold** problem
is below the threshold before fitting: a weak incident↔problem link is a bad label, and
training on it teaches the model to reproduce bad links. `null` = no filter.

**This must never run in production** — there the gold problem is what we are predicting,
so filtering on it would leak the answer and skip exactly the incidents that most need
linking. Enforced structurally:

- `similarity_col` is carried through `build_feature_matrix` as a **passthrough** column,
  deliberately **not** in `FEATURE_COLS`, so `inference.score` (`feature_df[FEATURE_COLS]`)
  can never read it.
- The filter lives in `train.filter_weak_links`, reached only from `run_gbm_train`, which
  only runs under `mode: train`.
- Pinned by `tests/test_train_filter.py` (drops below / keeps `>=`, `None` is a no-op,
  missing column skips with a warning, too-high threshold raises, and `similarity_col` is
  never a model feature).

Production drops **no** rows: the dedups in `features.py`/`linking.py` are join-key and
output-grain dedups, and every distinct incident survives to the linking table.

Config: `gbm_inference:` and `gbm_train:` in the shared root `../config.yml`.
