# Stage 04 — TRAIN-ONLY weak-link filter (min_semantic_similarity)

**Files:** `04_gradient_boost_inference/{train,features,pipeline}.py`, `tests/test_train_filter.py`
**Config keys:** `gbm_train.min_semantic_similarity` (0.35), `gbm_train.similarity_col` ("semantic_similarity")

## Problem found
The two config keys were **dead** — zero code references. The config comment claimed the filter was "TRAIN ONLY ... Enforced by a test," but neither the filter nor the test existed. `train.py` only deduped `(number, candidate_pid)` and dropped unlabeled rows; it never filtered on semantic similarity. So `min_semantic_similarity: 0.35` did nothing at runtime.

## What we want
Drop training incidents whose cosine to their GOLD problem is below the threshold (a weak link = a bad label). **Train only** — production must never drop a row on this, or it would delete real scored incidents.

## Why it's safe in production
- The GBM scores on `features.FEATURE_COLS = ["cosine_sim", "reranker_score", "bs_match"]` only.
- `semantic_similarity` is carried as a **passthrough** column, deliberately NOT in `FEATURE_COLS`.
- `inference.score` does `X = feature_df[FEATURE_COLS]` — it never reads the passthrough.
- The filter lives in `run_gbm_train`, reached only under `mode: train`.

So the column exists in both modes but is *acted on* only in train. Production distinct-incident count is untouched.

## Changes
1. **features.py** — `build_feature_matrix(..., sim_col=None)`. When `sim_col` is present in the incidents frame it is merged through as a passthrough (one value per incident). Never added to `FEATURE_COLS`.
2. **pipeline.py** — `build_features` passes `sim_col = cfg.gbm_train.similarity_col` so train + production call the builder identically.
3. **train.py** — new pure function `filter_weak_links(labeled, sim_col, min_sim)`:
   - `min_sim is None` -> no-op (returns frame unchanged).
   - `sim_col` missing -> WARNING + skip (no drop).
   - else drop rows with `sim < min_sim`; raise if all rows dropped (threshold too high).
   - Called after the unlabeled-row drop, before the by-incident split.
4. **tests/test_train_filter.py** — 6 tests pinning both halves of the contract:
   - drops below / keeps `>=` threshold
   - `None` threshold is a no-op (the production case)
   - missing column skips + warns
   - too-high threshold raises
   - `similarity_col` is never in `FEATURE_COLS`
   - `build_feature_matrix` carries the passthrough but it stays out of the features

## Verification
```
py_compile train.py features.py pipeline.py   -> OK
pytest tests/                                  -> 83 passed, 6 skipped
```

## Note on the threshold value
`min_semantic_similarity: 0.35` is inclusive — rows with cosine **exactly 0.35 are kept**, below are dropped. Change the value in config to retune; `null` disables the filter entirely.
