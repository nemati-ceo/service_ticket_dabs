# Cleanup: config.yml comment trim

**File:** `src/ServiceTicket/config.yml`
**Date:** 2026-07-17

## What
Collapsed all multi-line `#` comment blocks to single-line comments. No config values, keys, or SQL changed.

## Why
Comment paragraphs were long. Requested: keep one short `#` line per block, drop the rest.

## Result
- Every comment block now 1 line.
- Critical safety warnings preserved compactly:
  - `secrets.scope` — TEST scope, swap for prod on deploy.
  - `pii_redaction` — ai_query() only egress; unredacted = PII leak.
  - `score_threshold` — MUST stay below 0.4 (Presidio PHONE_NUMBER = 0.4).
  - `gbm_inference.problem_sql` — MUST be JOIN (ph02 has no business_service).
  - `gbm_train.min_semantic_similarity` — TRAIN ONLY, never production.
- Validated: `yaml.safe_load` OK, 19 top-level keys intact.

## Verify
```
python3 -c "import yaml; yaml.safe_load(open('src/ServiceTicket/config.yml'))"
```
