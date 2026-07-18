# 01b — PII Redaction

The security boundary. Sits between stage 01 and stage 02: stage 02 is the only stage
that sends text off-cluster, so **every text column it can reach is scrubbed here first**.
PII spans are replaced with `<ENTITY>` placeholders — irreversible, no re-identification
map is kept. Downstream stages read only the redacted tables.

## Modules
| File | Role |
|---|---|
| `pipeline.py` | orchestrator: build the udf once per executor, redact each table, mlflow |
| `redact.py` | Presidio + spaCy engine construction and text scrubbing |
| `stage_model.py` | one-time, offline: download the spaCy model into the Volume |

## Model loading — never downloads at runtime
`redact.build_engines` loads spaCy by **Volume PATH**, never by name, and **raises
FileNotFoundError** if the path is missing. The redzone firewalls spacy.io, so a runtime
`spacy.load("en_core_web_lg")` hangs instead of failing fast. Stage the model once:

```
python stage_model.py            # uses config.yml; SKIPS if already staged
python stage_model.py --force    # re-download and overwrite
```

Engines are built **once per executor** and cached on the module — loading `en_core_web_lg`
costs seconds and hundreds of MB, so never per row or per batch.

## Tables (all live Delta)
Every table whose text can reach the LLM must be listed in `pii_redaction.tables` —
`ai_query()` is the only egress point, and an unlisted table leaks raw PII off-cluster.

| Input | Output |
|---|---|
| `ph01_output_IncidentScore_SemanticSimilarity` | `ph01b_output_Redacted` |
| `cluster_synced` (unlinked incidents) | `ph01b_output_Redacted_Unlinked` |
| `problemzeroincidents_synced` | `ph01b_output_Redacted_ProblemsZero` |

## score_threshold — must stay below 0.4
Presidio scores a `PHONE_NUMBER` match at exactly **0.4**. A threshold of 0.5 silently
drops **every phone number** while still redacting names and emails, so the redaction
looks like it works and quietly leaks. Config uses `0.35`; `redact_text`'s default is
also 0.35 so a bare call can't reintroduce the trap. Pinned by a test.

## Fail-hard by design
`build_engines` raises on a missing model rather than degrading — you must never
"redact with nothing" and let raw PII flow downstream. No best-effort swallowing exists
anywhere text could leak.

## MLflow (`ph01b_pii_redaction`)
Wraps the redaction work, so a crash at the PII boundary lands as a FAILED run.
Params: `spacy_model`, `model_path`, `entities`, `score_threshold`.
Metrics: `rows_redacted`, `tables_redacted`, `wall_clock_s`.

Config: `pii_redaction:` in the shared root `../config.yml`.
