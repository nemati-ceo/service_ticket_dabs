# 01b — PII Redaction · flow

The gate. Stage 02 and stage 05 send text off-cluster, so every table either of them can
reach passes through here first. Redaction is **in place and irreversible** — the span is
replaced by an `<ENTITY>` tag, and the mirror is overwritten.

```mermaid
flowchart TB

  subgraph IN["inputs · pii_redaction.tables"]
    direction LR
    A1[("ph01_IncidentScore")]
    A2[("cluster_synced")]
    A3[("problemzeroincidents_synced")]
  end

  ADD["sparkContext.addPyFile(redact.py)"]
  UDF["build pandas_udf"]

  subgraph EXEC["on each EXECUTOR"]
    direction TB
    CACHE{{"_ENGINES cached?"}}
    BUILD["build_engines()<br>spaCy en_core_web_lg from Volume<br>+ Presidio recognizers"]
    SCRUB["redact_text per row<br>score_threshold 0.35"]
  end

  COLS["withColumn(c, udf(c))<br>per text column"]
  W["write.mode(overwrite)"]

  subgraph OUT["redacted mirrors"]
    direction LR
    B1[("ph01b_Redacted")]
    B2[("ph01b_Redacted_Unlinked")]
    B3[("ph01b_Redacted_ProblemsZero")]
  end

  IN --> ADD
  ADD -->|"workers cannot import<br>the driver's sys.path"| UDF
  UDF --> CACHE
  CACHE -->|"no · once per process"| BUILD
  BUILD --> SCRUB
  CACHE -->|"yes · reuse"| SCRUB
  SCRUB -->|"PERSON, EMAIL, PHONE,<br>LOCATION, USER_ID, STREET_ADDRESS"| COLS
  COLS --> W
  W --> B1
  W --> B2
  W --> B3

  classDef stage fill:#ffffff,stroke:#16202b,stroke-width:1.4px,color:#16202b;
  classDef store fill:#e6f0ef,stroke:#0f6b6b,stroke-width:1px,color:#0f4f4f;
  classDef guard fill:#ffffff,stroke:#16202b,stroke-width:1.4px,color:#16202b,stroke-dasharray:4 3;

  class ADD,UDF,BUILD,SCRUB,COLS,W stage;
  class A1,A2,A3,B1,B2,B3 store;
  class CACHE guard;
```

## Why `addPyFile` is in the picture

The `pandas_udf` closes over `redact.py`, and cloudpickle serializes that **by reference** —
the worker does `import redact`, which fails with `ModuleNotFoundError` because the driver's
`sys.path` entry for this stage folder does not propagate to Python workers. Shipping the
file explicitly is what makes the UDF runnable at all.

## Why the engine cache is a module global

`en_core_web_lg` costs seconds and hundreds of megabytes to load. Building it per row — or
even per batch — would dominate the stage. It is built once per executor **process** and
stashed on the module, so every later batch on that worker reuses it.

## The threshold trap

`score_threshold` **must stay below 0.4**. Presidio scores `PHONE_NUMBER` at exactly 0.4, so
a threshold of 0.5 still redacts names and emails — the output looks correct — while
silently leaking every phone number. Pinned by `tests/test_pii_redaction.py`.

A text column listed in config but absent from the table only **warns**; if *none* of the
listed columns exist the stage raises, because that means the config points at the wrong
table and nothing would be redacted at all.

See [`README.md`](README.md) for entities, recognizers and MLflow keys.
