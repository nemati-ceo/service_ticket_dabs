# 02 — LLM Summarization · flow

The stage is mostly a cache. The LLM call is one node in the middle; everything around it
exists so a row is sent **once** and never re-billed unless its text, the prompt, or the
model actually changed.

```mermaid
flowchart TB

  SRC[("ph01b_Redacted<br>+ ph01b_Redacted_ProblemsZero")]
  V1["src view<br>ROW_NUMBER() … _rn = 1"]
  HASH["summary_input_hash =<br>md5(text ‖ prompt ‖ model)"]
  ANTI["changed view<br>LEFT ANTI JOIN on (key, hash)"]
  Z{{"changed == 0?"}}
  SKIP["no LLM call<br>tokens = 0"]

  LLM["ai_query(model, prompt ‖ text)<br>failOnError ⇒ false"]
  STG[("staging TABLE<br>materialized, not a view")]
  FB["NO_CONTENT / null<br>→ keep original text"]
  TOK["count tokens<br>fallback rows excluded from output"]
  MRG["MERGE upsert by key"]
  DD["MERGE … NOT MATCHED BY SOURCE<br>THEN DELETE"]

  OUT[("ph02_IncidentSummaries<br>ph02_ProblemSummaries")]

  SRC -->|"one incident can arrive<br>on several problems"| V1
  V1 --> HASH
  HASH --> ANTI
  OUT -.->|"what was already summarized"| ANTI
  ANTI --> Z
  Z -->|"yes"| SKIP
  Z -->|"no"| LLM
  LLM --> STG
  STG --> FB
  STG --> TOK
  FB --> MRG
  MRG --> OUT
  OUT --> DD
  DD -->|"keys gone from the source"| OUT

  classDef stage fill:#ffffff,stroke:#16202b,stroke-width:1.4px,color:#16202b;
  classDef store fill:#e6f0ef,stroke:#0f6b6b,stroke-width:1px,color:#0f4f4f;
  classDef llm fill:#f7e6e2,stroke:#a03328,stroke-width:1.6px,color:#7d271e;
  classDef guard fill:#ffffff,stroke:#16202b,stroke-width:1.4px,color:#16202b,stroke-dasharray:4 3;

  class V1,HASH,ANTI,FB,TOK,MRG,DD,SKIP stage;
  class SRC,OUT,STG store;
  class LLM llm;
  class Z guard;
```

## Why the hash includes the prompt and the model

Text alone is not enough. Edit a prompt or swap the model and the old summaries are stale,
but their text is unchanged — a text-only key would keep serving them forever. Folding both
into the key means an edit re-summarizes exactly the rows it affects, and nothing else.

The flip side is the cost of a prompt change: **every** row's hash moves, so the whole
corpus is re-billed. That is the price tag on adding an acronym glossary.

## Why `_rn = 1` comes before anything else

The source key is `(number, problem_id)`, so an incident linked to three problems arrives
three times. Without the dedup the LLM is billed three times for identical text, and the
`MERGE` fails outright with `DELTA_MULTIPLE_SOURCE_ROW_MATCHING_TARGET_ROW`.

## Why the results are written to a table, not a view

A lazy view over `ai_query()` re-executes on every action. Counting fallbacks and then
merging would run — and bill — the model twice. Materializing to a staging table forces
exactly one execution.

## Two egress points, one summarizer

Stage 05's gap-fill calls this same `summarize_entity` on the unlinked tickets, writing to
its own table. Its spend lands on the `ph05` run as `gapfill_*`, so a run's true LLM cost is
`ph02_tokens_total` + `ph05_gapfill_tokens_total`.

See [`README.md`](README.md) for token accounting and MLflow keys.
