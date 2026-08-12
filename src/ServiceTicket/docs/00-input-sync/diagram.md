# 00 — Input Sync · flow

Three tables cross one boundary. `refine` belongs to data engineering and is never written
to; the pipeline works only on its own `consume` mirrors, so it can redact them, add columns
and re-run without touching anyone else's table.

```mermaid
%%{init: {"theme":"base","themeVariables":{
  "fontFamily":"ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
  "fontSize":"13px","primaryColor":"#ffffff","primaryTextColor":"#16202b",
  "primaryBorderColor":"#8fa2b2","lineColor":"#5c6f80",
  "clusterBkg":"#eef2f5","clusterBorder":"#b6c4d0"
}, "flowchart":{"curve":"linear","nodeSpacing":36,"rankSpacing":40}}%%
flowchart TB

  subgraph SRC["redzone_refine.servicenow_incidents_problems · READ-ONLY"]
    direction LR
    R1[("incidentstoopenproblem")]
    R2[("problemzeroincidents")]
    R3[("cluster")]
  end

  EN{{"input_sync.enabled?"}}
  SKIP["skip · return None"]
  PAIRS["_sources(cfg)<br>tables: [{source, target}]"]
  CP["_copy_table<br>per pair"]
  STAMP["+ last_synced_at"]
  W["write.mode(overwrite)<br>overwriteSchema"]

  M1[("incidentstoopenproblem_synced")]
  M2[("problemzeroincidents_synced")]
  M3[("cluster_synced")]

  ML["MLflow ph00<br>rows__&lt;table&gt;, rows_total"]

  SRC --> EN
  EN -->|"false"| SKIP
  EN -->|"true"| PAIRS
  PAIRS --> CP
  CP -->|"spark.table(source)"| STAMP
  STAMP --> W
  W -->|"full replace, not merge"| M1
  W --> M2
  W --> M3
  CP -.->|"raises → STOPS the pipeline"| ML
  W --> ML

  classDef stage fill:#ffffff,stroke:#16202b,stroke-width:1.4px,color:#16202b;
  classDef store fill:#e6f0ef,stroke:#0f6b6b,stroke-width:1px,color:#0f4f4f;
  classDef guard fill:#ffffff,stroke:#16202b,stroke-width:1.4px,color:#16202b,stroke-dasharray:4 3;

  class PAIRS,CP,STAMP,W,SKIP,ML stage;
  class R1,R2,R3,M1,M2,M3 store;
  class EN guard;
```

## Why there is no delete logic

The engineers drop a **complete** snapshot every run, so the new snapshot is the whole
truth. A closed or removed record disappears by being absent from it. There is no
add/update/delete path to get wrong — and nothing to reconcile when one goes stale.

## Why a failure here stops everything

`_copy_table` lets its exception propagate. A half-written mirror would still be readable
by stage 01, which would score it, and every stage after that would report success on
partial data. Failing loudly at the front is the only place that is cheap to notice.

## The gap worth knowing

There is **no empty-snapshot guard**. If an empty or truncated snapshot lands in `refine`,
stage 00 will overwrite good mirrors with it and every downstream stage runs on nothing.
Recovery is Delta time-travel (`VERSION AS OF`). A minimum-row-count check would be cheap
insurance before this reaches prod.

See [`README.md`](README.md) for config and MLflow keys.

PNG export (1920px-wide, for slides): [`diagram.png`](diagram.png).
