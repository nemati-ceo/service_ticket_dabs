# 01 — Problem Health · flow

The stage encodes two populations and then compares them row by row. Incidents are encoded
one-for-one; problems are **deduplicated first**, encoded once each, then mapped back out to
every incident that references them. That map-back is the only reason the two arrays line up.

```mermaid
%%{init: {"theme":"base","themeVariables":{
  "fontFamily":"ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
  "fontSize":"13px","primaryColor":"#ffffff","primaryTextColor":"#16202b",
  "primaryBorderColor":"#8fa2b2","lineColor":"#5c6f80",
  "clusterBkg":"#eef2f5","clusterBorder":"#b6c4d0"
}, "flowchart":{"curve":"linear","nodeSpacing":36,"rankSpacing":40}}%%
flowchart TB

  IN[("incidentstoopenproblem_synced")]
  L["1 · LOAD"]
  C["2 · CLEAN"]
  MDL[/"3 · all-MiniLM-L6-v2 · onnx<br>Volume → download → registry"/]

  EI["4 · encode incidents"]
  DD["4 · drop_duplicates(problem_id)"]
  EP["4 · encode unique problems"]
  MB["4 · map back by problem_id"]

  G{{"guard: len(df) == len(emb)"}}
  COS["5 · row-wise cosine"]
  O1[("ph01_IncidentScore<br>+ semantic_similarity")]
  AGG["6 · mean per problem"]
  O2[("ph01_ProblemHealth<br>ProblemHealth_Score")]

  IN -->|"N rows · Arrow-safe projection"| L
  L -->|"drops VOID + _databricks_internal<br>restores dotted names"| C
  C -->|"combined_cleaned_desc"| EI
  C -->|"combined_prob_desc"| DD
  MDL --> EI
  MDL --> EP

  EI -->|"N vectors"| G
  DD -->|"U unique problems"| EP
  EP -->|"U vectors"| MB
  MB -->|"N vectors, one per incident row"| G

  G -->|"raises if a reorder<br>changed the row count"| COS
  COS -->|"N scores, clipped to [-1,1]"| O1
  O1 --> AGG
  AGG -->|"P problems"| O2

  classDef stage fill:#ffffff,stroke:#16202b,stroke-width:1.4px,color:#16202b;
  classDef store fill:#e6f0ef,stroke:#0f6b6b,stroke-width:1px,color:#0f4f4f;
  classDef guard fill:#ffffff,stroke:#16202b,stroke-width:1.4px,color:#16202b,stroke-dasharray:4 3;

  class L,C,MDL,EI,DD,EP,MB,COS,AGG stage;
  class IN,O1,O2 store;
  class G guard;
```

`N` = incidents, `U` = distinct problems, `P` = problems with at least one incident.
Problems are encoded `U` times, not `N` — the same problem text is reused across every
incident linked to it.

## Why the dedup is safe

Encoding runs on `U` unique problems, but the map-back reattaches a vector to every one of
the `N` incident rows. No incident is dropped — the output keeps the full incident grain,
and only the encode cost shrinks.

## Why the guard exists

Embeddings are paired to rows **by position**, and that link lives nowhere but in memory. A
reorder between encoding and scoring keeps the array shape identical, so the cosine still
computes — it just scores every row against the wrong problem. The row-count check in
`similarity.add_similarity` is what turns that silent corruption into a crash.

See [`README.md`](README.md) for the stage's modules, outputs and MLflow keys.

PNG export (1920px-wide, for slides): [`diagram.png`](diagram.png).
