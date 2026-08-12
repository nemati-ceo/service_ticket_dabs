# 03 — Cross-encoder Rerank · flow

Two models in series, for a reason. Scoring every incident against every problem with a
cross-encoder is `N × catalog` forward passes; the bi-encoder cuts that to `N × 50` by
shortlisting first, and only then does the expensive model run.

```mermaid
%%{init: {"theme":"base","themeVariables":{
  "fontFamily":"ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
  "fontSize":"13px","primaryColor":"#ffffff","primaryTextColor":"#16202b",
  "primaryBorderColor":"#8fa2b2","lineColor":"#5c6f80",
  "clusterBkg":"#eef2f5","clusterBorder":"#b6c4d0"
}, "flowchart":{"curve":"linear","nodeSpacing":36,"rankSpacing":40}}%%
flowchart TB

  I1[("ph02_IncidentSummaries<br>⋈ ph01b_Redacted")]
  I2[("ph02_ProblemSummaries")]

  BI["bi-encoder · all-MiniLM-L6-v2<br>loaded ONCE"]
  EI["encode incidents → N vectors"]
  EP["encode catalog → C vectors"]

  TOP["top_k_candidates_from_embeddings<br>chunked argpartition"]
  CI[["candidate_indices (N × k)<br>cosine_sim (N × k)"]]
  CLAMP{{"k = min(top_k, C)"}}
  BOUND{{"max(index) &lt; C<br>rows == N"}}

  CE["cross-encoder · ms-marco-MiniLM<br>buffered in chunks of 5000 pairs"]
  SIG["sigmoid → [0,1]"]

  GOLD["gold-pair cosine<br>incident ⋈ its linked problem"]

  OUT[("ph03_RerankedScores<br>one row per incident × candidate")]

  I1 --> BI
  I2 --> BI
  BI --> EI
  BI --> EP
  EI --> TOP
  EP --> TOP
  TOP --> CLAMP
  CLAMP --> CI
  CI --> BOUND
  BOUND -->|"N × k pairs"| CE
  CE -->|"raw logits"| SIG

  EI -.->|"reuses the same vectors"| GOLD
  EP -.-> GOLD

  SIG --> OUT
  CI --> OUT
  GOLD -->|"summary_similarity<br>+ linked_problem_id"| OUT

  classDef stage fill:#ffffff,stroke:#16202b,stroke-width:1.4px,color:#16202b;
  classDef store fill:#e6f0ef,stroke:#0f6b6b,stroke-width:1px,color:#0f4f4f;
  classDef guard fill:#ffffff,stroke:#16202b,stroke-width:1.4px,color:#16202b,stroke-dasharray:4 3;

  class BI,EI,EP,TOP,CE,SIG,GOLD stage;
  class I1,I2,OUT,CI store;
  class CLAMP,BOUND guard;
```

`N` = linked incidents, `C` = problems in the catalog, `k` = candidates kept per incident
(`top_k: 50`, clamped down when the catalog is smaller).

## Why the shortlist never builds the full matrix

`N × C` cosines would materialize a matrix the size of the whole cross product. The
shortlist walks incidents in chunks, computes one chunk against the catalog, takes its
top-`k` with `argpartition`, and drops the rest before the next chunk.

## Why `top_k` clamping is drawn

If the catalog holds fewer problems than `top_k`, the shortlist returns that many instead —
37 problems with `top_k: 50` gives 37 candidates per incident. Every count in the logs and
in MLflow is taken from the **actual** shortlist width, so `pairs_reranked` reflects work
really done rather than work requested.

## Why the gold-pair cosine is free here

Stage 01 scores each incident against its already-linked problem on raw text. The same pair
scored on **summaries** is one dot product against embeddings this stage already has in
memory — no extra encode, no LLM call. `linked_problem_id` rides along because the grain is
`(number, linked problem)`, not incident: an incident linked to two problems produces two
rows with two different scores, and stage 04 needs the id to publish the right one.

See [`README.md`](README.md) for the output schema and MLflow keys.

PNG export (1920px-wide, for slides): [`diagram.png`](diagram.png).
