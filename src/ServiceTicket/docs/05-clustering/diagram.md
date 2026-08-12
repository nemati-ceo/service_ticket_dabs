# 05 — Clustering · flow

The other half of the pipeline: tickets with **no** problem record. There is nothing to
recommend against, so instead they are grouped into themes. Embedding happens once
globally; clustering happens per assignment group.

```mermaid
%%{init: {"theme":"base","themeVariables":{
  "fontFamily":"ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
  "fontSize":"13px","primaryColor":"#ffffff","primaryTextColor":"#16202b",
  "primaryBorderColor":"#8fa2b2","lineColor":"#5c6f80",
  "clusterBkg":"#eef2f5","clusterBorder":"#b6c4d0"
}, "flowchart":{"curve":"linear","nodeSpacing":36,"rankSpacing":40}}%%
flowchart TB

  UN[("ph01b_Redacted_Unlinked")]
  GAP["gap-fill summarize_entity<br>stage 02's summarizer"]
  OWN[("ph05_UnlinkedSummaries<br>stage 05's OWN table")]

  LOAD["load input_sql"]
  BLANK["drop blank summary_final"]
  NORM["normalize assignment_group"]
  EMB["embed ALL tickets · ONE pass<br>all-MiniLM-L6-v2"]
  SPLIT["split by assignment_group"]

  subgraph LOOP["per group"]
    direction TB
    SMALL{{"rows &lt; min_cluster_rows?"}}
    ALONE["stands alone · all noise"]
    UMAP["UMAP → 5-D"]
    HDB["HDBSCAN → cluster labels<br>-1 = noise"]
    CENT["cluster centroids"]
    MERGE["union-find merge<br>centroid cosine ≥ 0.9"]
    THEME["theme_group"]
  end

  CONCAT["concat groups<br>realign embeddings to the new order"]
  OV["theme_overlay<br>count + dominant category"]
  X1[("ph05_ClusterThemes<br>per ticket")]
  X2[("ph05_ThemeOverlay<br>per theme")]
  PLOT["2-D UMAP scatter → MLflow"]

  UN ==>|"ai_query"| GAP
  GAP --> OWN
  OWN --> LOAD
  UN --> LOAD
  LOAD --> BLANK
  BLANK --> NORM
  NORM --> EMB
  EMB --> SPLIT
  SPLIT --> SMALL
  SMALL -->|"yes"| ALONE
  SMALL -->|"no"| UMAP
  UMAP --> HDB
  HDB --> CENT
  CENT --> MERGE
  MERGE --> THEME
  ALONE --> CONCAT
  THEME --> CONCAT
  CONCAT --> X1
  CONCAT --> OV
  OV --> X2
  CONCAT --> PLOT

  classDef stage fill:#ffffff,stroke:#16202b,stroke-width:1.4px,color:#16202b;
  classDef store fill:#e6f0ef,stroke:#0f6b6b,stroke-width:1px,color:#0f4f4f;
  classDef llm fill:#f7e6e2,stroke:#a03328,stroke-width:1.6px,color:#7d271e;
  classDef guard fill:#ffffff,stroke:#16202b,stroke-width:1.4px,color:#16202b,stroke-dasharray:4 3;

  class LOAD,BLANK,NORM,EMB,SPLIT,UMAP,HDB,CENT,MERGE,THEME,ALONE,CONCAT,OV,PLOT stage;
  class UN,OWN,X1,X2 store;
  class GAP llm;
  class SMALL guard;
```

## Why embedding is outside the group loop

Every ticket gets one vector regardless of which group it lands in. Encoding per group would
re-batch — and on a cold cache reload the model — once per group, for identical vectors.
Clustering is what varies per group, so only that is in the loop.

## Why the ids are not globally unique

`cluster` and `theme_group` restart at 0 inside each group. The real key is
`(assignment_group, theme_group, cluster)`. Reading `theme_group = 3` without its group is
meaningless.

## Why the embeddings are realigned after the loop

Groups are concatenated back in group order, not original row order. The 2-D plot colours
each point by its vector, so skipping the realignment would paint every point with another
ticket's embedding — a plot that looks fine and means nothing.

## Why the gap-fill writes its own table

Sharing `ph02_output_IncidentSummaries` let stage 02's `drop_deleted` wipe these rows every
run, re-billing the LLM for every unlinked ticket. Its spend is logged separately as
`gapfill_*` on the `ph05` run.

## What cannot be validated yet

`cluster_synced` currently holds ~10 tickets in one assignment group. Below
`min_cluster_rows` every group takes the stand-alone path — all noise, no merges, empty
overlay. The group-loop branches are covered by synthetic tests; re-check the knobs when
real volume lands.

See [`README.md`](README.md) for how to read the output tables.

PNG export (1920px-wide, for slides): [`diagram.png`](diagram.png).
