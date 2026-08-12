# 04 — Gradient Boosting · flow

One stage, two modes off the same feature builder. Production scores and publishes the
sheet; `mode: train` fits a new model and writes no linking table. They share
`build_feature_matrix` on purpose — a separate training path is how train/serve skew starts.

```mermaid
%%{init: {"theme":"base","themeVariables":{
  "fontFamily":"ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
  "fontSize":"13px","primaryColor":"#ffffff","primaryTextColor":"#16202b",
  "primaryBorderColor":"#8fa2b2","lineColor":"#5c6f80",
  "clusterBkg":"#eef2f5","clusterBorder":"#b6c4d0"
}, "flowchart":{"curve":"linear","nodeSpacing":36,"rankSpacing":40}}%%
flowchart TB

  R[("ph03_RerankedScores")]
  I[("ph01b_Redacted<br>incidents")]
  P[("ph02_ProblemSummaries<br>⋈ business_service")]

  FM["build_feature_matrix<br>joins BY ID, never by position"]
  F[["cosine_sim · reranker_score · bs_match<br>+ label, + passthroughs"]]
  BS{{"bs_match all zero?"}}

  MODE{{"mode"}}

  FLT["filter_weak_links<br>drop cosine-to-gold &lt; 0.35"]
  SPLIT["holdout split BY INCIDENT"]
  FIT["GradientBoostingClassifier"]
  PKL[("PH04_gradient_boosting_model.pkl<br>Volume")]

  LOAD["joblib.load · once per run"]
  NF{{"n_features_in_ == len(FEATURE_COLS)"}}
  SCORE["predict_proba in batches<br>→ gbm_propensity"]
  RANK["rank_within_incident<br>sort by propensity desc"]
  PIVOT["pivot top-N to wide<br>pid · description · score · similarity"]
  TWIN["join summarized twin<br>ON linked_problem_id == problem_id"]
  OUT[("ph04_Linking_Top10")]

  R --> FM
  I --> FM
  P --> FM
  FM --> F
  F --> BS
  BS -->|"warn: model runs on 2 of 3"| MODE
  F --> MODE

  MODE -->|"train"| FLT
  FLT -->|"weak link = bad label"| SPLIT
  SPLIT --> FIT
  FIT --> PKL

  MODE -->|"production"| LOAD
  PKL -.-> LOAD
  LOAD --> NF
  NF --> SCORE
  SCORE --> RANK
  RANK --> PIVOT
  PIVOT --> TWIN
  TWIN --> OUT

  classDef stage fill:#ffffff,stroke:#16202b,stroke-width:1.4px,color:#16202b;
  classDef store fill:#e6f0ef,stroke:#0f6b6b,stroke-width:1px,color:#0f4f4f;
  classDef guard fill:#ffffff,stroke:#16202b,stroke-width:1.4px,color:#16202b,stroke-dasharray:4 3;

  class FM,FLT,SPLIT,FIT,LOAD,SCORE,RANK,PIVOT,TWIN stage;
  class R,I,P,PKL,OUT,F store;
  class BS,MODE,NF guard;
```

## Why the weak-link filter is train-only

`min_semantic_similarity` drops incidents whose cosine to their **gold** problem is weak —
a bad link is a bad label. In production the gold problem is the thing being predicted, so
filtering on it would leak the answer and skip exactly the incidents that most need linking.
It is kept out structurally: `similarity_col` is a passthrough that is deliberately **not**
in `FEATURE_COLS`, so `inference.score` reads `feature_df[FEATURE_COLS]` and can never see
it.

## Why two guards sit on the model

`bs_match` fails **silently**: if `business_service` is missing on either side the feature
is 0 for every row and the model quietly runs on two of its three inputs, with every
downstream number still looking plausible. And sklearn only checks feature *count*, not
names or order — a model fitted on a different column set scores garbage without raising.
Both are turned into loud output.

## Why the twin joins on `linked_problem_id`

`summary_similarity` is `(number, linked problem)` grain. An incident linked to two problems
has two of them, and picking one by row order put one problem's summarized score beside
another problem's raw score — a lift the summarizer never gave. The join is on the gold id
the sheet actually publishes. Pinned by `tests/test_linking.py`.

See [`README.md`](README.md) for the sheet's column reference.

PNG export (1920px-wide, for slides): [`diagram.png`](diagram.png).
