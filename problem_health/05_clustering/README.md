# 05 — Clustering (Theme Grouping)

Groups incidents into **themes** by embedding their summarized text, reducing
with UMAP, clustering with HDBSCAN, then merging near-duplicate clusters (by
centroid cosine) into themes. Produces a per-incident cluster/theme assignment,
a per-theme categorical overlay, and an interactive 2-D scatter.

## Modules
| File | Role |
|---|---|
| `run.py` | shared entry point (`run.stage05()`) — loads root `config.yml` |
| `pipeline.py` | orchestrator: load → embed → reduce → cluster → merge → overlay → viz → save |
| `clustering.py` | embed, UMAP reduce, HDBSCAN, cluster stats (silhouette) |
| `merge.py` | centroid cosine + union-find to merge clusters into themes |
| `overlay.py` | per-theme incident counts + dominant categorical values |
| `visualize.py` | 2-D UMAP projection + Plotly scatter saved to Volume (HTML) |

## Input
A table/parquet with the text to cluster plus display/category columns:
`number`, `text_col` (default `summary_final`), `short_description`,
`business_service`, `u_service_feature`, `u_incident_type`.

> **Gap-fill summarization:** before clustering, stage 05 calls stage-02's
> `summarize_entity` over `summarize_source_sql` (the full set of tickets to
> cluster). The hash-keyed `MERGE` means already-summarized tickets are **reused**
> (no re-billing) and only the **missing** ones are summarized into `ph02`
> (`drop_deleted=False`, so linking summaries are never removed). Result: no ticket
> is dropped and no summary is computed twice. Set `summarize_gap: false` to skip.

## Outputs
| Target | Content |
|---|---|
| Delta `ph05_output_ClusterThemes` | per-incident `cluster` + `theme_group` (+ export cols) |
| Delta `ph05_output_ThemeOverlay` | per-theme counts + top categorical values |
| **MLflow run** `ph05_clustering` | params (umap/hdbscan/merge), metrics (clusters, noise, silhouette, themes), and the 2-D scatter as `clusters_2d.html` (+ `.png` if kaleido) |

No Volume file-writes — the data goes to UC Delta tables and everything else
(plot, metrics, params) is logged to MLflow.

## Notes
- Test mode via the shared `run.limit`.
- UMAP/HDBSCAN/merge knobs live under `clustering:` in `../config.yml`.

Config: `clustering:` section in the shared root `../config.yml`.
