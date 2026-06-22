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

> The text is expected **already summarized** (e.g. stage 02 output). This stage
> does **not** re-run the LLM — that would duplicate stage 02. Point `text_col`
> at the summarized column (or `combined_text` for raw text).

## Outputs (Unity Catalog — never CSV)
| Target | Content |
|---|---|
| Delta `ph05_output_ClusterThemes` | per-incident `cluster` + `theme_group` (+ export cols) |
| Delta `ph05_output_ThemeOverlay` | per-theme counts + top categorical values |
| Volume `clusters_2d.html` / `clusters_2d.png` | interactive + static cluster scatter |
| Volume `ClusterThemes.parquet`, `ThemeOverlay.parquet` | same tables, parquet |

## Notes
- Test mode via the shared `run.limit`.
- UMAP/HDBSCAN/merge knobs live under `clustering:` in `../config.yml`.

Config: `clustering:` section in the shared root `../config.yml`.
