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
| `visualize.py` | 2-D UMAP projection + Plotly scatter (logged to MLflow, not written to a Volume) |

## Input
A live Delta table or SQL query (no parquet) with the text to cluster plus
display/category columns: `number`, `text_col` (default `summary_final`),
`short_description`, `business_service`, `u_service_feature`, `u_incident_type`.

> **Gap-fill summarization:** before clustering, stage 05 calls stage-02's
> `summarize_entity` over `summarize_source_sql` (the full set of tickets to
> cluster). The hash-keyed `MERGE` means already-summarized tickets are **reused**
> (no re-billing) and only the **missing** ones are summarized. Output goes to stage
> 05's OWN table `ph05_output_UnlinkedSummaries` — sharing stage 02's table let its
> `drop_deleted` wipe these every run and re-bill the LLM for every unlinked ticket.
> Set `summarize_gap: false` to skip.

## Outputs
| Target | Content |
|---|---|
| Delta `ph05_output_ClusterThemes` | per-incident `cluster` + `theme_group` (+ export cols) |
| Delta `ph05_output_ThemeOverlay` | per-theme counts + top categorical values |
| **MLflow run** `ph05_clustering` | params (umap/hdbscan/merge), metrics (clusters, noise, silhouette, themes), and the 2-D scatter as `clusters_2d.html` (+ `.png` if kaleido) |

No Volume file-writes — the data goes to UC Delta tables and everything else
(plot, metrics, params) is logged to MLflow.

## Guarantees this stage enforces
| Check | Fails how |
|---|---|
| both output tables raise on a failed write | the overlay write used to warn and continue, leaving a stale overlay beside a fresh cluster table |
| input comes from a live Delta table / SQL — no parquet fallback | a swallowed table error would silently cluster yesterday's file |
| input must have `text_col`, and not every row blank | a `KeyError` deep in a filter, or an empty list handed to the embedder |
| the scatter refuses a projection that does not match its frame | misaligned points draw a plot that looks perfectly fine |
| embedder loads from a Volume cache | it re-downloaded a few hundred MB on **every** run |

## Notes
- Test mode via the shared `run.limit`.
- UMAP/HDBSCAN/merge knobs live under `clustering:` in `../config.yml`.
- **Model load:** `clustering.embed_model_volume_path` caches the embedder on a Volume.
  Shared with stage 03 via root-level `model_cache.load_cached`. Without the path set,
  the load warns and re-downloads every run.
- **Merging is transitive:** A~B and B~C fuse A, B and C even when A·C is below
  `merge_threshold`. A too-low threshold collapses unrelated themes into one giant group —
  compare `n_themes` against `n_clusters` before trusting a run.
- **The plot is sampled** (`plot_sample_size`, default 20000, seeded). It runs a *second*
  UMAP purely for the picture, and every point with hover text makes an artifact too big
  to open.
- **Timing:** per-step laps print a breakdown and reach MLflow as `secs_*`.

Config: `clustering:` section in the shared root `../config.yml`.
