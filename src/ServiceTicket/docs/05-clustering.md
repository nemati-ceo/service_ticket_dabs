# 05 — Clustering (Theme Grouping)

Groups incidents into **themes** by embedding their summarized text, reducing
with UMAP, clustering with HDBSCAN, then merging near-duplicate clusters (by
centroid cosine) into themes. Produces a per-incident cluster/theme assignment,
a per-theme categorical overlay, and an interactive 2-D scatter.

Clustering is **partitioned by assignment group** (`clustering.group_col`): every
ticket is embedded once, then reduce → cluster → merge runs inside each group on
its own. Tickets from two different assignment groups can never land in the same
theme.

## Modules
| File | Role |
|---|---|
| `run.py` | shared entry point (`run.stage05()`) — loads root `config.yml` |
| `pipeline.py` | orchestrator: load → embed → **per group:** reduce → cluster → merge → overlay → viz → save |
| `clustering.py` | embed, UMAP reduce, HDBSCAN, cluster stats (silhouette) |
| `merge.py` | centroid cosine + union-find to merge clusters into themes |
| `overlay.py` | per-theme incident counts + dominant categorical values |
| `visualize.py` | 2-D UMAP projection + Plotly scatter (logged to MLflow, not written to a Volume) |

## Input
A live Delta table or SQL query (no parquet) with the text to cluster plus
display/category columns: `number`, `text_col` (default `summary_final`),
`short_description`, `business_service`, `u_service_feature`, and the group key
`assignment_group`.

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
| Delta `ph05_output_ClusterThemes` | per-incident `assignment_group` + `cluster` + `theme_group` + `cluster_status` (+ export cols) |
| Delta `ph05_output_ThemeOverlay` | per-group, per-theme counts + top categorical values |
| **MLflow run** `ph05_clustering` | params (group_col, umap/hdbscan/merge), run-level metrics, `per_group_stats.json`, `merge_log.json`, and the 2-D scatter as `clusters_2d.html` (+ `.png` if kaleido) |

Both tables are **fully overwritten** every run: last run's assignments are removed,
never merged into. No Volume file-writes — the data goes to UC Delta tables and
everything else (plot, metrics, params) is logged to MLflow.

---

# How to read the results

## The three levels

```
assignment_group          partition — clustering never crosses it
  └─ theme_group          merged clusters (centroid cosine >= merge_threshold)
       └─ cluster         raw HDBSCAN label, -1 = noise
            └─ ticket     one row in ph05_output_ClusterThemes
```

**Cluster and theme ids restart inside every group.** `cluster = 0` under
`ITSM.Field Career.T2` has nothing to do with `cluster = 0` under another group.
The key is the pair — always filter or group by `assignment_group` first:

```sql
-- themes in one assignment group, biggest first
SELECT theme_group, COUNT(*) AS tickets
FROM redzone_consume.tcs_servicenow_analytics.ph05_output_ClusterThemes
WHERE assignment_group = 'ITSM.Field Career.T2'
  AND theme_group <> -1
GROUP BY theme_group
ORDER BY tickets DESC
```

Forgetting the `WHERE`/`GROUP BY` on `assignment_group` silently sums unrelated
themes that happen to share an id. That is the one way to misread this table.

## Column meanings — `ph05_output_ClusterThemes`
| Column | Meaning |
|---|---|
| `number` | the incident |
| `assignment_group` | the partition it was clustered within. NULL/blank source values become `Unknown` |
| `cluster` | raw HDBSCAN label within the group. **`-1` = noise**, not a cluster |
| `theme_group` | the cluster's theme after merging. `-1` stays `-1`. Several `cluster` values sharing a `theme_group` were merged as near-duplicates |
| `cluster_status` | `clustered` = the group ran through UMAP + HDBSCAN. `small_group_no_clustering` = the group was too small, so it stands alone: every row is noise by construction, not by failure |
| `summary_final` | the LLM summary that was actually embedded — read this, not `short_description`, when asking "why are these together?" |

## Reading a theme
A theme is a set of tickets whose summaries sit close together in embedding space.
It has no name. To label one, read the `summary_final` of its largest members:

```sql
SELECT number, summary_final
FROM redzone_consume.tcs_servicenow_analytics.ph05_output_ClusterThemes
WHERE assignment_group = 'ITSM.Field Career.T2' AND theme_group = 3
LIMIT 20
```

`ph05_output_ThemeOverlay` gives the same answer faster: one row per
(group, theme) with `incident_count` and, for each category column, the dominant
value and its share — e.g. `top_business_service = 'Email'`, `top_business_service_pct = 82.0`
means 82% of that theme's tickets are Email. A high share is a usable theme label;
a share near the noise floor means the theme cuts across services and needs the
summaries read.

## `-1` and `cluster_status` — what "no clusters" means
Two different situations both produce all `-1`, and they are **not** the same problem:

| What you see | What happened | What to do |
|---|---|---|
| `cluster_status = small_group_no_clustering` | the group had fewer than `min_cluster_rows` tickets (floor is raised to UMAP's `n_neighbors + 1`), so clustering was skipped deliberately | nothing is wrong — the group is too small to have themes. Wait for volume or lower `min_cluster_rows` **and** `umap_params.n_neighbors` together |
| `cluster_status = clustered` but every row `-1` | HDBSCAN ran and found no dense region | loosen `hdbscan_params.min_cluster_size` / `min_samples`, or accept that these tickets are genuinely unrelated |

A high but non-total noise share is normal for ticket text — HDBSCAN refuses to
force outliers into clusters. Judge it against `noise_pct` from previous runs, not
against zero.

## MLflow — what to look at after a run
| Metric | Read it as |
|---|---|
| `n_groups` | how many assignment groups were processed |
| `groups_clustered` / `groups_too_small` | how many actually clustered vs stood alone. A jump in `groups_too_small` means the input thinned out |
| `total_clusters` / `total_themes` | summed across groups. `total_themes` far below `total_clusters` means merging is aggressive — check `merge_threshold` |
| `noise_pct` | row-weighted across every group, so one tiny all-noise group cannot swing it |
| `silhouette` | row-weighted mean over the groups that produced one. Higher = tighter, better-separated clusters. Missing when no group produced 2+ clusters |
| `n_merges` | how many cluster pairs fused into themes |

`per_group_stats.json` is the per-group breakdown (rows, status, clusters, noise %,
silhouette, themes, merges) — the place to look when a rollup moves and you need to
know which group moved it. Per-group numbers are deliberately **not** metrics: one
key per group would grow the experiment every time a new assignment group appears.

`merge_log.json` lists `[group, cluster_a, cluster_b, cosine]` for every merge, so a
suspicious theme can be traced back to the pair that fused it.

The `clusters_2d.html` scatter colours points by `"<group> #<theme>"` (noise shown
as `noise`), because a bare theme id would paint unrelated themes from different
groups with the same colour.

---

## Guarantees this stage enforces
| Check | Fails how |
|---|---|
| both output tables raise on a failed write | the overlay write used to warn and continue, leaving a stale overlay beside a fresh cluster table |
| input comes from a live Delta table / SQL — no parquet fallback | a swallowed table error would silently cluster yesterday's file |
| input must have `text_col`, and not every row blank | a `KeyError` deep in a filter, or an empty list handed to the embedder |
| `group_col` must exist in the input when set | otherwise every ticket silently clusters in one global pool, which is what the partitioning exists to prevent |
| group keys are cast to trimmed strings, blanks → `Unknown` | `NaN`, `"NaN"` and `" X "` would split one assignment group into four |
| `min_cluster_rows` is raised to `n_neighbors + 1` | UMAP errors when `n_neighbors >= n_rows`, so a small group would crash the stage instead of standing alone |
| the scatter refuses a projection that does not match its frame | misaligned points draw a plot that looks perfectly fine |
| embedder loads from a Volume cache | it re-downloaded a few hundred MB on **every** run |

## Notes
- Test mode via the shared `run.limit`.
- UMAP/HDBSCAN/merge knobs live under `clustering:` in `../config.yml`.
- **One embed pass:** every ticket is encoded once, before the group loop. Encoding per
  group would re-batch (and on a cold cache re-load) the model for identical vectors.
- `group_col: null` turns partitioning off — one global run over every ticket, the
  pre-partition behaviour, and no `assignment_group` column in the overlay.
- **Model load:** `clustering.embed_model_volume_path` caches the embedder on a Volume.
  Shared with stage 03 via root-level `model_cache.load_cached`. Without the path set,
  the load warns and re-downloads every run.
- **Merging is transitive:** A~B and B~C fuse A, B and C even when A·C is below
  `merge_threshold`. A too-low threshold collapses unrelated themes into one giant group —
  compare `total_themes` against `total_clusters` before trusting a run.
- **The plot is sampled** (`plot_sample_size`, default 20000, seeded). It runs a *second*
  UMAP purely for the picture, and every point with hover text makes an artifact too big
  to open.
- **Timing:** per-step laps print a breakdown and reach MLflow as `secs_*`.

Config: `clustering:` section in the shared root `../config.yml`.
