"""pipeline.py — stage 05 orchestrator (clustering -> theme grouping), orchestration only.

Clustering is partitioned by `clustering.group_col` (assignment group): tickets are
embedded ONCE, then UMAP + HDBSCAN + theme merge run inside each group separately, so a
theme is always "theme N within group X" and never merges tickets across groups. Cluster
and theme ids restart per group — `(group, theme_group, cluster)` is the key. Leave
`group_col` unset for a single global run over every ticket.
"""

import time

import numpy as np
import pandas as pd

import mlflow_utils as mu
from stage_io import load_frame
from timing import Timer, ts

import clustering as cl
import merge as mg
import overlay as ov
import visualize as vz

CLUSTERED = "clustered"
STANDALONE = "small_group_no_clustering"


def run_clustering(spark, cfg):
    cc = cfg["clustering"]
    num_col = cc.get("number_col", "number")
    text_col = cc.get("text_col", "summary_final")
    cat_cols = cc.get("cat_cols", ["business_service", "u_service_feature"])
    group_col = cc.get("group_col")

    t0 = time.perf_counter()
    print(f"[ph05] started {ts()} | embed={cc['embed_model']}")

    # MLflow wraps ALL the work so a crash mid-clustering lands as a FAILED run.
    with mu.stage_run(cfg, "ph05_clustering") as ml:
        timer = Timer()
        if cc.get("summarize_gap"):
            _ensure_summaries(spark, cfg)
            timer.lap("ensure summaries")
        df = load_frame(spark, cc.get("input_sql"), cc.get("input_table"), what="tickets")
        limit = cfg.get("run", {}).get("limit")
        if limit:
            df = df.head(limit).reset_index(drop=True)
            print(f"[ph05] TEST MODE: limited to {len(df)} rows")
        if text_col not in df.columns:
            raise ValueError(f"input has no '{text_col}' column (got {list(df.columns)})")
        before = len(df)
        df = df[df[text_col].astype(str).str.strip() != ""].reset_index(drop=True)
        print(f"[ph05] {len(df)} tickets to cluster ({before - len(df)} dropped for blank {text_col})")
        if df.empty:
            raise ValueError(f"no tickets with a non-empty '{text_col}' — nothing to cluster")
        if group_col:
            if group_col not in df.columns:
                raise ValueError(f"group_col '{group_col}' not in input (got {list(df.columns)}) "
                                 f"— add it to clustering.input_sql, or unset group_col to "
                                 f"cluster everything in one pass")
            df[group_col] = _normalize_groups(df[group_col])
        groups = _split_groups(df, group_col)
        print(f"[ph05] {len(groups)} group(s) by {group_col or '(none — single global run)'}")
        ml.log_params(_cluster_params(cc, n_rows=len(df), n_groups=len(groups)))
        timer.lap("load")

        # ONE encode pass over every ticket. Encoding per group would re-batch (and on a
        # cold cache re-load) the model once per group for identical vectors.
        embeddings = np.asarray(cl.embed(df[text_col].astype(str).tolist(), cc["embed_model"],
                                         batch_size=cc.get("embed_batch_size", 64),
                                         volume_path=cc.get("embed_model_volume_path")))
        timer.lap(f"embed {len(df)} texts")

        parts, stats_rows, merge_log = [], [], []
        for name, pos in groups:
            part, stats, group_merges = _cluster_group(
                df.iloc[pos].reset_index(drop=True), embeddings[pos], cc, name)
            parts.append(part)
            stats_rows.append(stats)
            merge_log.extend([[name, int(a), int(b), float(s)] for a, b, s in group_merges])
        df = pd.concat(parts, ignore_index=True)
        # Embeddings were sliced per group; realign them to the concatenated frame or the
        # plot colours every point by another ticket's vector.
        embeddings = embeddings[np.concatenate([pos for _, pos in groups])]
        # One lap for the whole loop: reduce, cluster and merge all happen per group now,
        # so separate "cluster" / "merge themes" laps would each be a fraction of a group.
        timer.lap(f"cluster {len(groups)} group(s)")

        totals = _rollup(stats_rows)
        print(f"[ph05] {totals['total_clusters']} clusters -> {totals['total_themes']} themes "
              f"across {totals['n_groups']} group(s) ({len(merge_log)} merges)")

        ml.set_tags({"output_table": cc.get("output_table"),
                     "overlay_table": cc.get("overlay_table")})
        if cc.get("input_sql"):
            ml.log_text(cc["input_sql"], "input.sql")
        # Per-group breakdown as an artifact, not as metrics: one metric key per group
        # would grow the run every time a new assignment group appears.
        ml.log_dict(stats_rows, "per_group_stats.json")
        if merge_log:
            # which clusters merged into which theme, with the centroid cosine that did it
            try:
                ml.log_dict(merge_log, "merge_log.json")
            except Exception as e:
                print(f"[ph05] merge_log artifact skipped ({e})")

        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown").astype(str)
        overlay_df = _build_overlay(df, [c for c in cat_cols if c in df.columns], group_col)

        _log_plot(ml, vz, df, embeddings, cc, num_col, text_col, cat_cols, group_col)
        print(f"[ph05] clusters -> live Delta table {cc.get('output_table')} ...")
        result, out_rows, overlay_rows = _save_tables(spark, df, overlay_df, cc)
        timer.lap("save")
        timer.summary()

        # wall_clock_s used to be computed AFTER this block closed, so it never reached
        # MLflow — stage 05 was the only stage with no duration recorded.
        total = time.perf_counter() - t0
        ml.log_metrics({**totals, "n_merges": len(merge_log),
                        "rows_clustered": len(df), "output_rows": out_rows,
                        "overlay_rows": overlay_rows, "wall_clock_s": total,
                        **mu.step_timings(timer.laps)})

    print("=" * 60)
    print("Stage 05 complete! Live Delta tables written:")
    print(f"  {cc.get('output_table')}  ({out_rows} rows)")
    print(f"  {cc.get('overlay_table')}  ({overlay_rows} rows)")
    print(f"  Rows: {len(df)} | groups: {totals['n_groups']} "
          f"({totals['groups_too_small']} too small to cluster) | "
          f"clusters: {totals['total_clusters']} | themes: {totals['total_themes']}")
    print(f"  Total wall-clock: {total:.2f}s  (finished {ts()})")
    print("=" * 60)
    return result, overlay_df


# --- grouping -----------------------------------------------------------------

def _normalize_groups(series, unknown="Unknown"):
    """Force the group key to a clean string: NULLs and blanks become `Unknown`.

    The column arrives from Spark as object dtype and can hold NaN, ints, or padded
    text. Left alone, `NaN`, `"NaN"` and `" ITSM.X "` split one group into three, and a
    float group id would name a group `1.0`.
    """
    out = series.astype(str).str.strip()
    return out.mask(series.isna() | (out == "") | out.str.lower().isin(["nan", "none"]), unknown)


def _split_groups(df, group_col):
    """[(group_name, positions)] — one entry per group, or a single (None, all) pass.

    Sorted by name so the row order of the output table (and the plot's colour
    assignment) is the same on every run over the same data.
    """
    if not group_col:
        return [(None, np.arange(len(df)))]
    return sorted(((str(name), np.asarray(idx, dtype=int))
                   for name, idx in df.groupby(group_col, sort=True).indices.items()),
                  key=lambda g: g[0])


def _cluster_group(df, embeddings, cc, name):
    """Cluster ONE group end to end: reduce -> cluster -> merge themes.

    Groups below `min_cluster_rows` stand alone: every ticket is noise, no UMAP, no
    HDBSCAN, no merge. UMAP also needs more rows than `n_neighbors` (it errors otherwise),
    so the floor is raised to n_neighbors + 1 whenever the config sets it lower.
    """
    tag = f"[ph05:{name}]" if name is not None else "[ph05]"
    n_neighbors = (cc.get("umap_params") or {}).get("n_neighbors", 15)
    min_rows = max(cc.get("min_cluster_rows", 15), n_neighbors + 1)
    if len(df) < min_rows:
        print(f"{tag} {len(df)} rows (< {min_rows}) — too few to cluster; "
              f"stand-alone, all noise, no merge")
        labels, (n_clusters, n_noise, noise_pct, sil) = cl.small_sample_noise(len(df))
        status = STANDALONE
    else:
        emb5 = cl.reduce_umap(embeddings, cc["umap_params"])
        labels = cl.cluster_hdbscan(emb5, cc["hdbscan_params"])
        n_clusters, n_noise, noise_pct, sil = cl.cluster_stats(
            emb5, labels, sample_size=cc.get("silhouette_sample_size", 5000))
        status = CLUSTERED

    df = df.copy()
    df["cluster"] = labels
    df["cluster_status"] = status

    # With every point noise (all -1) or a single cluster there is nothing to merge —
    # resolve_themes maps each cluster to itself (noise stays -1).
    if n_clusters < 2:
        print(f"{tag} {n_clusters} cluster(s) found — skipping merge step")
    theme_map, merge_log, _cluster_ids = mg.resolve_themes(
        embeddings, labels, n_clusters, cc.get("merge_threshold", 0.9))
    df["theme_group"] = df["cluster"].map(theme_map)
    n_themes = int(df[df.theme_group != -1].theme_group.nunique())

    stats = {"group": name, "rows": len(df), "status": status,
             "n_clusters": n_clusters, "n_noise": n_noise,
             "noise_pct": round(float(noise_pct), 2),
             "silhouette": round(float(sil), 4) if sil is not None else None,
             "n_themes": n_themes, "n_merges": len(merge_log)}
    return df, stats, merge_log


def _rollup(stats_rows):
    """Run-level metrics from the per-group stats — stable keys as groups come and go."""
    rows = sum(s["rows"] for s in stats_rows) or 0
    noise = sum(s["n_noise"] for s in stats_rows)
    sils = [(s["silhouette"], s["rows"]) for s in stats_rows if s["silhouette"] is not None]
    return {
        "n_groups": len(stats_rows),
        "groups_clustered": sum(1 for s in stats_rows if s["status"] == CLUSTERED),
        "groups_too_small": sum(1 for s in stats_rows if s["status"] == STANDALONE),
        "total_clusters": sum(s["n_clusters"] for s in stats_rows),
        "total_themes": sum(s["n_themes"] for s in stats_rows),
        "n_noise": noise,
        # row-weighted, so one tiny all-noise group cannot drag the headline number
        "noise_pct": (noise / rows * 100) if rows else 0.0,
        "silhouette": (sum(v * w for v, w in sils) / sum(w for _, w in sils)) if sils else None,
    }


def _build_overlay(df, cat_cols, group_col):
    """Theme overlay, one block per group (theme ids only mean something within a group).

    A group with no themes (all noise, or too small to cluster) contributes NOTHING —
    its empty block is dropped before the concat rather than passed in. pandas deprecated
    inferring result dtypes from empty entries, so concatenating them would silently
    change the overlay's column types on a future release.
    """
    if not group_col:
        return ov.theme_overlay(df, cat_cols)
    blocks = []
    for name, sub in df.groupby(group_col, sort=True):
        block = ov.theme_overlay(sub, cat_cols)
        if block.empty:
            continue
        block.insert(0, group_col, name)
        blocks.append(block)
    if blocks:
        return pd.concat(blocks, ignore_index=True)
    # Every group all-noise: keep the schema (plus the group column) so the overwrite
    # still replaces last run's rows instead of failing on an unknown shape.
    empty = ov.theme_overlay(df.iloc[:0], cat_cols)
    empty.insert(0, group_col, pd.Series(dtype="object"))
    return empty


def _load_sibling(stage_dir, mod):
    """Import a module from a sibling stage folder (reuse code, never duplicate it)."""
    import importlib.util
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, stage_dir, mod + ".py")
    spec = importlib.util.spec_from_file_location(f"_ph_{stage_dir}_{mod}", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _ensure_summaries(spark, cfg):
    """Summarize the unlinked tickets we cluster, reusing stage-02's summarizer.

    Writes to stage 05's OWN table: sharing stage 02's output let its drop_deleted
    wipe these every run, re-billing the LLM for every unlinked ticket.
    """
    cc, sc = cfg["clustering"], cfg["summarization"]
    src = cc.get("summarize_source_sql")
    if not src:
        return
    out_table = cc.get("summarize_output_table") or sc["output_incident"]
    summarize = _load_sibling("02_llm_summarization", "summarize")
    changed, total, _fallback = summarize.summarize_entity(
        spark, entity="cluster_incident", model=sc["model"], source_sql=src,
        key_col="number", text_col=cc.get("summarize_text_col", "combined_cleaned_desc"),
        summary_col="incident_summary", prompt_prefix=summarize.INCIDENT_PROMPT,
        out_table=out_table, fail_on_error=sc.get("fail_on_error", False),
        drop_deleted=False)
    print(f"[ph05] summaries ensured -> live Delta table {out_table}: "
          f"{changed} new / {total} tickets ({total - changed} reused)")


def _cluster_params(cc, n_rows, n_groups):
    """Build the clustering knobs dict (embed/merge + flattened umap/hdbscan params)."""
    params = {"embed_model": cc["embed_model"], "merge_threshold": cc.get("merge_threshold", 0.9),
              "n_rows": n_rows, "n_groups": n_groups,
              "group_col": cc.get("group_col") or "(none)",
              "min_cluster_rows": cc.get("min_cluster_rows", 15)}
    params.update({f"umap_{k}": v for k, v in cc.get("umap_params", {}).items()})
    params.update({f"hdbscan_{k}": v for k, v in cc.get("hdbscan_params", {}).items()})
    return params


def _log_plot(ml, vz, df, embeddings, cc, num_col, text_col, cat_cols, group_col=None):
    """2-D cluster scatter, logged as html (+ png if kaleido is present).

    ONE figure for every group. Theme ids restart per group, so points are coloured by
    "<group> #<theme>" — colouring by the raw theme id would paint unrelated themes from
    different groups the same.

    Sampled: this runs a SECOND UMAP purely for the picture, and a scatter of every point
    with hover text makes an artifact too big to open. The sample is seeded, so the plot
    is reproducible run to run.
    """
    try:
        cap = cc.get("plot_sample_size", 20000)
        idx = np.arange(len(df))
        if cap and len(df) > cap:
            idx = np.sort(np.random.default_rng(42).choice(len(df), int(cap), replace=False))
            print(f"[ph05] plot sampled to {len(idx)} of {len(df)} points")
        plot_df, plot_emb = df.iloc[idx].copy(), np.asarray(embeddings)[idx]
        color_col = "theme_group"
        if group_col and group_col in plot_df.columns:
            color_col = "theme_label"
            plot_df[color_col] = np.where(
                plot_df["theme_group"] == -1, "noise",
                plot_df[group_col].astype(str) + " #" + plot_df["theme_group"].astype(str))
        hover = [c for c in [num_col, group_col, "short_description", text_col, *cat_cols,
                             "cluster", "cluster_status"] if c and c in plot_df.columns]
        proj = vz.project_2d(plot_emb, cc.get("umap_2d_params", {"n_components": 2}))
        fig = vz.build_scatter(plot_df, proj, color_col, hover,
                               title=cc.get("plot_title", "LLM-Summarized Clusters (merged themes)"))
        ml.log_figure(fig, "clusters_2d.html")
        ml.log_figure(fig, "clusters_2d.png")  # best-effort; logger warns if kaleido missing
        print("[ph05] cluster scatter logged to MLflow")
    except Exception as e:
        print(f"[ph05] WARNING: plot step skipped ({e})")


def _save_tables(spark, df, overlay_df, cc):
    """Persist cluster/theme assignments + overlay to live Delta tables."""
    export_cols = [c for c in cc.get("export_cols", []) if c in df.columns] or df.columns.tolist()
    result = df[export_cols]
    out_rows = _write(spark, result, cc.get("output_table"), "clusters")
    overlay_rows = _write(spark, overlay_df, cc.get("overlay_table"), "overlay")
    return result, out_rows, overlay_rows


def _write(spark, pdf, table, what):
    """Overwrite one live Delta table. Raises: a swallowed write leaves a stale table
    that looks like a healthy run.

    An EMPTY frame is still written (as an all-string schema): Spark cannot infer a
    schema from an empty pandas frame, and skipping the write would leave last run's
    rows in place looking current. This is the normal case for the overlay when every
    group is too small to cluster — all noise means no themes to summarize.
    """
    if not table:
        return 0
    try:
        sdf = (spark.createDataFrame([], schema=", ".join(f"`{c}` string" for c in pdf.columns))
               if pdf.empty else spark.createDataFrame(pdf))
        (sdf.write.format("delta")
            .option("overwriteSchema", "true").mode("overwrite").saveAsTable(table))
        print(f"[ph05] saved {what} -> {table}  ({pdf.shape[0]} rows, {pdf.shape[1]} cols) "
              f"[live Delta table]")
        return len(pdf)
    except Exception as e:
        print(f"[ph05] ERROR saving {what} to {table}: {e}")
        raise
