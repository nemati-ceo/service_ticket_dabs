"""pipeline.py — stage 05 orchestrator (clustering -> theme grouping), orchestration only."""

import time

import numpy as np

import mlflow_utils as mu
from stage_io import load_frame
from timing import Timer, ts

import clustering as cl
import merge as mg
import overlay as ov
import visualize as vz


def run_clustering(spark, cfg):
    cc = cfg["clustering"]
    num_col = cc.get("number_col", "number")
    text_col = cc.get("text_col", "summary_final")
    cat_cols = cc.get("cat_cols", ["business_service", "u_service_feature"])

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
        ml.log_params(_cluster_params(cc, n_rows=len(df)))
        timer.lap("load")

        embeddings = cl.embed(df[text_col].astype(str).tolist(), cc["embed_model"],
                              batch_size=cc.get("embed_batch_size", 64))
        timer.lap(f"embed {len(df)} texts")

        # Edge case 1: too few tickets to cluster meaningfully. Below this many rows
        # HDBSCAN/UMAP produce noise (and UMAP with n_neighbors>=len would error), so
        # mark everything as noise and skip clustering entirely.
        min_rows = cc.get("min_cluster_rows", 15)
        if len(df) < min_rows:
            print(f"[ph05] only {len(df)} rows (< {min_rows}) — too few to cluster; all marked noise")
            labels, (n_clusters, n_noise, noise_pct, sil) = cl.small_sample_noise(len(df))
        else:
            emb5 = cl.reduce_umap(embeddings, cc["umap_params"])
            labels = cl.cluster_hdbscan(emb5, cc["hdbscan_params"])
            n_clusters, n_noise, noise_pct, sil = cl.cluster_stats(
                emb5, labels, sample_size=cc.get("silhouette_sample_size", 5000))
        df["cluster"] = labels
        timer.lap("cluster")

        # Edge case 2: with every point noise (all -1) or a single cluster there is
        # nothing to merge — resolve_themes skips the merge and maps each cluster to
        # itself (noise stays -1).
        if n_clusters < 2:
            print(f"[ph05] {n_clusters} cluster(s) found — skipping merge step")
        theme_map, merge_log, cluster_ids = mg.resolve_themes(
            embeddings, labels, n_clusters, cc.get("merge_threshold", 0.9))
        df["theme_group"] = df["cluster"].map(theme_map)
        n_themes = int(df[df.theme_group != -1].theme_group.nunique())
        print(f"[ph05] {len(cluster_ids)} clusters -> {n_themes} themes ({len(merge_log)} merges)")
        timer.lap("merge themes")

        ml.set_tags({"output_table": cc.get("output_table"),
                     "overlay_table": cc.get("overlay_table")})
        if cc.get("input_sql"):
            ml.log_text(cc["input_sql"], "input.sql")
        if merge_log:
            # which clusters merged into which theme, with the centroid cosine that did it
            try:
                ml.log_dict([[int(a), int(b), float(s)] for a, b, s in merge_log],
                            "merge_log.json")
            except Exception as e:
                print(f"[ph05] merge_log artifact skipped ({e})")

        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown").astype(str)
        overlay_df = ov.theme_overlay(df, [c for c in cat_cols if c in df.columns])

        _log_plot(ml, vz, df, embeddings, cc, num_col, text_col, cat_cols)
        print(f"[ph05] clusters -> live Delta table {cc.get('output_table')} ...")
        result, out_rows, overlay_rows = _save_tables(spark, df, overlay_df, cc)
        timer.lap("save")
        timer.summary()

        # wall_clock_s used to be computed AFTER this block closed, so it never reached
        # MLflow — stage 05 was the only stage with no duration recorded.
        total = time.perf_counter() - t0
        ml.log_metrics({"n_clusters": n_clusters, "n_noise": n_noise, "noise_pct": noise_pct,
                        "silhouette": sil, "n_themes": n_themes, "n_merges": len(merge_log),
                        "rows_clustered": len(df), "output_rows": out_rows,
                        "overlay_rows": overlay_rows, "wall_clock_s": total,
                        **mu.step_timings(timer.laps)})

    print("=" * 60)
    print("Stage 05 complete! Live Delta tables written:")
    print(f"  {cc.get('output_table')}  ({out_rows} rows)")
    print(f"  {cc.get('overlay_table')}  ({overlay_rows} rows)")
    print(f"  Rows: {len(df)} | clusters: {len(cluster_ids)} | themes: {n_themes}")
    print(f"  Total wall-clock: {total:.2f}s  (finished {ts()})")
    print("=" * 60)
    return result, overlay_df


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


def _cluster_params(cc, n_rows):
    """Build the clustering knobs dict (embed/merge + flattened umap/hdbscan params)."""
    params = {"embed_model": cc["embed_model"], "merge_threshold": cc.get("merge_threshold", 0.9),
              "n_rows": n_rows}
    params.update({f"umap_{k}": v for k, v in cc.get("umap_params", {}).items()})
    params.update({f"hdbscan_{k}": v for k, v in cc.get("hdbscan_params", {}).items()})
    return params


def _log_plot(ml, vz, df, embeddings, cc, num_col, text_col, cat_cols):
    """2-D cluster scatter, logged as html (+ png if kaleido is present).

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
        plot_df, plot_emb = df.iloc[idx], np.asarray(embeddings)[idx]
        hover = [c for c in [num_col, "short_description", text_col, *cat_cols, "cluster"] if c in df.columns]
        proj = vz.project_2d(plot_emb, cc.get("umap_2d_params", {"n_components": 2}))
        fig = vz.build_scatter(plot_df, proj, "theme_group", hover,
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
    that looks like a healthy run."""
    if not table:
        return 0
    try:
        (spark.createDataFrame(pdf).write.format("delta")
            .option("overwriteSchema", "true").mode("overwrite").saveAsTable(table))
        print(f"[ph05] saved {what} -> {table}  ({pdf.shape[0]} rows, {pdf.shape[1]} cols) "
              f"[live Delta table]")
        return len(pdf)
    except Exception as e:
        print(f"[ph05] ERROR saving {what} to {table}: {e}")
        raise
