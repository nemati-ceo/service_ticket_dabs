"""pipeline.py — stage 05 orchestrator (clustering -> theme grouping), orchestration only."""

import time
from datetime import datetime

import pandas as pd

import clustering as cl
import merge as mg
import overlay as ov
import visualize as vz


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_clustering(spark, cfg):
    cc = cfg["clustering"]
    num_col = cc.get("number_col", "number")
    text_col = cc.get("text_col", "summary_final")
    cat_cols = cc.get("cat_cols", ["business_service", "u_service_feature"])

    t0 = time.perf_counter()
    print(f"[ph05] started {_ts()} | embed={cc['embed_model']}")

    mu = _mlflow_utils()
    with mu.stage_run(cfg, "ph05_clustering") as ml:
        if cc.get("summarize_gap"):
            _ensure_summaries(spark, cfg)
        df = _load_frame(spark, cc.get("input_sql"), cc.get("input_table"), cc.get("input_parquet"))
        limit = cfg.get("run", {}).get("limit")
        if limit:
            df = df.head(limit).reset_index(drop=True)
            print(f"[ph05] TEST MODE: limited to {len(df)} rows")
        df = df[df[text_col].astype(str).str.strip() != ""].reset_index(drop=True)
        ml.log_params(_cluster_params(cc, n_rows=len(df)))

        embeddings = cl.embed(df[text_col].astype(str).tolist(), cc["embed_model"],
                              batch_size=cc.get("embed_batch_size", 64))

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

        ml.log_metrics({"n_clusters": n_clusters, "n_noise": n_noise, "noise_pct": noise_pct,
                        "silhouette": sil, "n_themes": n_themes, "n_merges": len(merge_log)})
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
        result = _save_tables(spark, df, overlay_df, cc)

    total = time.perf_counter() - t0
    print("=" * 60)
    print("Stage 05 complete!")
    print(f"  Rows: {len(df)} | clusters: {len(cluster_ids)} | themes: {n_themes}")
    print(f"  Total wall-clock: {total:.2f}s  (finished {_ts()})")
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
    """Make sure every ticket we cluster has a summary, reusing stage-02's summarizer.

    Hash-keyed MERGE: already-summarized tickets are reused (no re-billing), only the
    gap is sent to the LLM, and `drop_deleted=False` so linking summaries are never
    removed. So clustering drops no tickets and duplicates no summary.
    """
    cc, sc = cfg["clustering"], cfg["summarization"]
    src = cc.get("summarize_source_sql")
    if not src:
        return
    summarize = _load_sibling("02_llm_summarization", "summarize")
    changed, total, _fallback = summarize.summarize_entity(
        spark, entity="cluster_incident", model=sc["model"], source_sql=src,
        key_col="number", text_col=cc.get("summarize_text_col", "combined_cleaned_desc"),
        summary_col="incident_summary", prompt_prefix=summarize.INCIDENT_PROMPT,
        out_table=sc["output_incident"], fail_on_error=sc.get("fail_on_error", False),
        drop_deleted=False)
    print(f"[ph05] summaries ensured: {changed} new / {total} tickets ({total - changed} reused)")


def _load_frame(spark, sql, table, parquet_path):
    """Load a frame from (in order) a Spark SQL query, a Delta table, or parquet."""
    if sql:
        return spark.sql(sql).toPandas()
    if table:
        try:
            return spark.table(table).toPandas()
        except Exception as e:
            print(f"  could not read table {table} ({e}); trying parquet...")
    if parquet_path:
        return pd.read_parquet(parquet_path)
    raise ValueError("no input source: set clustering.input_sql / input_table / input_parquet")


def _mlflow_utils():
    """Load the shared root-level mlflow_utils.py (best-effort logging helpers)."""
    import importlib.util
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "mlflow_utils", os.path.join(root, "mlflow_utils.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _cluster_params(cc, n_rows):
    """Build the clustering knobs dict (embed/merge + flattened umap/hdbscan params)."""
    params = {"embed_model": cc["embed_model"], "merge_threshold": cc.get("merge_threshold", 0.9),
              "n_rows": n_rows}
    params.update({f"umap_{k}": v for k, v in cc.get("umap_params", {}).items()})
    params.update({f"hdbscan_{k}": v for k, v in cc.get("hdbscan_params", {}).items()})
    return params


def _log_plot(ml, vz, df, embeddings, cc, num_col, text_col, cat_cols):
    """Build the 2-D cluster scatter and log it as html (+ png if kaleido is present)."""
    try:
        hover = [c for c in [num_col, "short_description", text_col, *cat_cols, "cluster"] if c in df.columns]
        proj = vz.project_2d(embeddings, cc.get("umap_2d_params", {"n_components": 2}))
        fig = vz.build_scatter(df, proj, "theme_group", hover,
                               title=cc.get("plot_title", "LLM-Summarized Clusters (merged themes)"))
        ml.log_figure(fig, "clusters_2d.html")
        ml.log_figure(fig, "clusters_2d.png")  # best-effort; logger warns if kaleido missing
        print("[ph05] cluster scatter logged to MLflow")
    except Exception as e:
        print(f"[ph05] WARNING: plot step skipped ({e})")


def _save_tables(spark, df, overlay_df, cc):
    """Persist cluster/theme assignments + overlay to UC Delta tables."""
    export_cols = [c for c in cc.get("export_cols", []) if c in df.columns] or df.columns.tolist()
    result = df[export_cols]
    out = cc.get("output_table")
    if out:
        try:
            (spark.createDataFrame(result).write.format("delta")
                .option("overwriteSchema", "true").mode("overwrite").saveAsTable(out))
            print(f"[ph05] saved -> {out} ({result.shape})")
        except Exception as e:
            print(f"[ph05] ERROR saving to {out}: {e}")
            raise
    overlay_out = cc.get("overlay_table")
    if overlay_out:
        try:
            (spark.createDataFrame(overlay_df).write.format("delta")
                .option("overwriteSchema", "true").mode("overwrite").saveAsTable(overlay_out))
            print(f"[ph05] saved -> {overlay_out} ({overlay_df.shape})")
        except Exception as e:
            print(f"[ph05] WARNING: could not save overlay ({e})")
    return result
