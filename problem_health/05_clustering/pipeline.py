"""pipeline.py — stage 05 orchestrator (clustering -> theme grouping), orchestration only."""

import os
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
    base = cc.get("volume_base_path")
    num_col = cc.get("number_col", "number")
    text_col = cc.get("text_col", "summary_final")
    cat_cols = cc.get("cat_cols", ["business_service", "u_service_feature", "u_incident_type"])

    t0 = time.perf_counter()
    print(f"[ph05] started {_ts()} | embed={cc['embed_model']}")

    df = _load_frame(spark, cc.get("input_sql"), cc.get("input_table"), cc.get("input_parquet"))
    limit = cfg.get("run", {}).get("limit")
    if limit:
        df = df.head(limit).reset_index(drop=True)
        print(f"[ph05] TEST MODE: limited to {len(df)} rows")
    df = df[df[text_col].astype(str).str.strip() != ""].reset_index(drop=True)

    embeddings = cl.embed(df[text_col].astype(str).tolist(), cc["embed_model"],
                          batch_size=cc.get("embed_batch_size", 64))
    emb5 = cl.reduce_umap(embeddings, cc["umap_params"])
    labels = cl.cluster_hdbscan(emb5, cc["hdbscan_params"])
    cl.cluster_stats(emb5, labels)
    df["cluster"] = labels

    centroids, cluster_ids = mg.cluster_centroids(embeddings, labels)
    theme_map, merge_log = mg.merge_clusters(centroids, cluster_ids, cc.get("merge_threshold", 0.9))
    df["theme_group"] = df["cluster"].map(theme_map)
    n_themes = df[df.theme_group != -1].theme_group.nunique()
    print(f"[ph05] {len(cluster_ids)} clusters -> {n_themes} themes ({len(merge_log)} merges)")

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
    overlay_df = ov.theme_overlay(df, [c for c in cat_cols if c in df.columns])

    if cc.get("save_to_volume") and base:
        hover = [c for c in [num_col, "short_description", text_col, *cat_cols, "cluster"] if c in df.columns]
        proj = vz.project_2d(embeddings, cc.get("umap_2d_params", {"n_components": 2}))
        vz.scatter_html(df, proj, "theme_group", hover, f"{base}/clusters_2d.html",
                        title=cc.get("plot_title", "LLM-Summarized Clusters (merged themes)"))

    _save(spark, df, overlay_df, cc, base)

    total = time.perf_counter() - t0
    print("=" * 60)
    print("Stage 05 complete!")
    print(f"  Rows: {len(df)} | clusters: {len(cluster_ids)} | themes: {n_themes}")
    print(f"  Total wall-clock: {total:.2f}s  (finished {_ts()})")
    print("=" * 60)
    return df, overlay_df


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


def _save(spark, df, overlay_df, cc, base):
    """Persist cluster/theme assignments + overlay to UC (+ Volume parquet). Never CSV."""
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
    if cc.get("save_to_volume") and base:
        try:
            os.makedirs(base, exist_ok=True)
            result.to_parquet(f"{base}/ClusterThemes.parquet", index=False)
            overlay_df.to_parquet(f"{base}/ThemeOverlay.parquet", index=False)
            print(f"[ph05] tables saved to volume: {base}")
        except Exception as e:
            print(f"[ph05] WARNING: could not save to volume ({e})")
