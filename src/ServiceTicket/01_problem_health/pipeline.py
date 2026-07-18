"""pipeline.py — ProblemHealth 01 main pipeline (6 steps), orchestration only.

Full run every time: every row in the snapshot is cleaned, embedded and scored.
The engineers deliver a complete snapshot each run, so there is nothing to diff.
"""

import os

import numpy as np
import pandas as pd

import embeddings as emb
import similarity as sim
from timing import Timer, ts
from cleaning import clean_text_step
from storage import save_incident_scores, save_delta, save_parquet


def _mlflow_utils():
    """Load the shared root-level mlflow_utils.py (best-effort logging helpers)."""
    import importlib.util
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "mlflow_utils", os.path.join(root, "mlflow_utils.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _data_quality(df, key):
    """Best-effort input data-quality metrics (row count, dup-key %, null-text %).

    Surfaces silent upstream rot — a spike in duplicate keys or empty descriptions
    shows up in MLflow before it quietly degrades scores. Never raises.
    """
    metrics = {}
    try:
        n = len(df)
        metrics["input_rows"] = n
        if n and key in df.columns:
            metrics["dup_key_pct"] = round(df[key].duplicated().sum() / n * 100, 3)
        for col in ("short_description", "description"):
            if n and col in df.columns:
                blank = df[col].isna() | (df[col].astype(str).str.strip() == "")
                metrics[f"null_{col}_pct"] = round(blank.sum() / n * 100, 3)
    except Exception as e:
        print(f"[ph01] data-quality metrics skipped ({e})")
    return metrics


def _load_input_table(spark, table_name):
    """Read the stage-01 input table into pandas, tolerant of the ServiceNow schema.

    The source table has columns whose names contain literal dots (e.g.
    `problem_id.u_jira_url`) and several VOID/NULL-typed columns (always-null Jira
    fields), plus Databricks internal edge columns. Arrow `toPandas()` cannot
    convert VOID columns and mis-resolves the dotted names, raising
    [INTERNAL_ERROR]. So we project only the real (non-void, non-internal) columns
    with backtick-quoted names and disable Arrow for this collect.
    """
    from pyspark.sql import functions as F
    sdf = spark.table(table_name)
    sel, restore, dropped = [], {}, []
    for f in sdf.schema.fields:
        tn = f.dataType.typeName().lower()
        if tn in ("void", "null") or f.name.startswith("_databricks_internal"):
            dropped.append(f.name)                        # unconvertible / RLS-internal
            continue
        # Alias the literal dots OUT for the collect: Arrow's schema resolver
        # mis-resolves dotted names (and the `problem_id` vs `problem_id.x`
        # collision) and raises INTERNAL_ERROR. We restore the names in pandas.
        safe = f.name.replace(".", "__")
        sel.append(F.col("`%s`" % f.name).alias(safe))
        restore[safe] = f.name
    if dropped:
        print(f"  skipping {len(dropped)} null/void/internal column(s): {dropped}")
    try:
        spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")
    except Exception:
        pass
    pdf = sdf.select(*sel).toPandas()
    pdf.columns = [restore.get(c, c) for c in pdf.columns]  # dotted names back for downstream code
    return pdf


def run_problem_health(spark, cfg):
    t = cfg["tables"]
    vol = cfg["volume"]
    base_path = vol["base_path"]
    key = cfg["keys"]["key_column"]
    limit = cfg.get("run", {}).get("limit")
    timer = Timer()

    # MLflow run wraps ALL the work: a crash in any step lands as a FAILED run with
    # its traceback, and the run duration covers the whole stage (best-effort, never
    # raises). Matches the stage-05 pattern.
    mu = _mlflow_utils()
    with mu.stage_run(cfg, "ph01_problem_health") as ml:
        ml.log_params({"embed_model": cfg["model"]["name"],
                       "backend": cfg["model"].get("backend", "onnx"),
                       "batch_size": cfg["model"].get("batch_size", 256),
                       "limit": limit})
        ml.set_tags({"input_table": t["input"], "output_table": t["output_incident"]})

        print("[1/6] Loading input data...")
        source_type = (cfg.get("source") or {}).get("type", "table")
        if source_type == "servicenow":
            import servicenow_source as snow
            print("  source: ServiceNow REST gateway")
            df_all = snow.fetch_incidents(cfg)
        else:
            df_all = _load_input_table(spark, t["input"])
        if limit:
            df_all = df_all.head(limit)
            print(f"  TEST MODE: limited to {len(df_all)} rows")
        print(f"[1/6] Done. {df_all.shape[0]} rows, {df_all.shape[1]} cols")
        timer.lap("[1/6] load")

        print("[2/6] Cleaning text...")
        df = clean_text_step(spark, df_all, cfg)
        print("[2/6] Done. Text cleaning complete.")
        timer.lap("[2/6] clean")

        print("[3/6] Loading model...")
        model = emb.load_or_save_model(
            cfg["model"]["name"], cfg["model"]["registry_name"],
            backend=cfg["model"].get("backend", "onnx"),
            volume_path=cfg["model"].get("volume_path"))
        timer.lap("[3/6] load model")

        bs = cfg["model"].get("batch_size", 256)
        print("[4/6] Encoding incident embeddings...")
        combined_embeddings = emb.encode_texts(model, df["combined_cleaned_desc"], bs)

        prob_key = cfg.get("aggregation", {}).get("problem_key", "problem_id")
        print(f"[4/6] Encoding problem embeddings (deduplicated by {prob_key})...")
        uniq = df.drop_duplicates(subset=[prob_key]).reset_index(drop=True)
        uniq_emb = emb.encode_texts(model, uniq["combined_prob_desc"], bs)
        pe_by_problem = pd.Series(list(uniq_emb), index=uniq[prob_key])
        problem_embeddings = np.vstack(df[prob_key].map(pe_by_problem).to_numpy())
        print(f"[4/6]   encoded {len(uniq)} unique problems for {len(df)} incidents")
        print(f"[4/6] Done. Encoded {len(combined_embeddings)} incident + "
              f"{len(uniq)} unique problem embeddings.")
        timer.lap("[4/6] encode")

        if vol.get("save_embeddings"):
            save_parquet(pd.DataFrame(combined_embeddings), base_path,
                         "combined_embeddings.parquet", "Incident embeddings")
            save_parquet(pd.DataFrame(problem_embeddings), base_path,
                         "problem_embeddings.parquet", "Problem embeddings")

        print("[5/6] Computing cosine similarity...")
        df_incidentscore = sim.add_similarity(df, combined_embeddings, problem_embeddings)
        timer.lap("[5/6] similarity")

        print("[5/6] Saving incident-level scores to Delta...")
        save_incident_scores(spark, df_incidentscore, t["output_incident"], vol, base_path)
        timer.lap("[5/6] save incidents")

        print("[6/6] Aggregating problem-level health scores...")
        problem_health = sim.aggregate_problem_health(df_incidentscore)
        save_delta(spark, problem_health, t["output_problem"])
        if vol.get("save_problem_health"):
            save_parquet(problem_health, base_path, "ProblemHealth.parquet", "Problem health")
        timer.lap("[6/6] problem health")

        total = timer.summary()

        ml.log_metrics({"incidents_scored": len(df_incidentscore),
                        "problems_scored": len(problem_health),
                        "wall_clock_s": total})
        ml.log_metrics(mu.step_timings(timer.laps))          # per-step duration breakdown
        ml.log_metrics(_data_quality(df_all, key))           # input null/duplicate rates

    print("=" * 60)
    print("Pipeline complete!")
    print(f"  Incidents scored: {len(df_incidentscore)}")
    print(f"  Problems scored:  {len(problem_health)}")
    print(f"  Total wall-clock: {total:.2f}s  (finished {ts()})")
    print("=" * 60)
    return df_incidentscore, problem_health
