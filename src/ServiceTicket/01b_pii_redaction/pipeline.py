"""pipeline.py — stage 01b orchestrator (PII redaction), orchestration only.

Sits between stage 01 (clean + embed) and stage 02 (LLM summarization). Stage 02 is
the only stage that sends text off-cluster, so every text column it can reach must be
scrubbed here first. Downstream stages read the redacted table; the raw ph01 table is
never read again by the pipeline.

Full run every time — like every other stage, this reprocesses the whole snapshot.
"""

import os
import time
from datetime import datetime

import redact as rd


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _mlflow_utils():
    """Load the shared root-level mlflow_utils.py (best-effort logging helpers)."""
    import importlib.util
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "mlflow_utils", os.path.join(root, "mlflow_utils.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _redact_udf(pc):
    """A pandas_udf that scrubs one text column.

    The engines are built once per executor process and cached on the module, not once
    per row or per batch: loading en_core_web_lg costs seconds and hundreds of MB.
    """
    import pandas as pd
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import StringType

    model_path = pc["model_path"]
    spacy_model = pc.get("spacy_model", "en_core_web_lg")
    custom = dict(pc.get("custom_recognizers") or {})
    entities = list(pc["entities"])
    language = pc.get("language", "en")
    threshold = pc.get("score_threshold", 0.35)

    @pandas_udf(StringType())
    def _udf(col: pd.Series) -> pd.Series:
        global _ENGINES
        try:
            analyzer, anonymizer = _ENGINES
        except NameError:
            analyzer, anonymizer = rd.build_engines(
                model_path, spacy_model, custom, language)
            _ENGINES = (analyzer, anonymizer)
        return col.map(lambda t: rd.redact_text(
            analyzer, anonymizer, t, entities, language, threshold))

    return _udf


def _tables(pc):
    """Config accepts `tables: [{input_table, output_table, text_columns}, ...]`,
    or a single input_table/output_table/text_columns triple."""
    tables = pc.get("tables")
    if tables:
        return list(tables)
    if pc.get("input_table") and pc.get("output_table"):
        return [{"input_table": pc["input_table"],
                 "output_table": pc["output_table"],
                 "text_columns": pc["text_columns"]}]
    raise ValueError(
        "pii_redaction needs `tables: [{input_table, output_table, text_columns}, ...]`")


def _redact_table(spark, udf, spec, entities):
    """Scrub one table's text columns and overwrite its redacted mirror."""
    src_table = spec["input_table"]
    out_table = spec["output_table"]
    text_cols = list(spec["text_columns"])

    sdf = spark.table(src_table)
    present = [c for c in text_cols if c in sdf.columns]
    missing = [c for c in text_cols if c not in sdf.columns]
    if missing:
        print(f"[ph01b] WARNING: text_columns not in {src_table}, skipping: {missing}")
    if not present:
        raise ValueError(
            f"none of text_columns={text_cols} exist in {src_table} — nothing to redact")

    n = sdf.count()
    print(f"[ph01b] {src_table} -> {out_table}")
    print(f"[ph01b]   redacting {len(present)} column(s) over {n} rows: {present}")

    for c in present:
        sdf = sdf.withColumn(c, udf(sdf[c].cast("string")))

    (sdf.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(out_table))
    print(f"[ph01b]   written ({n} rows)")
    return n, present


def run_pii_redaction(spark, cfg):
    pc = cfg.get("pii_redaction") or {}
    if not pc.get("enabled", False):
        print("[ph01b] pii_redaction disabled (pii_redaction.enabled=false) — skipping")
        return None

    specs = _tables(pc)
    entities = list(pc["entities"])

    t0 = time.perf_counter()
    print(f"[ph01b] started {_ts()}")
    print(f"[ph01b] redacting {len(specs)} table(s) | entities={entities}")
    print(f"[ph01b] spaCy model from Volume: {pc['model_path']}")

    # Ship redact.py to the executors. The pandas_udf below closes over this module,
    # and cloudpickle serializes it BY REFERENCE — the worker does `import redact`,
    # which fails with ModuleNotFoundError because the driver's sys.path entry for
    # this stage folder does not propagate to Python workers.
    spark.sparkContext.addPyFile(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              "redact.py"))

    udf = _redact_udf(pc)

    # MLflow run wraps the redaction work: this stage is the PII boundary, so a crash
    # mid-redaction must land as a FAILED run (best-effort, never raises itself).
    mu = _mlflow_utils()
    with mu.stage_run(cfg, "ph01b_pii_redaction") as ml:
        ml.log_params({"spacy_model": pc.get("spacy_model"),
                       "model_path": pc["model_path"],
                       "entities": ",".join(entities),
                       "score_threshold": pc.get("score_threshold", 0.35)})

        counts = {}
        for spec in specs:
            n, present = _redact_table(spark, udf, spec, entities)
            counts[spec["output_table"]] = n

        total_rows = sum(counts.values())
        total = time.perf_counter() - t0

        ml.set_tags({"output_tables": ",".join(counts)})
        ml.log_metrics({"rows_redacted": total_rows,
                        "tables_redacted": len(specs),
                        "wall_clock_s": total})

    print("=" * 60)
    print("Stage 01b complete!")
    for table, n in counts.items():
        print(f"  {n} rows -> {table}")
    print(f"  Total wall-clock: {total:.2f}s  (finished {_ts()})")
    print("=" * 60)
    return counts
