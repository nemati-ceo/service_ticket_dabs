"""storage.py — read/write helpers for Delta tables and Volume files."""

import os


def save_parquet(df, base_path, filename, label):
    """Best-effort parquet dump to the Volume. Never raises."""
    try:
        os.makedirs(base_path, exist_ok=True)
        df.to_parquet(f"{base_path}/{filename}")
        print(f"  {label} saved to volume: {base_path}/{filename}")
    except Exception as e:
        print(f"  WARNING: could not save {label} to volume ({e})")


def save_incident_scores(spark, df_incidentscore, table, vol, base_path):
    """Persist incident-level scores to Delta (+ volume parquet if enabled)."""
    save_delta(spark, df_incidentscore, table)
    if vol.get("save_incident_scores"):
        save_parquet(df_incidentscore, base_path,
                     "IncidentScore_SemanticSimilarity.parquet", "Incident scores")


def save_delta(spark, pdf, table):
    try:
        (spark.createDataFrame(pdf)
            .write.format("delta")
            .option("overwriteSchema", "true")
            .mode("overwrite")
            .saveAsTable(table))
        print(f"  saved -> {table} ({pdf.shape})")
    except Exception as e:
        print(f"  ERROR saving to {table}: {e}")
        raise
