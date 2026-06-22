"""storage.py — read/write helpers for Delta tables and Volume files."""


def get_existing_scores(spark, table):
    try:
        existing = spark.table(table).toPandas()
        print(f"  Loaded {len(existing)} existing scored incidents.")
        return existing
    except Exception as e:
        msg = str(e).lower()
        if any(s in msg for s in (
            "table_or_view_not_found", "not found", "cannot be found",
            "does not exist", "no such table",
        )):
            print("  No existing scores table found. Will score all incidents.")
            return None
        print(f"  ERROR reading existing scores from {table}: {e}")
        raise


def save_incident_scores(spark, df_incidentscore, table, vol, base_path):
    """Persist incident-level scores to Delta (+ volume parquet if enabled)."""
    save_delta(spark, df_incidentscore, table)
    if vol.get("save_incident_scores"):
        try:
            df_incidentscore.to_parquet(f"{base_path}/IncidentScore_SemanticSimilarity.parquet")
            print(f"  Incident scores saved to volume: {base_path}")
        except Exception as e:
            print(f"  WARNING: could not save incident scores to volume ({e})")


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
