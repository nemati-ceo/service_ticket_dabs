"""storage.py — write helpers for the live Delta output tables."""


def save_incident_scores(spark, df_incidentscore, table):
    """Persist incident-level scores to the live Delta table."""
    save_delta(spark, df_incidentscore, table)


def save_delta(spark, pdf, table):
    try:
        (spark.createDataFrame(pdf)
            .write.format("delta")
            .option("overwriteSchema", "true")
            .mode("overwrite")
            .saveAsTable(table))
        print(f"  saved -> {table}  ({pdf.shape[0]} rows, {pdf.shape[1]} cols) [live Delta table]")
    except Exception as e:
        print(f"  ERROR saving to {table}: {e}")
        raise
