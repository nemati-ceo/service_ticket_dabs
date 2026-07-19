"""stage_io.py — shared stage input loading. Root-level, imported by every stage."""


def load_frame(spark, sql, table, what):
    """Load a frame from a Spark SQL query or a live Delta table.

    No parquet fallback: swallowing the table error and reading a stale file is how a
    run reports success on yesterday's data.
    """
    if sql:
        print(f"  loading {what} via SQL")
        return spark.sql(sql).toPandas()
    if table:
        print(f"  loading {what} from table {table}")
        return spark.table(table).toPandas()
    raise ValueError(f"no input source for {what}: set a sql or table in config")
