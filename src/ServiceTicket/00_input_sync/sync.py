"""sync.py — Stage 00: copy the refine snapshots into consume mirrors.

The engineers drop a COMPLETE snapshot of each source table into refine every run, so
each mirror is simply replaced. Closed/removed records disappear on their own by being
absent from the new snapshot — there is no delete logic to get wrong.

refine is READ-ONLY. Nothing in this pipeline ever writes to it.
"""

from pyspark.sql import functions as F


def _copy_table(spark, source, target):
    """Full overwrite of one refine table into its consume mirror."""
    src = spark.table(source).withColumn("last_synced_at", F.current_timestamp())
    n = src.count()

    (src.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(target))

    print(f"[ph00]   {source}")
    print(f"[ph00]     -> {target}  ({n} rows)")
    return n


def _sources(sc):
    """Config accepts `tables: [{source, target}, ...]`, or a single source/target pair."""
    tables = sc.get("tables")
    if tables:
        return [(t["source"], t["target"]) for t in tables]
    if sc.get("source") and sc.get("target"):
        return [(sc["source"], sc["target"])]
    raise ValueError(
        "input_sync needs either `tables: [{source, target}, ...]` or a `source`/`target` pair")


def run_input_sync(spark, cfg):
    sc = cfg.get("input_sync") or {}
    if not sc.get("enabled", False):
        print("[ph00] input_sync disabled (input_sync.enabled=false) — skipping")
        return None

    pairs = _sources(sc)
    print(f"[ph00] full copy of {len(pairs)} refine table(s) -> consume")

    counts = {}
    for source, target in pairs:
        counts[target] = _copy_table(spark, source, target)

    total = sum(counts.values())
    print(f"[ph00] sync complete — {len(pairs)} table(s), {total} rows total")
    return counts
