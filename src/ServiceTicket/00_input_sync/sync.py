"""sync.py — Stage 00: MERGE the full refine snapshot into a consume input table.

Nancy's design (Slack, 2026-07-11): the engineers drop **full snapshots** of the
incident tables into prod `refine`. We never mutate refine. Instead we MERGE each
new snapshot into a `consume` mirror that the pipeline reads. One MERGE handles all
three cases at once:

  1. row in consume, NOT in new snapshot   -> DELETE  (problem closed / removed)
  2. row in new snapshot, NOT in consume   -> INSERT  (new ticket)
  3. row in both, content changed          -> UPDATE  (re-summarize downstream)
  4. row in both, unchanged                -> no-op

"Closed" is inferred by ABSENCE from the new snapshot, not from a status column —
the source `incidentstoopenproblem` table only carries incidents on OPEN problems,
so a row that disappears means its problem closed. This is ONLY correct because the
snapshot is FULL (`WHEN NOT MATCHED BY SOURCE THEN DELETE` would wipe consume on an
incremental feed). Guard with `input_sync.hard_delete` — set false for incrementals.

A `row_hash` (content hash) + `last_synced_at` are stamped so downstream can look up
which rows are new/changed and (re)run the LLM summarization only for those.
"""

from pyspark.sql import functions as F


def _bt(col):
    """Backtick-quote a column name that may contain literal dots (ServiceNow schema
    has columns like `problem_id.short_description`)."""
    return "`%s`" % col


def run_input_sync(spark, cfg):
    """Full-snapshot MERGE: refine source -> consume mirror. Idempotent."""
    sc = cfg.get("input_sync") or {}
    if not sc.get("enabled", False):
        print("[ph00] input_sync disabled (input_sync.enabled=false) — skipping")
        return None

    source = sc["source"]                      # refine full-snapshot table
    target = sc["target"]                      # consume mirror the pipeline reads
    key_cols = sc.get("key_columns", ["number", "problem_id"])
    hash_cols = sc.get("hash_columns") or cfg["incremental"]["hash_columns"]
    hard_delete = sc.get("hard_delete", True)  # True only for FULL snapshots

    mode = "FULL SNAPSHOT" if hard_delete else "INCREMENTAL"
    print(f"[ph00] input sync: {source} -> {target}")
    print(f"[ph00] mode={mode} | key={key_cols} | hash over {len(hash_cols)} col(s) | hard_delete={hard_delete}")

    # Build the source with a content hash + sync timestamp. Backticks keep the
    # dotted ServiceNow column names intact.
    src = spark.table(source)
    hash_expr = F.md5(F.concat_ws("||", *[F.coalesce(F.col(_bt(c)).cast("string"), F.lit("")) for c in hash_cols]))
    src = src.withColumn("row_hash", hash_expr).withColumn("last_synced_at", F.current_timestamp())

    # First run: the mirror does not exist yet -> just materialize it.
    if not spark.catalog.tableExists(target):
        n0 = src.count()
        print(f"[ph00] target {target} absent — INITIAL LOAD of {n0} rows (all new)")
        src.write.format("delta").mode("overwrite").saveAsTable(target)
        print("[ph00] created.")
        return n0

    # Row-count snapshot before the MERGE so the delta is visible in logs.
    src_n = src.count()
    before = spark.table(target).count()
    print(f"[ph00] snapshot rows={src_n} | mirror before={before}")

    from delta.tables import DeltaTable
    tgt = DeltaTable.forName(spark, target)
    cond = " AND ".join(f"t.{_bt(k)} = s.{_bt(k)}" for k in key_cols)

    merge = (
        tgt.alias("t")
        .merge(src.alias("s"), cond)
        .whenMatchedUpdateAll(condition="t.row_hash <> s.row_hash")   # case 3: content changed
        .whenNotMatchedInsertAll()                                    # case 2: new ticket
    )
    if hard_delete:
        merge = merge.whenNotMatchedBySourceDelete()                 # case 1: gone from snapshot
    merge.execute()

    # Pull per-operation counts from the Delta MERGE we just ran (last history entry)
    # so the log shows exactly what changed: new rows, content updates, closures.
    ins = upd = dele = None
    try:
        m = tgt.history(1).select("operationMetrics").collect()[0][0] or {}
        ins = int(m.get("numTargetRowsInserted", 0))
        upd = int(m.get("numTargetRowsUpdated", 0))
        dele = int(m.get("numTargetRowsDeleted", 0))
        print(f"[ph00] MERGE metrics: inserted(new)={ins} updated(changed)={upd} deleted(closed)={dele}")
        if ins == 0 and upd == 0 and dele == 0:
            print("[ph00] NO CHANGES — snapshot identical to mirror (no new rows)")
    except Exception as e:
        print(f"[ph00] WARNING: could not read MERGE metrics ({e})")

    n = spark.table(target).count()
    print(f"[ph00] merge complete — mirror now {n} rows"
          + ("" if hard_delete else "  (hard_delete off: stale rows NOT removed)"))
    return n
