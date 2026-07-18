# Stage 00 — Input Sync

**File:** `src/ServiceTicket/00_input_sync/sync.py`

## What it does (short)
Takes the raw ServiceNow snapshot from `refine` and makes a fresh full copy into `consume` (the working area). Every later stage reads the copy, never `refine`.

- Full **overwrite** each run — no add/update/delete logic. New snapshot = whole truth.
- Stamps a `last_synced_at` timestamp on every row.
- Closed/removed records vanish by being absent from the new snapshot.
- `refine` = read-only source, never modified.
- 3 tables copied; only `incidentstoopenproblem_synced` is read by the pipeline today.

## Flow
```
                        run_input_sync(spark, cfg)
                                 |
                  cfg["input_sync"].enabled ?
                    |                      |
                  false                  true
                    |                      |
             print "skip"          _sources(sc)  ── read config pairs
             return None                   |
                              ┌────────────┴─────────────┐
                              │ tables: [{source,target}]│  (list form)
                              │        OR                │
                              │ single source/target pair│
                              └────────────┬─────────────┘
                                           |
                            loop each (source, target)
                                           |
                                   _copy_table(...)
                                           |
                  ┌────────────────────────┴────────────────────────┐
                  │  src = spark.table(source)                       │
                  │        + column last_synced_at = now()           │
                  │  n = src.count()                                 │
                  │  src.write.delta                                 │
                  │      .mode("overwrite")        <── full replace  │
                  │      .option("overwriteSchema","true")           │
                  │      .saveAsTable(target)                        │
                  └────────────────────────┬────────────────────────┘
                                           |
                              print rows, return n
                                           |
                        total = sum(counts), print "complete"


  REFINE (read-only)                 CONSUME (mirror, overwritten each run)
  ┌───────────────────────────┐      ┌──────────────────────────────────────┐
  │ incidentstoopenproblem     │ ──▶ │ incidentstoopenproblem_synced         │
  │ problemzeroincidents        │ ──▶ │ problemzeroincidents_synced           │
  │ cluster                     │ ──▶ │ cluster_synced                        │
  └───────────────────────────┘      └──────────────────────────────────────┘
        (source snapshot)               + last_synced_at timestamp column
```

## Functions
- `_copy_table(spark, source, target)` — full overwrite of one table + row count.
- `_sources(sc)` — parse config: `tables` list or single `source`/`target` pair, else raise.
- `run_input_sync(spark, cfg)` — entry: skip if disabled, loop pairs, sum rows.

## Cleanup verdict
Nothing to remove — tight already. Optional: add empty-snapshot guard (skip overwrite if `n == 0`) so a bad/empty snapshot can't wipe good data. Delta time-travel (`VERSION AS OF`) is the current recovery path.
