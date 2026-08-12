# 00 — Input Sync

Takes the raw ServiceNow snapshot from `refine` and makes a fresh full copy into
`consume` (the working area). Every later stage reads the copy, never `refine`.

## Behaviour
- **Full overwrite each run** — no add/update/delete logic. The engineers drop a complete
  snapshot every time, so the new snapshot is the whole truth.
- Closed/removed records disappear by being **absent** from the new snapshot.
- Stamps `last_synced_at` on every row.
- `refine` is **READ-ONLY** — nothing in this pipeline writes to it.
- A failed copy **raises and stops the pipeline** (a bad mirror would poison every stage).

## Flow
```
run_input_sync(spark, cfg)
  ├─ input_sync.enabled? no ──► skip, return None
  ├─ _sources(sc)  ── config: `tables: [{source,target}]` OR a single source/target pair
  └─ for each pair: _copy_table
        spark.table(source) + last_synced_at
        .write.delta.mode("overwrite").option("overwriteSchema").saveAsTable(target)

  REFINE (read-only)              CONSUME (live Delta, overwritten each run)
  incidentstoopenproblem   ──►    incidentstoopenproblem_synced
  problemzeroincidents     ──►    problemzeroincidents_synced
  cluster                  ──►    cluster_synced
```

All three mirrors are read: `incidentstoopenproblem_synced` by stage 01, and
`cluster_synced` + `problemzeroincidents_synced` by stage 01b (which redacts them for
stages 02 and 05).

## MLflow (`ph00_input_sync`)
Best-effort, never breaks the sync: `rows_total`, `tables_synced`, `wall_clock_s`, and
`rows__<table>` per mirrored table.

## Note
There is no empty-snapshot guard: a bad/empty source snapshot would overwrite good data.
Delta time-travel (`VERSION AS OF`) is the recovery path.

Config: `input_sync:` in the shared root `../config.yml`.

Flow diagram: [`diagram.md`](diagram.md).
