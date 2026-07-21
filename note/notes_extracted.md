# Open items

Done items removed (2026-07-21 audit). Recover full original: `git show 2ae586e:note/notes_extracted.md`.

---

## 1. Clustering schedule — every 4 weeks

Clustering per assignment group is done; its schedule is not. The job runs
`0 0 6 ? * MON,THU` (`assets/databricks/_workflow.yml:21`) — twice a week, all stages.

Quartz has no "every 4 weeks". Options:
- first Monday of the month (`0 0 6 ? * MON#1`) — 4-5 weeks apart
- keep the weekly trigger, gate stage 05 in code on a last-run timestamp
- split stage 05 into its own job with its own schedule

Decide which, and whether stage 05 stays a task inside the main pipeline job.

## 2. Report output

No report module exists — stdout summaries only. Decide what the report is.
(`docs/05-clustering.md` now has a "How to read the results" section — that may cover part of it.)

## 3. Remove extra commands, clean the code

Refactor branch covers stages 01-05. Stage 00 and 01b not passed over yet.

## 4. Stage 02 does not consume the redacted `description` (found 2026-07-21)

`ph01b_output_Redacted_ProblemsZero.description` is redacted now, but
`summarization.problem_source_sql` still selects `short_description` only
(`config.yml:137-139`). Concat both if zero-incident problem summaries should use it.

---

## Blocked

- **GPU configuration** — Nancy must supply. `_compute.yml:13` is `c5ad.xlarge` (CPU, no GPU pool).
- **New task from Nancy** — placeholder.

---

## Cannot be validated on current data

`cluster` / `cluster_synced` hold 10 tickets in ONE assignment group
(`ITSM.Field Career.T2`). Every group-loop branch is covered by synthetic tests
instead. With 10 rows the real run takes the stand-alone path: below
`min_cluster_rows` (15, floored at UMAP `n_neighbors + 1` = 16), so all noise, no
merge, empty overlay. Re-check the knobs when real volume lands.
