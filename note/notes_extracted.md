# Open items

Done items removed (2026-07-21 audit, again 2026-08-11).
Recover full original: `git show 2ae586e:note/notes_extracted.md`.

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

## 4. Catalog prefix is hardcoded to redzone (Confluence Backlog #13)

`config.yml` holds **45 hardcoded catalog references** — 42 x `redzone_consume`,
3 x `redzone_refine` — and no variable interpolation. `databricks.yml` defines four
targets (`dataint`/`dataqa`/`redzone`/`prod`) with an `ENVIRONMENT` variable, but
`config.yml` is plain YAML loaded by `run.py`, so DAB `${var.X}` never reaches it.

**`databricks bundle deploy -t prod` therefore succeeds and then writes everything to
`redzone_consume`.** No error anywhere. Confluence says the prod Service Principal and
the `prod_consume` volume are already provisioned and waiting.

Mechanism to pick: cluster env var (`CLUSTER_ENV_VARS` in `_compute.yml` already carries
`ENVIRONMENT`), a `run.py` argument, or a `catalog:` block in `config.yml`. The env var is
the least code and already plumbed.

---

## Reference-repo comparison (started 2026-08-11)

Comparing this checkout against the reference tree Ali is screenshotting. **9 of 90 files
checked.** 3 of the first 8 came back *ours is ahead* — the reference is an older snapshot,
so blanket-comparing everything is mostly noise.

Checked and clean: all four `assets/databricks/*.yml`, `stage_io.py`.
Checked, **ours ahead**: `model_cache.py` (stale-cache recovery), `config.yml` in two
places (see below).

Still worth checking:
- `config.yml` lines **101-144** and **197-240** — the only ranges never shown
- `run.py`
- `__init__.py` — **missing from this repo entirely** (zero `__init__.py` anywhere).
  `pyproject.toml`'s `[[tool.bumpversion.files]]` points at
  `src/ServiceTicket/__init__.py`, so `bump-my-version` has a dangling target. The wheel
  still builds (setuptools namespace discovery, 74 entries).

Two bugs found in the REFERENCE config that must not be copied back:
- it redacts only `short_description` on the zero-incident problems table, but
  `summarization.problem_source_sql` concatenates `description` too — **unredacted PII
  reaches `ai_query`**
- it redacts `close_notes` on the unlinked table; that column does not exist there
  (ours uses `resolution_notes`)

Also reference-only and dead: `clustering.embed_volume_path` (code reads
`embed_model_volume_path`, so stage 05 would re-download the embedder every run) and
`gbm_inference.volume_base_path` / `save_to_volume` (read by nothing).

### Decisions still open
- **`requirements/`** — deleted in `6f36f5b` (nothing installed from it; it had drifted to
  `scikit-learn>=1.5` against pyproject's `==1.4.2`). It exists in the reference tree.
  Revert with `git revert 6f36f5b` if the team wants it back — but generate it from
  `pyproject.toml`, do not hand-maintain it.
- **In-package `databricks.yml`** — the reference has one at the repo root AND at
  `src/ServiceTicket/`. We have only the root one. Recommend keeping it that way: two
  bundle definitions means one is dead, and the in-package copy ships inside the wheel.
- **`EXISTING_CLUSTER_ID`** — confirmed `0417-160658-0whkjf2o`, matches the reference.

---

## Top-10 sheet — review follow-ups (2026-08-11)

Shipped: `top_N_score`, `top_N_similarity`, `semantic_similarity` ->
`linked_problem_similarity`, and `linked_problem_similarity_summarized`.

Open:
- **Acronym glossary in the summarizer prompt** — deliberately NOT done. Jeremiah wants to
  see the summarization comparison first. Cost note for that decision: the prompt is part
  of the summary cache key (`md5(text + prompt + model)`), so any prompt edit re-summarizes
  the whole corpus. "FP" is also ambiguous (fingerprint vs false positive), so a glossary
  beats a bare "expand acronyms" instruction.
- **The sheet still shows one problem per incident.** `linking.py` does
  `drop_duplicates(subset=[number])`, so a multi-linked incident collapses to its first
  row. Pre-existing, but more visible now that two similarity columns sit side by side.
  Decide whether the sheet should be one row per link instead — it changes the shape, so
  it is Jeremiah's call.
- **`gbm_inference.reranker_col` is `rerank_score` (raw logit), not `rerank_score_sigmoid`.**
  Fine today: a tree model is invariant to a monotone transform, and train and inference
  read the same column. **Do not "fix" it by flipping the key** — scoring sigmoid values
  against thresholds learned in logit units is silently wrong with no error.

---

## Blocked

- **Data ingestion (Confluence Backlog #1)** — the ServiceNow Experience API does not fit.
  Waiting on Data Mesh hydration, then UDP ingestion (~2 sprints), then a pipeline to build
  the three raw inputs (`incidentstoopenproblem`, `problemzeroincidents`, `cluster`) — that
  last piece is ours. Status in `#servicenow-oda-dse-tcs`; Phil Burkland for progress;
  Asaf Board / JoAnne Schmitt / Eric Brooks initiate the UDP work.
- **GBM retrain (Confluence Backlog #7)** — `scikit-learn` is pinned `==1.4.2` purely
  because the staged `.pkl` was fit under it. Nothing else holds that pin.
- **Clustering re-evaluation (Confluence Backlog #9)** — needs ticket volume from other TCS
  teams (Andrew Jandron). See "cannot be validated" below.

---

## Cannot be validated on current data

`cluster` / `cluster_synced` hold 10 tickets in ONE assignment group
(`ITSM.Field Career.T2`). Every group-loop branch is covered by synthetic tests
instead. With 10 rows the real run takes the stand-alone path: below
`min_cluster_rows` (15, floored at UMAP `n_neighbors + 1` = 16), so all noise, no
merge, empty overlay. Re-check the knobs when real volume lands.
