# ServiceTicket — Production Pipeline (Databricks DAB)

Batch-only pipeline on Databricks. Links ServiceNow incidents to their root-cause
problems, scores problem health, and clusters unlinked incidents into themes.
Outputs are Unity Catalog tables; PowerBI reads them via SQL.
**No API, no UI, no model serving.** PowerBI is the UI; UDP is the interface.

Repo: git.nmlv.nml.com/anlytcs/gen_ai/ServiceTicket
Workspace: redzone (https://nml-udpr-redzone.cloud.databricks.com)
GROUP_ID analyticsg . ARTIFACT_ID ServiceTicket . utan 84682

---

## 1. The pipeline

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  redzone_refine.servicenow_incidents_problems          READ-ONLY. 0 writes.   ║
╚══════════════════════════════════════════════════════════════════════════════╝
        │                          │                          │
  incidentstoopenproblem     cluster                  problemzeroincidents
  linked incidents           UNLINKED incidents       problems, 0 incidents
  + their problems           (+ close_notes)          (short_desc only)
        │                          │                          │
        └──────────────┬───────────┴──────────────┬───────────┘
                       ▼                          ▼
        ┌──────────────────────────────────────────────────────┐
        │  00  INPUT SYNC        full copy, overwrite, no diff  │
        │      3 refine tables ──► 3 consume mirrors            │
        └──────────────┬───────────────────────────────────────┘
                       │
        ┌──────────────▼───────────────┐
        │  01  CLEAN + EMBED           │   linked incidents only
        │      MiniLM · cosine · health│   → ph01_output_*
        └──────────────┬───────────────┘
                       │
   ╔═══════════════════▼══════════════════════════════════════════════════════╗
   ║  01b  PII REDACTION          Presidio + spaCy (from Volume, never DL'd)   ║
   ║  ──────────────────────────────────────────────────────────────────────  ║
   ║   PERSON · EMAIL_ADDRESS · PHONE_NUMBER · LOCATION · USER_ID · STREET_ADDRESS
   ║                                                                          ║
   ║   ph01_output_*      ──►  ph01b_output_Redacted            (linked)      ║
   ║   cluster_synced     ──►  ph01b_output_Redacted_Unlinked   (+close_notes)║
   ║   problemzero_synced ──►  ph01b_output_Redacted_ProblemsZero             ║
   ╚═══════════════════┬══════════════════════════════════════════════════════╝
                       │
  ═════════════════════╪═══ PII BOUNDARY — redacted text only below ══════════
                       │
        ┌──────────────▼───────────────────────────────────────┐
        │  02  LLM SUMMARIZATION            ⚠ EGRESS #1        │
        │      ai_query() → Databricks Claude                   │
        │      incidents (deduped by number)                    │
        │      problems  = linked ∪ zero-incident   ← UNION     │
        │      cache: text+prompt+model hash, NO re-billing     │
        └──────────────┬───────────────────────────────────────┘
                       │
        ┌──────────────▼───────────────┐
        │  03  CROSS-ENCODER RERANK    │  top-50 candidates/incident
        │      raw logits + sigmoid    │  all pairs, every run
        └──────────────┬───────────────┘
                       │
          ┌────────────┴─────────────┐
     mode: train                mode: production
          │                          │
 ┌────────▼─────────┐      ┌─────────▼──────────┐
 │ 04a  TRAIN GBM   │      │ 04b  SCORE GBM     │
 │ ──────────────── │ .pkl │ ──────────────────  │
 │ group-split by   │─────►│ load model          │
 │   incident       │Volume│ score all rows      │
 │ n_est=200 d=3    │      │ rank · link top-10  │
 │ lr=0.1 seed=42   │      │ → ph04_output_*     │
 │ train+test top-k │      └─────────┬──────────┘
 │ writes NO prod   │                │
 │   table          │                │
 └──────────────────┘                │
                       ┌─────────────▼─────────────────────────┐
                       │  05  CLUSTERING       ⚠ EGRESS #2     │
                       │      UNLINKED incidents               │
                       │      gap-fill → ai_query() (redacted) │
                       │      UMAP + HDBSCAN, seed=42          │
                       │      → ph05_output_*                  │
                       └───────────────────────────────────────┘
```

**Invariants.** Break one of these and the pipeline is wrong, usually silently:

| | |
|---|---|
| `refine` | read-only. Nothing here ever writes to it. |
| Every run | full reprocess. No incremental state, no partial runs, no skip-if-exists. |
| All writes | `redzone_consume.*` + Volumes. |
| All models | staged in a Volume. Nothing is downloaded at runtime (the redzone firewalls PyPI/HF/spaCy). |
| Egress | exactly two points — stage 02 and stage 05's gap-fill. Both read redacted tables only. |

---

## 2. Sources

The engineers drop a **complete snapshot** of each table into `refine` every run, so
stage 00 replaces each mirror wholesale. A closed problem disappears by being absent
from the new snapshot — there is no delete logic to get wrong.

| refine table | mirror | redacted as | consumed by |
|---|---|---|---|
| `incidentstoopenproblem` | `incidentstoopenproblem_synced` | `ph01b_output_Redacted` | 01 → 04 (labels + scoring) |
| `cluster` | `cluster_synced` | `ph01b_output_Redacted_Unlinked` | 05 (themes) |
| `problemzeroincidents` | `problemzeroincidents_synced` | `ph01b_output_Redacted_ProblemsZero` | 02 → 04 (extra candidates) |

`incidentstoopenproblem` carries **both** business-service columns: `business_service`
(the incident's own) and `problem_id.business_service` (the problem's). The GBM's
`bs_match` feature compares them. Note the natural key is the **pair**
`(number, problem_id)` — one incident can sit on several problems, so anything keyed on
`number` alone must dedupe first.

---

## 3. Train vs production

One flag in `config.yml`:

```yaml
mode: "production"    # or "train"
```

Stages 00–03 are **identical** in both modes — same data, same cleaning, same redaction,
same summaries, same rerank. Only stage 04 forks:

- **`train`** — fits a `GradientBoostingClassifier` on the labeled data
  (`n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42`), splits the
  holdout **by incident** (never by row — each incident produces ~50 candidate rows, and
  a row-wise split would leak), reports top-k on train *and* test, writes the `.pkl` to
  the Volume. **Does not touch the production linking table.**
- **`production`** — loads that `.pkl`, scores every candidate, writes the top-10 linking
  table.

Both branches build features through the same `build_features()`, so the columns the
model is fitted on cannot drift from the columns it is scored on.

The label is free: a candidate row is positive when `candidate_problem_id` equals the
incident's gold `problem_id` (`features.py`).

---

## 4. PII redaction (stage 01b)

Presidio + spaCy NER, run on-cluster. Matched spans are replaced in place with
`<ENTITY>` placeholders. **Irreversible — no re-identification map is kept.**

| Entity | Source |
|---|---|
| `PERSON`, `LOCATION` | spaCy NER (`en_core_web_lg`) |
| `EMAIL_ADDRESS`, `PHONE_NUMBER` | Presidio built-ins |
| `USER_ID` | custom regex, `[A-Za-z]{3}\d{4}` (e.g. `abc1234`) |
| `STREET_ADDRESS` | custom regex — spaCy's `LOCATION` tags "Milwaukee" but leaves "720 E Wisconsin Ave" in the clear |

**`score_threshold` must stay below 0.4.** Presidio scores a `PHONE_NUMBER` match at
exactly 0.4, so a threshold of 0.5 silently drops every phone number while still
redacting names and emails — the redaction looks like it works and quietly leaks phone
numbers. A test enforces this.

### One-time: stage the spaCy model into the Volume

Production **never downloads** the model — `spacy.load("en_core_web_lg")` resolves against
spacy.io, which the redzone firewalls, so the call hangs rather than failing fast. Run
once from somewhere with internet (or the Artifactory mirror):

```bash
python src/ServiceTicket/01b_pii_redaction/stage_model.py
```

Stage 01b then loads it by path and **fails loudly** if the path is missing.

The libraries (`presidio-analyzer`, `presidio-anonymizer`, `spacy`) are pip packages, not
Volume artifacts — they must resolve through the Artifactory mirror.

---

## 5. Run

```bash
databricks bundle validate -t redzone
databricks bundle deploy   -t redzone
databricks bundle run service_ticket_pipeline -t redzone
```

`run.py` is the single entrypoint and runs stages 00 → 01 → 01b → 02 → 03 → 04 → 05.
Individual stages are importable (`stage00(...)`, `stage01(...)`, …) for notebook use.

First run on new data: set `run.limit` to a few hundred rows. Stage 01b runs
`en_core_web_lg` over every ticket — it's a new full-dataset NLP pass.

```bash
cd src/ServiceTicket && python3 -m pytest tests/ -q
```

Tests that need Presidio/spaCy skip when they're absent; the config guards (the phone
threshold, the PII egress boundary, the dead-`bs_match` check) always run.

---

## 6. MLflow monitoring

One parent run per pipeline; each stage logs a **nested child run**. Best-effort:
tracking failures never break a run. Toggle in `config.yml` under `mlflow:`.

| What | Detail |
|---|---|
| Run structure | parent + nested `ph01..ph05` children (own status/duration) |
| Params + metrics | per stage: model, batch sizes, row counts, `wall_clock_s`, top-k |
| Redaction | `rows_redacted`, `tables_redacted` |
| Summarization | `*_no_content` — rows the LLM refused, which fell back to original text. A spike means summaries are silently degrading to raw ticket text. |
| Training | `train_*` and `test_*` top-k side by side — the gap is the overfitting signal |
| Data quality | stage 01: `input_rows`, `dup_key_pct`, `null_*_pct` |
| Eval tables | `topk_accuracy.json`; stage 05 `merge_log.json` |
| Failure capture | crashed stage = nested **FAILED** run + traceback |

GPU is used only by the sentence-transformer encoders (01/03/05) and the cross-encoder
(03). Stage 02's LLM is a remote endpoint; 04 is CPU.

---

## 7. Gotchas

**Cluster/theme IDs are not stable across runs.** Every run is a full refit, so
`cluster` / `theme_group` integers get reassigned — theme 3 today is not theme 3
tomorrow. Do not key a dashboard or a ticket field on them.

**Stage 02's cache is keyed on text + prompt + model.** Editing a prompt *does*
re-summarize the affected rows. (It is deliberately not keyed on text alone — that way a
prompt edit would silently change nothing and the old summaries would live forever.)

**`nltk` and the cross-encoder still fall back to a network download** if their data
isn't already cached on the cluster. Same firewall trap as the spaCy model. They work
today only because the cluster has them cached.

**Terraform download fails: "openpgp: key expired".** Known Databricks CLI bug in older
builds (incl. 0.240.0):
```bash
curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh
# or bypass with a local Terraform:
export DATABRICKS_TF_EXEC_PATH=$(which terraform)
```

**Multiple auth profiles matched.** `export DATABRICKS_CONFIG_PROFILE=nml-udpr-redzone`.

---

## 8. Open items

1. **Model versioning** — `train` writes `PH04_gradient_boosting_model.pkl`, the same path
   production reads. Training overwrites the live model with no rollback.
2. **Service principal for prod** — not provisioned. Running as a user on redzone works.
3. **Artifactory** — confirm the mirror carries `presidio-analyzer==2.2.363`,
   `presidio-anonymizer==2.2.363`, `spacy==3.8.2`.
4. **PII at rest** — redaction happens at 01b, so the consume mirrors and the ph01 tables
   still hold raw text. Nothing downstream reads them, but the PII is in `redzone_consume`.
   If policy forbids that, redaction moves up into stage 00.
5. **Zero-incident problems widen the haystack.** They are now rankable candidates, which
   adds no new gold links — so top-k will *drop* versus the old closed-world numbers. That
   is a more honest measurement, not a regression.
