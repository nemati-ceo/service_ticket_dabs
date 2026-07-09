# Problem-Health Pipeline — Data Requirements & Field Mapping

Maps what the pipeline reads from the database to the **ODA Sponsored ITSM**
dataset field names (see `ITSM_Incident_Data_Dictionary.md`,
`ITSM_Problem_Data_Dictionary.md`). Use this to build/refresh the input table
`redzone_consume.model_governance.ph01_incidents_to_open_problems`.

---

## What the pipeline does (5 stages)

| Stage | Name | Job | Output table |
|---|---|---|---|
| 01 | Embed + similarity | Clean incident text, embed (`all-MiniLM-L6-v2`), cosine vs problems | `ph01_output_IncidentScore_SemanticSimilarity` |
| 02 | LLM summarization | Claude summarizes each incident + problem | `ph02_output_IncidentSummaries`, `ph02_output_ProblemSummaries` |
| 03 | Cross-encoder rerank | Re-score top candidates (`ms-marco-MiniLM-L-6-v2`) | `ph03_output_RerankedScores` |
| 04 | Gradient-boost link | GBM picks top-10 problem candidates per incident | `ph04_output_Incident_Problem_Linking_Top10` |
| 05 | Clustering | UMAP + HDBSCAN theme discovery | `ph05_output_ClusterThemes`, `ph05_output_ThemeOverlay` |

Value: auto-link incidents to root-cause problems, surface recurring
themes / tech debt, rank top-10 candidates for analysts, eval@k logged to MLflow.

---

## Required input columns (raw source)

### Incident side
| Pipeline column | ODA field | Type | Required | Note |
|---|---|---|---|---|
| `number` | `inc_number` | string | YES | key column (`incremental.key_column`) |
| `short_description` | `inc_short_description` | string | YES | incident text (cleaned + hashed) |
| `description` | — | string | YES | **GAP — no long-description field in ODA incident dict** |
| `problem_id` | `prb_number` (on incident row) | string | YES | link to problem |
| create timestamp | `inc_opened_at` | datetime | — | record created |
| update timestamp | — | datetime | YES | **GAP — no updated-timestamp in ODA incident dict** |

### Problem side
| Pipeline column | ODA field | Type | Required | Note |
|---|---|---|---|---|
| `problem_id.short_description` | `prb_short_desc` | string | YES | problem statement |
| `problem_id.description` | `prb_desc` | string | YES | problem description |
| `problem_id.business_service` | `prb_business_service` | string | optional | category overlay (clustering) |
| `problem_id.u_service_feature` | `prb_service_feature` | string | optional | category overlay (clustering) |
| problem number | `prb_number` | string | YES | join key |
| create timestamp | `prb_created_date` | datetime | — | does not change |
| update timestamp | `prb_updated_date` | datetime | — | edited timestamp |

---

## Create / update data (incremental logic)

Config `incremental`:
- `key_column: number`
- `update_column: sys_updated_on`
- `hash_columns: [short_description, description, problem_id.short_description, problem_id.description]`

| Side | Create | Update | Status |
|---|---|---|---|
| Problem | `prb_created_date` | `prb_updated_date` | OK |
| Incident | `inc_opened_at` | — | **MISSING** |

---

## OPEN GAPS — decide before deploy

1. **Incident update timestamp.** Config expects `sys_updated_on`; ODA incident
   dict has none. Options:
   - pull raw `sys_updated_on` from source ServiceNow (best), OR
   - fall back to `inc_opened_at` (create-only, misses edits), OR
   - use `etl_load_datetime` (daily grain, coarse).

2. **Incident `description`.** ODA gives only `inc_short_description`. Either
   source the long description from raw ServiceNow, or drop `description` from
   `incremental.hash_columns` and the stage-01 text concat.

3. **Dataset choice.** ODA sponsored data = Power BI dataflow, **daily refresh**.
   Pipeline incremental logic assumes raw `sys_updated_on`. Confirm daily-refresh
   grain is acceptable, or point at the raw ServiceNow table.
