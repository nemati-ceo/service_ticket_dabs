# Open items

Done items removed (2026-07-21 audit). Recover full original: `git show HEAD:note/notes_extracted.md`.

---

## 1. Clustering per assignment group

Entire universe: run clustering grouped by `assignment_group` (column in the database), one group at a time.

1. Separate output — live table. Remove all clustering results, create new ones. Keep 4-week schedule.
2. If only `-1` and `0`, skip top for this part. If dataset too small, no clustering and no merging (skip merging) — stand alone.

```python
for file_name in file_names[3:]:
    df = data_preprocess(file_name)
    summary_df = LLM_Summary(df)
    df = merge_summary(df, summary_df)
    biz_services = df["business_service"].unique()
    for biz_service in biz_services:
        output_loc = f"{OUTPUT_PATH}{file_name}_{biz_service.replace('/', '_')}.csv"
        df_subset = df[df["business_service"] == biz_service].reset_index(drop=True)
        if len(df_subset) >= 10:
            df_subset, embeddings = clustering(df_subset)
            centroid_merge(df_subset, embeddings)
        else:
            df_subset["cluster"] = "Small Group, no clustering"
            df_subset[[
                "number",
                "short_description",
                "description",
                "summary_final",
                "business_service",
                "u_service_feature",
                "u_incident_type", "cluster"
            ]].to_csv(output_loc, index=False)
            print(f"\nExported: {output_loc}")
            print(f"{file_name} - {biz_service.replace('/', '_')} - Small Group, no Clustering")
```

## 2. Assignment group name must be string

Check format after item 1 lands.

```python
df["short_description_clean"] = df["short_description"].astype(str).apply(
    clean_shortDescription_text
)
df["description_clean"] = df["description"].astype(str).apply(
    clean_description_text
)
```

## 3. Report output

No report module exists — stdout summaries only. Decide what the report is.

## 4. Remove extra commands, clean the code

Refactor branch covers stages 01-05. Stage 00 and 01b not passed over yet.

---

## Blocked

- **GPU configuration** — Nancy must supply. `_compute.yml:13` is `c5ad.xlarge` (CPU, no GPU pool).
- **New task from Nancy** — placeholder.
