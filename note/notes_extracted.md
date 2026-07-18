```
STAGE 00 - Input Sync ==> SUCCESS

STAGE 01 - Problem Health 01 ==> SUCCESS

STAGE 01b - PII Redaction (before ANY text leaves the cluster) ==> STAGE 01b SUCCESS (check this )
```

1.
2026/07/16 05:08:54 INFO mlflow.tracking.fluent: Experiment with name '/Users/nancyhuang@northwesternmutual.com/ProblemHealth' does not exist. Creating a new experiment.

[mlflow] could not set experiment /Users/nancyhuang@northwesternmutual.com/ProblemHealth (PERMISSION_DENIED: User e0e59759-0fb9-4b21-916d-d857ffd33b73 does not have read permission for node with aclPath /workspace/4030492182474975/3758144595252413. Config: host=https://nvirginia.cloud.databricks.com, auth_type=runtime, retry_timeout_seconds=500)
[mlflow] could not start run ph01_problem_health (RESOURCE_DOES_NOT_EXIST: Could not find experiment with ID None.)

2.
Error in stage 2:
# STAGE 01b — PII Redaction (before ANY text leaves the cluster)

```
STAGE 02 - LLM Summarization ==> SUCCESS

STAGE 03 - Cross-encoder Reranking ==> SUCCESS
STAGE 04 - Gradient Boosting (production) ==> SUCCESS, rehceck

STAGE 05 - Clustering ==> => SUCCESS

cluster scatter logged to MLFlow ==> Success
```

3. Check if we need to redownalod for second time
4. Check if data safe and not del each time
5. Check repot output
6. Check deduce
7. Remove extra command and clean the code
8. New task from Nancy
9. Cosine sim 0.35
10. Make sure "=== Top-K Evaluation (Incident-level) ===" make sure this is monitoring only for train mode), Must show "Top K match rate". Must be ml flow metrics too.
11. Number of cluster ==> MLFlow
12. Percentage of outliers ==> MLFlow
13. How close cluster score

14.
[ph01b] redzone_consume.tcs_servicenow_analytics.problemzeroincidents_synced -> redzone_consume.tcs_servicenow_analytics.ph01b_output_Redacted_ProblemsZero
[ph01b] redacting 1 column(s) over 20 rows: ['short_description'], ==> must have "description"

15.
Later: entire universe, we need to update we will run clustering with assignment group, which assignment group (we need to group all incident by one by one, assitmnet group is a col in the database.
1. Output must be separate too, live table, we must remove all clustering result. And create new one. Keep 4 weeks schedule
2. If only -1 and 0 , skip top for this part . And if database is to small, we do not cluster and merging (skip merging ) but will be stand alone.

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

[3] Spark Jobs
df: pandas.core.frame.DataFrame = [number: object, short_description: object ... 16 more fields]
df_subset: pandas.core.frame.DataFrame = [number: object, short_description: object ... 16 more fields]

16. Name of assignment group. It must be string . Check for mat make sure we all good
```python
df["short_description_clean"] = df["short_description"].astype(str).apply(
    clean_shortDescription_text
)
df["description_clean"] = df["description"].astype(str).apply(
    clean_description_text
)
```

18. GPU , configurations Nancy must give me, 
