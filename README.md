# service_ticket_dabs

DAB-only batch pipeline. 3 tasks (incident linking, clustering, dashboard prep)
write to UDP; PowerBI reads via SQL. No API, no UI, no serving.

## Connect to Databricks (one-time)

The CLI runs locally but everything executes in the redzone workspace.

```bash
# 1. Check CLI (need >= 0.218)
databricks --version

# 2. Auth to redzone (opens browser)
databricks auth login --host https://nml-udpr-redzone.cloud.databricks.com
# when prompted, profile name: nml-udpr-redzone

# 3. Pin the profile so you don't pass --profile every time
export DATABRICKS_CONFIG_PROFILE=nml-udpr-redzone

# 4. Confirm identity
databricks current-user me
```

## Run the pipeline

```bash
databricks bundle validate -t redzone
databricks bundle deploy   -t redzone
databricks bundle run service_ticket_pipeline -t redzone
```

## Notes
- Running as your own user on redzone (no service principal yet).
  Prod requires an SP — see commented blocks in databricks.yml.
- GROUP_ID / ARTIFACT_ID / utan are working defaults; confirm with Nancy and change.
- Notebooks 01/02/03 are placeholders that just print — prove plumbing first,
  then port real PH05 / Approach B code.
