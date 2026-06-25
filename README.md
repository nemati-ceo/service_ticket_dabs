# ServiceTicket — Production Pipeline (Databricks DAB)

Batch-only pipeline on Databricks. Three tasks (incident->problem linking,
clustering, dashboard prep) write to UDP tables. PowerBI reads them via SQL.
**No API, no UI, no model serving.** PowerBI is the UI; UDP is the interface.

Repo: git.nmlv.nml.com/anlytcs/gen_ai/ServiceTicket
Workspace: redzone (https://nml-udpr-redzone.cloud.databricks.com)
GROUP_ID analyticsg . ARTIFACT_ID ServiceTicket . utan 84682

---

## 0. Status

| Thing | State |
|---|---|
| CLI auth to redzone | working (profile nml-udpr-redzone) |
| Hand-rolled scaffold | built, validates locally |
| Sanctioned nmlops_stack generation | NOT done yet -- required |
| Stack choice (dab vs model) | OPEN -- needs Nancy |
| Service principal (prod) | not provisioned |
| Artifactory pip access | unconfirmed |

Decision: regenerate with nmlops_stack, then port our code in.
The hand-rolled databricks.yml + the 3 pipelines' logic port over; the
CI file and folder layout get replaced by the generator.

---

## 1. Connect to Databricks (one-time)

The CLI runs locally; everything executes in the redzone workspace.

```bash
databricks --version          # need a Terraform-patched build (see Known Issues)
databricks auth login --host https://nml-udpr-redzone.cloud.databricks.com
#   profile name when prompted: nml-udpr-redzone
export DATABRICKS_CONFIG_PROFILE=nml-udpr-redzone
databricks current-user me    # should print your NM email
```

---

## 2. Sanctioned generation (NMLOPS Stack) -- the correct pattern

Per the internal End-to-End docs. Generate the skeleton, do NOT hand-write it.

```bash
# Prereq: pip must reach NM Artifactory (see Open Items)
pip install --force-reinstall nmlops_stack

nmlops-stack
#   NM email : alinemati@northwesternmutual.com
#   slug     : ServiceTicket
#   stack    : 2 (dab)   <-- pending Nancy; 3 (model) if classifier is registered

cd <generated>
git init
git remote add origin https://git.nmlv.nml.com/anlytcs/gen_ai/ServiceTicket.git
git checkout -b <branch>
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Generator collides with the existing ServiceTicket/ folder -- run it in a
temp dir, then move generated files into the real repo WITHOUT overwriting .git.

---

## 3. Generated structure (target shape)

```
ServiceTicket/
|-- databricks.yml                 # LOCAL deploy entrypoint (you edit this)
|-- pyproject.toml                 # metadata + deps  (pip install -e .[dev])
|-- .gitlab-ci.yml                 # generated CI (replaces our stub)
|-- .gico_config.yaml
|-- CHANGELOG.md  README.md  sonar-project.properties
`-- src/ServiceTicket/
    |-- cli.py                     # Click CLI; steps invoked as commands
    |-- config.py                  # table names live here
    |-- databricks.yml             # GITLAB CI entrypoint (do NOT edit)
    |-- steps/
    |   |-- incident_linking.py    # main()  <- was notebook 01
    |   |-- clustering.py          # main()  <- was notebook 02
    |   `-- dashboard.py           # main()  <- was notebook 03
    `-- assets/databricks/
        |-- _variables.yml         # targets/hosts (dataint/dataqa/redzone/prod)
        |-- _compute.yml           # cluster definitions
        |-- _permissions.yml
        `-- service_ticket_workflow.yml   # the job/tasks
```

Key facts from the docs:
- Steps are Python modules with a Click CLI, not notebooks.
- Two databricks.yml: root = local deploy; src/.../databricks.yml = CI.
- Workflow config is split: _variables / _compute / _permissions / <workflow>.

---

## 4. Port plan (hand-rolled -> generated)

| Our file | Goes to | Action |
|---|---|---|
| databricks.yml (redzone, run_as user, 84682) | root databricks.yml + _variables.yml | merge values |
| notebooks/01_incident_linking.py | steps/incident_linking.py | rewrite as main() (PH05 + reranker) |
| notebooks/02_clustering.py | steps/clustering.py | rewrite as main() (Approach B) |
| notebooks/03_dashboard.py | steps/dashboard.py | rewrite as main() |
| service_ticket.job.yml | assets/databricks/*_workflow.yml | reshape to generated format |
| .gitlab-ci.yml (my stub) | -- | DELETE, use generated |
| UDP table names | config.py | redzone_consume.model_governance.* |

---

## 5. Run

```bash
databricks bundle validate -t redzone
databricks bundle deploy   -t redzone
databricks bundle run service_ticket_workflow -t redzone
```

Then in Databricks: Workflows -> Accessible by me -> the job -> Run.
First run uses placeholder steps (just print) to prove plumbing, then
port real PH05 / Approach B logic.

---

## 5b. MLflow monitoring (what you get)

One parent run per pipeline; each stage logs a **nested child run**. Best-effort:
tracking failures never break a run. Toggle in `config.yml` under `mlflow:`
(`enabled` / `experiment` / `tracking_uri` / `log_system_metrics`).

| What | Detail |
|---|---|
| Run structure | parent `problem_health_pipeline` + nested `ph01..ph05` children (own status/duration) |
| Params + metrics | per stage: model, batch sizes, row counts, `wall_clock_s`, top-k accuracy |
| Per-step timings | stage 01: `step_*_s` breakdown of the 8 steps |
| Data quality | stage 01: `input_rows`, `dup_key_pct`, `null_*_pct` |
| Baselines + deltas | stage 03: measured top-k vs PH02/PH05 baselines |
| Eval tables | stage 03/04: `topk_accuracy.json`; stage 05: `merge_log.json`, `input.sql` |
| Artifacts | `config_snapshot.yaml`, cluster 2-D plot (stage 05) |
| Run tags | git commit/branch, cluster id, user, `run_mode` (test/full) |
| System metrics | CPU/GPU/memory (needs `psutil`; `pynvml` for GPU curves) |
| Failure capture | crashed stage = nested **FAILED** run + traceback |

GPU is used only by the sentence-transformer encoders (stages 01/03/05) and the
cross-encoder (03); all cosine math is CPU. Stage 02 LLM is a remote endpoint; 04 is CPU.

---

## 6. Known issues

Terraform download fails: "openpgp: key expired". Known Databricks CLI bug
in older builds (incl. 0.240.0). bundle deploy needs Terraform; validate
may not. Fix:
```bash
# upgrade CLI to a patched build (>= ~0.296)
curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh
# OR bypass the built-in download with a local Terraform:
brew install terraform
export DATABRICKS_TF_EXEC_PATH=$(which terraform)
```
If it still fails after upgrade, Zscaler is breaking the checksum -> use the
local-Terraform bypass, or open a network ticket.

Multiple auth profiles matched. Pass --profile nml-udpr-redzone or
export DATABRICKS_CONFIG_PROFILE=nml-udpr-redzone.

---

## 7. Open items (blockers)

1. Stack: dab vs model -- register the GB classifier in MLflow (->model) or
   batch-score only (->dab)? Nancy decides. Pick BEFORE generating.
2. Artifactory pip access -- does pip install nmlops_stack resolve on your
   laptop? If 404, configure pip for NM Artifactory first.
3. Service principal for prod -- not provisioned. Validate/run as user on
   redzone works without it; prod deploy is gated until it exists.
4. Secret scope -- name for Artifactory + service-credential secrets
   (commented in databricks.yml until known).
5. Job task type/binding -- confirm against a generated *_workflow.yml.

GitLab is the remote (not GitHub) -- matches the .gitlab-ci.yml and Nancy's mandate.