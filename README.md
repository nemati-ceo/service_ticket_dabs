# service_ticket_dabs

DAB-only batch pipeline. 3 tasks (incident linking, clustering, dashboard prep)
write to UDP; PowerBI reads via SQL. No API, no UI, no serving.

## Fill placeholders in databricks.yml
GROUP_ID, ARTIFACT_ID, UTAN, OWNER_EMAIL, SECRET_SCOPE

## Commands
databricks bundle validate -t redzone
databricks bundle deploy   -t redzone
databricks bundle run service_ticket_pipeline -t redzone


service_ticket_dabs/
├── databricks.yml            ← batch 1 (8 placeholders to fill)
├── .gitlab-ci.yml            ← validate → deploy (GitLab, not GitHub)
├── .gitignore
├── README.md  CHANGELOG.md
└── src/main/databricks/
    ├── jobs/
    │   └── service_ticket.job.yml   ← 3 tasks, parallel + depends_on
    └── notebooks/service_ticket/
        ├── 00_test_smoke.py
        ├── 01_incident_linking.py   ← port PH05 + reranker
        ├── 02_clustering.py         ← port Approach B
        ├── 03_dashboard.py
        └── requirements.txt