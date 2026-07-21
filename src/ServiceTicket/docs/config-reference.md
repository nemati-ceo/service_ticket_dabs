# config.yml — Reference

`src/ServiceTicket/config.yml` — single config driving the whole ProblemHealth pipeline (stages 00–05). One flat YAML, 19 top-level keys. Below: what each block does and the non-obvious gotchas learned while cleaning it.

## mode
- `production` — load GBM from `gbm_inference.model_path`, score, write linking table.
- `train` — fit new GBM (`gbm_train`), save `.pkl`, no linking-table write.
- Stages 00–03 identical in both modes: full run over whole snapshot, no incremental path.

## tables
Stage-01 input/output. Pipeline reads the CONSUME mirror (full copy from Stage 00), never `refine` directly.

## input_sync (Stage 00)
Full copy `refine -> consume` (overwrite). `refine` is READ-ONLY. Mirrors 3 tables; only `incidentstoopenproblem` (the labeled set) is read today. `problemzeroincidents` + `cluster` synced but not yet wired in.

## secrets
Databricks secret scope for external API tokens (HF, ServiceNow).
- **Gotcha:** `tcs-snow-test-cred` is the TEST scope — must swap for prod on deploy.
- `hf_endpoint` = NM Artifactory proxy (public huggingface.co firewalled in redzone). Blank = public hub (local dev only).

## source / servicenow
Stage-01 input source: `table` (reads `tables.input`) or `servicenow` (live REST via NM gateway). Gateway auth = `apikey` header (NOT `Authorization: Bearer`).

## model / volume / cleaning / script / keys
Bi-encoder (`all-MiniLM-L6-v2`, onnx) + Volume paths, spark cleaning engine, key columns (`number`, `problem_id`).

## pii_redaction (Stage 01b)
In-place span redaction → `<ENTITY>` placeholders, irreversible, no re-id map.
- spaCy model loaded FROM VOLUME, never downloaded (redzone firewalls downloads).
- **EVERY table whose text reaches the LLM must be listed** — `ai_query()` is the only egress point; an unredacted table = raw PII leak off-cluster. 3 tables listed: linked incidents, unlinked `cluster`, zero-incident problems.
- Custom regex recognizers: `USER_ID` (3 letters + 4 digits), `STREET_ADDRESS` (spaCy LOCATION misses street lines).
- **Gotcha:** `score_threshold: 0.35` MUST stay below 0.4 — Presidio scores PHONE_NUMBER at exactly 0.4, so 0.5 silently leaks every phone number while still redacting names/emails.

## run
`limit: null` = full run (no row cap). Integer only for quick test runs.

## mlflow
One parent run; each stage logs a nested child run (status/params/metrics/artifacts). Best-effort — failures never break a pipeline run. Run-as SP needs CAN_EDIT on the experiment folder.

## summarization (Stage 02)
LLM summary via `databricks-claude-opus-4-6`. **Input MUST be the redacted table** (stage 02 sends text off-cluster).
- Problem catalog = UNION of linked + zero-incident problems (else zero-incident problems never summarized → GBM can never link them). `UNION ALL` safe (dedupes by key downstream).

## reranking (Stage 03)
Cross-encoder `ms-marco-MiniLM-L-6-v2` reranks top-50 candidates.
- `eval.k_values: [5, 10]` — Top-K accuracy: fraction of incidents whose gold problem_id lands in the top-K reranked candidates.
- `eval.baselines` — frozen reference numbers (PH02, PH05) from prior stages, echoed for side-by-side comparison. Keys = K (1/5/7/10), values = accuracy. Not computed, just printed.
- **Mismatch:** baselines carry K=1 and 7 but `k_values` only computes 5 and 10 → no true head-to-head at Top-1/Top-7.

## gbm_inference (Stage 04, production)
GBM scores reranked candidates, writes Top-10 linking table.
- **incident_table MUST be redacted** — `linking.py` copies every column into output, so a raw source republishes unredacted text.
- **problem_sql MUST be a JOIN**, not bare `ph02_output_ProblemSummaries` — that table has NO `business_service`, so a bare table silently zeroes `bs_match` (GBM runs on 2 of 3 features, no error). BS lives on source as dotted col `problem_id.business_service`.
- **No `eval:` block** — Top-K match rate is train-mode monitoring (`gbm_train.eval.k_values`). Production incidents carry no gold `problem_id`, so scoring the labeled leftovers would report a rate that is not pipeline quality.

## gbm_train (Stage 04, train)
Fits new GBM, writes `.pkl`, no linking-table write.
- Holdout split BY INCIDENT (`group_col: number`) — row split leaks candidates of same incident across sides, inflating top-k.
- **`min_semantic_similarity: 0.35` is TRAIN ONLY** — drops weak (bad-label) links before fitting. NEVER in production: filtering on gold leaks the answer + skips the incidents that most need linking. Enforced by a test. `null` = no filter.

## clustering (Stage 05)
Clusters UNLINKED incidents (`cluster` table) — tickets with no open problem, the population themes should surface. (Previously read linked incidents = wrong set.)
- Gap-fill: unlinked incidents skip stage 02 (no summary), so summarizer reused (hash MERGE, no re-bill). Reads REDACTED unlinked table (`ai_query()` egress).
- `min_cluster_rows: 15` — skip clustering below this (too few; UMAP needs `n_neighbors < n_rows`); all marked noise.
- UMAP (5D for clustering, 2D for plot) → HDBSCAN → merge similar themes (`merge_threshold: 0.9`).
- `silhouette_sample_size: 5000` — silhouette is O(N²); subsample non-noise points. `null` = exact (small N only).
- Plot + metrics + params logged to MLflow (no Volume writes).

## Cross-cutting rules learned
1. **Redaction is the security boundary.** Any table feeding `ai_query()` (stage 02 summary, stage 05 gap-fill) must be the `ph01b_output_Redacted*` variant. Raw = PII off-cluster.
2. **Full snapshot every run.** Engineers deliver a complete DB snapshot each time; no incremental/delete logic. Closed problems vanish by absence.
3. **Train-only filters must never touch production** (`min_semantic_similarity`) — enforced by test.
4. **business_service must survive to the GBM** — needs the dotted-column JOIN, or `bs_match` silently zeroes.
