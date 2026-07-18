# Pipeline Parameters

Every numeric knob in `config.yml`, grouped by stage. Values shown are the
current defaults. Note that **`top_k` / `top_n`** appear in several places with
**different meanings** — see the note at the bottom.

Legend: 🎯 = changes results (quality) · ⚙️ = speed/memory only (no effect on output)

## Shared
| Param | Value | Meaning | |
|---|---|---|---|
| `run.limit` | 250 | Test row cap — # incidents stages 01 & 05 process (`null` = full dataset) | 🎯 |

## Stage 01 — Problem Health
| Param | Value | Meaning | |
|---|---|---|---|
| `model.batch_size` | 256 | Rows per embedding-encode batch | ⚙️ |

## Stage 03 — Cross-encoder rerank
| Param | Value | Meaning | |
|---|---|---|---|
| `reranking.top_k` | 50 | **Candidate problems shortlisted per incident**, then reranked. ⬆️ = better recall, slower | 🎯 |
| `reranking.limit` | 250 | Test row cap for stage 03 | 🎯 |
| `max_length` | 512 | Max tokens per (incident, problem) pair the cross-encoder reads | 🎯 |
| `chunk_size` | 5000 | Pairs buffered before each flush | ⚙️ |
| `batch_size` | 128 | Pairs per cross-encoder `predict` batch | ⚙️ |
| `bi_encoder_batch_size` | 64 | Batch for the shortlist embedder | ⚙️ |
| `candidate_chunk_size` | 1000 | Incidents per chunk in the top-K shortlist matmul | ⚙️ |
| `eval.k_values` | [5, 10] | Which Top-K hit-rates to report | — |

## Stage 04 — GBM inference
| Param | Value | Meaning | |
|---|---|---|---|
| `gbm_inference.top_n` | 10 | **# linked problems written per incident** (output width: `top_1..top_10`) | 🎯 |
| `batch_size` | 500000 | Rows per `predict_proba` batch | ⚙️ |
| `eval.k_values` | [1, 5, 7, 10] | Top-K accuracies to report | — |

## Stage 05 — Clustering
| Param | Value | Meaning | |
|---|---|---|---|
| `merge_threshold` | 0.9 | Centroid-cosine cutoff to merge clusters into a theme. ⬆️ = fewer merges / more themes | 🎯 |
| `embed_batch_size` | 64 | Encode batch | ⚙️ |
| `umap_params.n_neighbors` | 15 | UMAP locality for the 5-D cluster space | 🎯 |
| `umap_params.n_components` | 5 | Dimensions HDBSCAN clusters in | 🎯 |
| `hdbscan.min_cluster_size` | 8 | Smallest allowed cluster (⬆️ = fewer, bigger clusters) | 🎯 |
| `hdbscan.min_samples` | 3 | Density conservativeness (⬆️ = more points become noise) | 🎯 |
| `hdbscan.cluster_selection_epsilon` | 0.0 | Distance below which clusters are merged | 🎯 |
| `umap_2d_params.n_neighbors` | 30 | UMAP for the 2-D **plot only** | ⚙️ |
| `umap_2d_params.min_dist` | 0.1 | Point spread in the plot | ⚙️ |
| `random_state` (both UMAP) | 42 | Reproducibility seed | — |

## Stage 02 — Summarization eval (currently OFF)
| Param | Value | Meaning | |
|---|---|---|---|
| `eval.sample_size` | 5000 | Incidents sampled for the offline metric | 🎯 |
| `eval.top_k` | 10 | Top-K for that metric | — |

## ⚠️ The several "top_k"/"top_n" are different things
- `reranking.top_k` = **50** — how many candidates to *rerank* per incident.
- `gbm_inference.top_n` = **10** — how many problems to *save* per incident.
- `*.eval.k_values` — which accuracy *cutoffs* to print (reporting only).
- `summarization.eval.top_k` — Top-K for the (off-by-default) summary metric.

The two knobs that most change outputs: **`reranking.top_k`** (recall) and
**`clustering.merge_threshold`** (theme granularity). All `*batch_size` /
`chunk_size` values are pure speed/memory and do not change results.
