"""pipeline.py — stage 02 orchestrator (LLM summarization), orchestration only."""

import os
import time
from datetime import datetime

import summarize
import evaluate


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _avg_len(spark, table, col):
    """Mean summary length — a drop signals truncation/degradation. None on failure."""
    try:
        return float(spark.sql(f"SELECT AVG(LENGTH({col})) FROM {table}").collect()[0][0] or 0.0)
    except Exception as e:
        print(f"[ph02] avg length for {table}.{col} skipped ({e})")
        return None


def _pct(part, whole):
    return round(part / whole * 100, 2) if whole else 0.0


def _mlflow_utils():
    """Load the shared root-level mlflow_utils.py (best-effort logging helpers)."""
    import importlib.util
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "mlflow_utils", os.path.join(root, "mlflow_utils.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run_summarization(spark, cfg):
    sc = cfg["summarization"]
    model = sc["model"]
    inp = sc["input_table"]
    foe = sc.get("fail_on_error", False)
    drop = sc.get("drop_deleted", True)

    t0 = time.perf_counter()
    print(f"[ph02] started {_ts()} | model={model} | input={inp}")

    # MLflow run wraps ALL the work so a crash mid-summarization lands as a FAILED run.
    mu = _mlflow_utils()
    with mu.stage_run(cfg, "ph02_summarization") as ml:
        # Fingerprints identify WHICH prompt version produced a run's summaries — a prompt
        # edit silently changes output, and without this runs are indistinguishable.
        ml.log_params({"model": model, "input_table": inp,
                       "drop_deleted": drop, "limit": cfg.get("run", {}).get("limit"),
                       "problem_prompt_fingerprint":
                           summarize.prompt_fingerprint(summarize.PROBLEM_PROMPT, model),
                       "incident_prompt_fingerprint":
                           summarize.prompt_fingerprint(summarize.INCIDENT_PROMPT, model)})
        ml.set_tags({"output_incident": sc.get("output_incident"),
                     "output_problem": sc.get("output_problem")})

        # Problem catalog unions linked + zero-incident problems; without the union a
        # zero-incident problem is never summarized and the GBM can never link it.
        problem_sql = sc.get("problem_source_sql") or (
            f"SELECT problem_id, any_value(combined_prob_desc) AS combined_prob_desc "
            f"FROM {inp} WHERE problem_id IS NOT NULL GROUP BY problem_id")

        print(f"[ph02] summarizing problems -> live Delta table {sc['output_problem']} ...")
        p_changed, p_total, p_fallback = summarize.summarize_entity(
            spark, entity="problem", model=model,
            source_sql=problem_sql,
            key_col="problem_id", text_col="combined_prob_desc",
            summary_col="problem_summary", prompt_prefix=summarize.PROBLEM_PROMPT,
            out_table=sc["output_problem"], fail_on_error=foe, drop_deleted=drop)

        # Dedupe by number: the source key is (number, problem_id), so one incident
        # arrives once per problem — duplicates double-bill the LLM and break the MERGE.
        print(f"[ph02] summarizing incidents -> live Delta table {sc['output_incident']} ...")
        i_changed, i_total, i_fallback = summarize.summarize_entity(
            spark, entity="incident", model=model,
            source_sql=(f"SELECT number, any_value(combined_cleaned_desc) AS combined_cleaned_desc "
                        f"FROM {inp} WHERE number IS NOT NULL GROUP BY number"),
            key_col="number", text_col="combined_cleaned_desc",
            summary_col="incident_summary", prompt_prefix=summarize.INCIDENT_PROMPT,
            out_table=sc["output_incident"], fail_on_error=foe, drop_deleted=drop)

        acc = None
        if sc.get("eval", {}).get("enabled"):
            try:
                acc = evaluate.run(spark, cfg)
            except Exception as e:
                print(f"[ph02:eval] skipped ({e})")

        total = time.perf_counter() - t0
        p_rows = spark.table(sc["output_problem"]).count()
        i_rows = spark.table(sc["output_incident"]).count()
        p_len = _avg_len(spark, sc["output_problem"], "problem_summary")
        i_len = _avg_len(spark, sc["output_incident"], "incident_summary")

        ml.log_metrics({"problems_total": p_total, "problems_summarized": p_changed,
                        "incidents_total": i_total, "incidents_summarized": i_changed,
                        "problems_no_content": p_fallback,
                        "incidents_no_content": i_fallback,
                        "problem_rows_out": p_rows, "incident_rows_out": i_rows,
                        # cost: rows actually sent to the LLM this run
                        "llm_calls_total": p_changed + i_changed,
                        # cache: how much the hash-skip saved (0% = cache defeated)
                        "problems_cache_hit_pct": _pct(p_total - p_changed, p_total),
                        "incidents_cache_hit_pct": _pct(i_total - i_changed, i_total),
                        # quality: a spike means summaries degraded to raw ticket text
                        "problems_fallback_pct": _pct(p_fallback, p_changed),
                        "incidents_fallback_pct": _pct(i_fallback, i_changed),
                        "problem_summary_len_avg": p_len, "incident_summary_len_avg": i_len,
                        "topk_accuracy": acc, "wall_clock_s": total})

    print("=" * 60)
    print("Stage 02 complete! Live Delta tables written:")
    print(f"  {sc['output_problem']}  ({p_rows} rows)")
    print(f"  {sc['output_incident']}  ({i_rows} rows)")
    print(f"  Problems:  {p_changed} summarized / {p_total} total  "
          f"({p_fallback} NO_CONTENT -> original text, cache hit {_pct(p_total - p_changed, p_total)}%)")
    print(f"  Incidents: {i_changed} summarized / {i_total} total  "
          f"({i_fallback} NO_CONTENT -> original text, cache hit {_pct(i_total - i_changed, i_total)}%)")
    print(f"  LLM calls this run: {p_changed + i_changed}")
    print(f"  Total wall-clock: {total:.2f}s  (finished {_ts()})")
    print("=" * 60)
    return p_total, i_total
