"""pipeline.py — stage 02 orchestrator (LLM summarization), orchestration only."""

import time
from datetime import datetime

import summarize
import evaluate


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_summarization(spark, cfg):
    sc = cfg["summarization"]
    model = sc["model"]
    inp = sc["input_table"]
    foe = sc.get("fail_on_error", False)
    drop = sc.get("drop_deleted", True)

    t0 = time.perf_counter()
    print(f"[ph02] started {_ts()} | model={model} | input={inp}")

    p_changed, p_total = summarize.summarize_entity(
        spark, entity="problem", model=model,
        source_sql=(f"SELECT problem_id, any_value(combined_prob_desc) AS combined_prob_desc "
                    f"FROM {inp} WHERE problem_id IS NOT NULL GROUP BY problem_id"),
        key_col="problem_id", text_col="combined_prob_desc",
        summary_col="problem_summary", prompt_prefix=summarize.PROBLEM_PROMPT,
        out_table=sc["output_problem"], fail_on_error=foe, drop_deleted=drop)

    i_changed, i_total = summarize.summarize_entity(
        spark, entity="incident", model=model,
        source_sql=f"SELECT number, combined_cleaned_desc FROM {inp}",
        key_col="number", text_col="combined_cleaned_desc",
        summary_col="incident_summary", prompt_prefix=summarize.INCIDENT_PROMPT,
        out_table=sc["output_incident"], fail_on_error=foe, drop_deleted=drop)

    if sc.get("save_to_volume"):
        _save_to_volume(spark, sc)

    if sc.get("eval", {}).get("enabled"):
        try:
            evaluate.run(spark, cfg)
        except Exception as e:
            print(f"[ph02:eval] skipped ({e})")

    total = time.perf_counter() - t0
    print("=" * 60)
    print("Stage 02 complete!")
    print(f"  Problems:  {p_changed} summarized / {p_total} total")
    print(f"  Incidents: {i_changed} summarized / {i_total} total")
    print(f"  Total wall-clock: {total:.2f}s  (finished {_ts()})")
    print("=" * 60)
    return p_total, i_total


def _save_to_volume(spark, sc):
    base = sc["volume_base_path"]
    try:
        spark.table(sc["output_problem"]).toPandas().to_parquet(
            f"{base}/ProblemSummaries.parquet", index=False)
        spark.table(sc["output_incident"]).toPandas().to_parquet(
            f"{base}/IncidentSummaries.parquet", index=False)
        print(f"[ph02] summaries saved to volume: {base}")
    except Exception as e:
        print(f"[ph02] WARNING: could not save to volume ({e})")
