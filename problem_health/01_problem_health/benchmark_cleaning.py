"""
benchmark_cleaning.py — compare the pandas (driver) vs Spark (distributed)
text-cleaning paths on speed AND output correctness.

Run inside a Databricks notebook:
    import sys
    sys.path.insert(0, "/Workspace/.../TCS/script/problem_health/01_problem_health")
    import benchmark_cleaning as bm
    bm.run(N=20000)          # set N=None for the full dataset

It prints wall-clock for each path and verifies the 6 cleaned columns match
row-for-row on a sample, so you can trust the Spark version before switching.
"""

import time

from run import load_config, get_spark
import cleaning
import cleaning_spark as cs

CLEAN_COLS = [
    "cleaned_short_description", "cleaned_description", "combined_cleaned_desc",
    "cleaned_prob_short_desc", "cleaned_problem_desc", "combined_prob_desc",
]


def _time(label, fn):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    print(f"  [{label}] {dt:8.2f}s")
    return out, dt


def run(N=20000, config_path=None):
    cfg = load_config(config_path)
    spark = get_spark()
    table = cfg["tables"]["input"]

    # ---- pandas / driver path ----
    print(f"[pandas]  reading {'full' if N is None else N} rows -> driver...")
    pdf = spark.table(table)
    if N:
        pdf = pdf.limit(N)
    pdf = pdf.toPandas()
    print(f"[pandas]  {len(pdf)} rows; cleaning with pandas .apply()...")
    df_pandas, t_pandas = _time("pandas .apply", lambda: cleaning.apply_cleaning(pdf.copy()))

    # ---- Spark / distributed path ----
    print("[spark]   cleaning with pandas_udf (distributed)...")
    sdf = spark.table(table)
    if N:
        sdf = sdf.limit(N)
    sdf_clean = cs.apply_cleaning_spark(sdf)

    # force full computation WITHOUT collecting to driver (fair Spark timing).
    # run twice — the first action includes JVM/UDF warm-up.
    _time("spark warm-up", lambda: sdf_clean.write.format("noop").mode("overwrite").save())
    _, t_spark = _time("spark pandas_udf", lambda: sdf_clean.write.format("noop").mode("overwrite").save())

    # ---- correctness check on a sample ----
    print("[check]   comparing outputs on a sample...")
    df_spark = sdf_clean.limit(min(2000, N or 2000)).toPandas()
    # align both to the same rows by the key column for a fair comparison
    key = cfg["incremental"]["key_column"]
    a = df_pandas.set_index(key)[CLEAN_COLS].sort_index()
    b = df_spark.set_index(key)[CLEAN_COLS].sort_index()
    common = a.index.intersection(b.index)
    mismatches = {}
    for c in CLEAN_COLS:
        diff = (a.loc[common, c].fillna("") != b.loc[common, c].fillna("")).sum()
        if diff:
            mismatches[c] = int(diff)

    print("=" * 60)
    print(f"  pandas:  {t_pandas:8.2f}s")
    print(f"  spark:   {t_spark:8.2f}s")
    if t_spark > 0:
        print(f"  speedup: {t_pandas / t_spark:7.2f}x")
    print(f"  rows compared: {len(common)}")
    if mismatches:
        print(f"  ⚠ COLUMN MISMATCHES: {mismatches}")
    else:
        print("  ✓ outputs identical on sample")
    print("=" * 60)
    return {"t_pandas": t_pandas, "t_spark": t_spark, "mismatches": mismatches}


if __name__ == "__main__":
    run()
