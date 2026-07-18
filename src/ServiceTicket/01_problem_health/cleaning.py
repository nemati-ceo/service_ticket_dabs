"""cleaning.py — step 3 text cleaning."""

from preprocessing import (
    clean_inc_short as _clean_inc_short,
    clean_inc_desc as _clean_inc_desc,
    clean_prob as _clean_prob,
)


def apply_cleaning(df):
    """Add the 6 cleaned/combined columns. Only called on rows to process."""
    df["cleaned_short_description"] = df["short_description"].fillna("").apply(_clean_inc_short)
    print("[2/6]   - short_description cleaned")
    df["cleaned_description"] = df["description"].fillna("").apply(_clean_inc_desc)
    print("[2/6]   - description cleaned")
    df["combined_cleaned_desc"] = df["cleaned_short_description"] + " " + df["cleaned_description"]
    df["cleaned_prob_short_desc"] = df["problem_id.short_description"].fillna("").apply(_clean_prob)
    print("[2/6]   - problem short_description cleaned")
    df["cleaned_problem_desc"] = df["problem_id.description"].fillna("").apply(_clean_prob)
    print("[2/6]   - problem description cleaned")
    df["combined_prob_desc"] = df["cleaned_prob_short_desc"] + " " + df["cleaned_problem_desc"]
    return df


def _clean_with_spark(spark, df_to_score):
    """Distributed cleaning: pandas -> Spark DataFrame -> pandas_udf clean -> pandas."""
    import cleaning_spark as cs
    try:
        spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    except Exception:
        pass
    sdf = spark.createDataFrame(df_to_score)
    sdf_clean = cs.apply_cleaning_spark(sdf)
    return sdf_clean.toPandas()


def clean_text_step(spark, df_to_score, cfg):
    """Dispatch cleaning by config: cleaning.engine = "spark" (distributed pandas_udf)
    or "pandas" (single-thread driver). Spark failures fall back to pandas."""
    engine = cfg.get("cleaning", {}).get("engine", "pandas").lower()
    if engine == "spark":
        print("[2/6]   engine=spark (distributed pandas_udf)")
        try:
            return _clean_with_spark(spark, df_to_score)
        except Exception as e:
            print(f"[2/6]   WARNING: Spark cleaning failed ({e}); falling back to pandas.")
    else:
        print("[2/6]   engine=pandas (single-thread driver)")
    return apply_cleaning(df_to_score)
