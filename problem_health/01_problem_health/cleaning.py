"""cleaning.py — step 3 text cleaning."""

from preprocessing import (
    clean_text, clean_shortDescription_text,
    clean_description_text, removeGeneralProblemText,
)


def _clean_inc_short(s): return clean_text(clean_shortDescription_text(str(s)))
def _clean_inc_desc(s):  return clean_text(clean_description_text(str(s)))
def _clean_prob(s):      return removeGeneralProblemText(str(s))


def apply_cleaning(df):
    """Add the 6 cleaned/combined columns. Only called on rows to process."""
    df["cleaned_short_description"] = df["short_description"].fillna("").apply(_clean_inc_short)
    print("[3/8]   - short_description cleaned")
    df["cleaned_description"] = df["description"].fillna("").apply(_clean_inc_desc)
    print("[3/8]   - description cleaned")
    df["combined_cleaned_desc"] = df["cleaned_short_description"] + " " + df["cleaned_description"]
    df["cleaned_prob_short_desc"] = df["problem_id.short_description"].fillna("").apply(_clean_prob)
    print("[3/8]   - problem short_description cleaned")
    df["cleaned_problem_desc"] = df["problem_id.description"].fillna("").apply(_clean_prob)
    print("[3/8]   - problem description cleaned")
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
    """Dispatch cleaning by config: cleaning.engine = "spark" (distributed) or"""
    engine = cfg.get("cleaning", {}).get("engine", "pandas").lower()
    if engine == "spark":
        print("[3/8]   engine=spark (distributed pandas_udf)")
        try:
            return _clean_with_spark(spark, df_to_score)
        except Exception as e:
            print(f"[3/8]   WARNING: Spark cleaning failed ({e}); falling back to pandas.")
    else:
        print("[3/8]   engine=pandas (single-thread driver)")
    return apply_cleaning(df_to_score)
