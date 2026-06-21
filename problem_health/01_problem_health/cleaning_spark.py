"""
cleaning_spark.py — Spark-native (distributed) version of cleaning.apply_cleaning.

Produces the SAME 6 columns as cleaning.apply_cleaning, but the per-row text
cleaning runs in parallel across all executor cores via pandas_udf (vectorized
Arrow UDFs) instead of single-threaded pandas .apply() on the driver.

This module reuses the exact same element-level functions from preprocessing.py,
so the cleaning logic is identical — only the execution engine changes.

Usage:
    import cleaning_spark as cs
    sdf = spark.table(cfg["tables"]["input"])      # keep it as a Spark DataFrame
    sdf_clean = cs.apply_cleaning_spark(sdf)       # 6 cleaned cols added, distributed
"""

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import StringType

from preprocessing import (
    clean_text, clean_shortDescription_text,
    clean_description_text, removeGeneralProblemText,
)


# --- element-level cleaning (identical to pipeline.py) ---
def _clean_inc_short(s): return clean_text(clean_shortDescription_text(str(s)))
def _clean_inc_desc(s):  return clean_text(clean_description_text(str(s)))
def _clean_prob(s):      return removeGeneralProblemText(str(s))


# --- vectorized pandas UDFs: each runs on an Arrow batch, on an executor core ---
@pandas_udf(StringType())
def _udf_inc_short(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).map(_clean_inc_short)


@pandas_udf(StringType())
def _udf_inc_desc(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).map(_clean_inc_desc)


@pandas_udf(StringType())
def _udf_prob(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).map(_clean_prob)


def apply_cleaning_spark(sdf):
    """
    Add the 6 cleaned/combined columns to a Spark DataFrame (distributed).

    NOTE: the source problem columns are literally named "problem_id.short_description"
    and "problem_id.description" — the dot is part of the name, not nested-field
    access, so they MUST be back-tick quoted in Spark.
    """
    sdf = (
        sdf
        .withColumn("cleaned_short_description", _udf_inc_short(col("short_description")))
        .withColumn("cleaned_description",       _udf_inc_desc(col("description")))
        .withColumn("cleaned_prob_short_desc",   _udf_prob(col("`problem_id.short_description`")))
        .withColumn("cleaned_problem_desc",      _udf_prob(col("`problem_id.description`")))
    )
    # concat (not concat_ws) to match the original "a b" spacing exactly;
    # the UDFs always return non-null strings, so concat never yields null.
    sdf = (
        sdf
        .withColumn(
            "combined_cleaned_desc",
            F.concat(col("cleaned_short_description"), F.lit(" "), col("cleaned_description")),
        )
        .withColumn(
            "combined_prob_desc",
            F.concat(col("cleaned_prob_short_desc"), F.lit(" "), col("cleaned_problem_desc")),
        )
    )
    return sdf
