"""cleaning_spark.py — Spark-native (distributed) version of cleaning.apply_cleaning."""

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import StringType

from preprocessing import (
    clean_text, clean_shortDescription_text,
    clean_description_text, removeGeneralProblemText,
)


def _clean_inc_short(s): return clean_text(clean_shortDescription_text(str(s)))
def _clean_inc_desc(s):  return clean_text(clean_description_text(str(s)))
def _clean_prob(s):      return removeGeneralProblemText(str(s))


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
    """Add the 6 cleaned/combined columns to a Spark DataFrame (distributed)."""
    sdf = (
        sdf
        .withColumn("cleaned_short_description", _udf_inc_short(col("short_description")))
        .withColumn("cleaned_description",       _udf_inc_desc(col("description")))
        .withColumn("cleaned_prob_short_desc",   _udf_prob(col("`problem_id.short_description`")))
        .withColumn("cleaned_problem_desc",      _udf_prob(col("`problem_id.description`")))
    )
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
