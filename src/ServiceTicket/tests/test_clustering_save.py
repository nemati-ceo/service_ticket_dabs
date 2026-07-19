"""Stage 05 pipeline — live-table writes and row counts.

Both outputs (clusters + overlay) must reach live Delta tables, and both counts must be
reported: they feed the MLflow metrics and the run summary. The overlay write used to
swallow its error with a WARNING, which left a stale overlay table looking healthy.
"""

import importlib.util
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE05 = os.path.join(ROOT, "05_clustering")
sys.path.insert(0, STAGE05)
sys.path.append(ROOT)

spec = importlib.util.spec_from_file_location("ph05_pipeline",
                                              os.path.join(STAGE05, "pipeline.py"))
p5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p5)


class _Writer:
    def __init__(self, sink, fail_on=None):
        self.sink, self.fail_on = sink, fail_on
        self.opts = {}

    def format(self, _):
        return self

    def option(self, k, v):
        self.opts[k] = v
        return self

    def mode(self, m):
        self.mode_used = m
        return self

    def saveAsTable(self, table):
        if table == self.fail_on:
            raise RuntimeError(f"write denied: {table}")
        self.sink[table] = self.opts


class _SparkDF:
    def __init__(self, pdf, sink, fail_on):
        self.pdf = pdf
        self.write = _Writer(sink, fail_on)


class _Spark:
    def __init__(self, fail_on=None):
        self.saved, self.fail_on = {}, fail_on

    def createDataFrame(self, pdf):
        return _SparkDF(pdf, self.saved, self.fail_on)


DF = pd.DataFrame({"number": ["A", "B", "C"], "cluster": [0, 0, 1],
                   "theme_group": [0, 0, 1], "extra": [1, 2, 3]})
OVERLAY = pd.DataFrame({"theme_group": [0, 1], "business_service": ["x", "y"]})
CC = {"output_table": "cat.sch.clusters", "overlay_table": "cat.sch.overlay"}


def test_both_tables_are_written_with_their_row_counts():
    spark = _Spark()
    result, out_rows, overlay_rows = p5._save_tables(spark, DF, OVERLAY, CC)
    assert set(spark.saved) == {"cat.sch.clusters", "cat.sch.overlay"}
    assert (out_rows, overlay_rows) == (3, 2)
    assert len(result) == 3


def test_export_cols_subsets_the_output():
    spark = _Spark()
    cc = {**CC, "export_cols": ["number", "theme_group"]}
    result, out_rows, _ = p5._save_tables(spark, DF, OVERLAY, cc)
    assert result.columns.tolist() == ["number", "theme_group"] and out_rows == 3


def test_unknown_export_cols_fall_back_to_every_column():
    spark = _Spark()
    cc = {**CC, "export_cols": ["not_a_column"]}
    result, _, _ = p5._save_tables(spark, DF, OVERLAY, cc)
    assert result.columns.tolist() == DF.columns.tolist()


def test_writes_overwrite_the_live_table_with_schema_evolution():
    spark = _Spark()
    p5._save_tables(spark, DF, OVERLAY, CC)
    assert spark.saved["cat.sch.clusters"] == {"overwriteSchema": "true"}


def test_a_failed_overlay_write_raises_instead_of_warning():
    """It used to print a WARNING and continue, leaving a stale overlay table behind."""
    spark = _Spark(fail_on="cat.sch.overlay")
    with pytest.raises(RuntimeError, match="write denied"):
        p5._save_tables(spark, DF, OVERLAY, CC)


def test_a_failed_cluster_write_raises():
    spark = _Spark(fail_on="cat.sch.clusters")
    with pytest.raises(RuntimeError, match="write denied"):
        p5._save_tables(spark, DF, OVERLAY, CC)


def test_unconfigured_table_is_skipped_and_counts_zero():
    spark = _Spark()
    _, out_rows, overlay_rows = p5._save_tables(
        spark, DF, OVERLAY, {"output_table": "cat.sch.clusters"})
    assert out_rows == 3 and overlay_rows == 0
    assert "cat.sch.overlay" not in spark.saved
