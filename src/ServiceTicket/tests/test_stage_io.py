"""stage_io.load_frame — the shared stage input loader.

Replaces the three near-identical _load_frame copies that lived in stages 03/04/05.
The parquet fallback is deliberately gone: it caught the table error and read a stale
file instead, so a broken table looked like a healthy run.
"""

import importlib.util
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("stage_io", os.path.join(ROOT, "stage_io.py"))
sio = importlib.util.module_from_spec(spec)
sys.modules["stage_io"] = sio
spec.loader.exec_module(sio)

FRAME = pd.DataFrame({"number": ["INC1"], "text": ["hello"]})


class _Spark:
    def __init__(self, table_error=None):
        self.table_error = table_error
        self.sql_called = self.table_called = None

    def sql(self, q):
        self.sql_called = q
        return _Result()

    def table(self, t):
        self.table_called = t
        if self.table_error:
            raise self.table_error
        return _Result()


class _Result:
    def toPandas(self):
        return FRAME


def test_sql_wins_when_both_are_set():
    spark = _Spark()
    out = sio.load_frame(spark, "SELECT 1", "some.table", what="tickets")
    assert out.equals(FRAME)
    assert spark.sql_called == "SELECT 1" and spark.table_called is None


def test_table_is_used_when_no_sql():
    spark = _Spark()
    assert sio.load_frame(spark, None, "cat.sch.tbl", what="tickets").equals(FRAME)
    assert spark.table_called == "cat.sch.tbl"


def test_a_broken_table_raises_instead_of_falling_back():
    """The regression: this used to be swallowed and a stale parquet read instead."""
    spark = _Spark(table_error=RuntimeError("TABLE_OR_VIEW_NOT_FOUND"))
    with pytest.raises(RuntimeError, match="TABLE_OR_VIEW_NOT_FOUND"):
        sio.load_frame(spark, None, "cat.sch.missing", what="tickets")


def test_no_source_configured_names_the_stage_input():
    with pytest.raises(ValueError, match="no input source for tickets"):
        sio.load_frame(_Spark(), None, None, what="tickets")


def test_what_label_appears_in_the_log(capsys):
    sio.load_frame(_Spark(), None, "cat.sch.tbl", what="problem catalog")
    assert "problem catalog" in capsys.readouterr().out
