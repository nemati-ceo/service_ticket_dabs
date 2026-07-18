"""Stage 00 input-sync orchestration.

sync.py copies refine snapshots into consume mirrors. The Spark write itself needs a
cluster, but the orchestration — how config is parsed into (source, target) pairs, the
disabled short-circuit, and the per-table loop — is pure and pinned here. A fake pyspark
is injected so the module imports without a cluster.
"""

import importlib.util
import os
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE00 = os.path.join(ROOT, "00_input_sync")
sys.path.insert(0, STAGE00)

# Stub pyspark so `from pyspark.sql import functions as F` imports off-cluster.
_fp = types.ModuleType("pyspark")
_fs = types.ModuleType("pyspark.sql")
_ff = types.ModuleType("pyspark.sql.functions")
_ff.current_timestamp = lambda: "NOW"
_fs.functions = _ff
_fp.sql = _fs
sys.modules.setdefault("pyspark", _fp)
sys.modules.setdefault("pyspark.sql", _fs)
sys.modules.setdefault("pyspark.sql.functions", _ff)

spec = importlib.util.spec_from_file_location("sync", os.path.join(STAGE00, "sync.py"))
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


# --- _sources: config -> (source, target) pairs ------------------------------

def test_sources_list_form():
    sc = {"tables": [{"source": "r.a", "target": "c.a"},
                     {"source": "r.b", "target": "c.b"}]}
    assert sync._sources(sc) == [("r.a", "c.a"), ("r.b", "c.b")]


def test_sources_single_pair_form():
    assert sync._sources({"source": "r.a", "target": "c.a"}) == [("r.a", "c.a")]


def test_sources_list_takes_precedence_over_single():
    sc = {"tables": [{"source": "r.a", "target": "c.a"}], "source": "x", "target": "y"}
    assert sync._sources(sc) == [("r.a", "c.a")]


def test_sources_raises_when_neither_form_present():
    with pytest.raises(ValueError, match="input_sync needs"):
        sync._sources({})


# --- run_input_sync: skip / loop / totals ------------------------------------

def test_disabled_skips_and_returns_none(capsys, monkeypatch):
    calls = []
    monkeypatch.setattr(sync, "_copy_table", lambda *a: calls.append(a))
    out = sync.run_input_sync(object(), {"input_sync": {"enabled": False}})
    assert out is None
    assert calls == []                                  # nothing copied
    assert "disabled" in capsys.readouterr().out


def test_missing_input_sync_block_is_treated_as_disabled(monkeypatch):
    monkeypatch.setattr(sync, "_copy_table",
                        lambda *a: pytest.fail("must not copy when block absent"))
    assert sync.run_input_sync(object(), {}) is None


def test_enabled_copies_every_pair_and_sums_rows(monkeypatch):
    seen = []

    def fake_copy(spark, source, target):
        seen.append((source, target))
        return {"c.a": 10, "c.b": 5}[target]            # row counts per target

    monkeypatch.setattr(sync, "_copy_table", fake_copy)
    cfg = {"input_sync": {"enabled": True, "tables": [
        {"source": "r.a", "target": "c.a"},
        {"source": "r.b", "target": "c.b"}]}}
    counts = sync.run_input_sync(object(), cfg)
    assert seen == [("r.a", "c.a"), ("r.b", "c.b")]      # every pair copied, in order
    assert counts == {"c.a": 10, "c.b": 5}
    assert sum(counts.values()) == 15
