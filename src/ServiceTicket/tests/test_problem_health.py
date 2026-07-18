"""Stage 01 problem-health pure helpers.

Full run_problem_health needs a cluster + the embedding model. These tests pin the two
pure helpers that were refactored: `_data_quality` (input-rot metrics for MLflow) and
`_save_parquet` (the deduped best-effort Volume dump). A fake sentence_transformers is
injected so the module imports without the heavy dependency.
"""

import importlib.util
import os
import sys
import types

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE01 = os.path.join(ROOT, "01_problem_health")
sys.path.insert(0, STAGE01)

# Stub sentence_transformers so `import embeddings` (-> pipeline) imports off-cluster.
_st = types.ModuleType("sentence_transformers")
_st.SentenceTransformer = object
sys.modules.setdefault("sentence_transformers", _st)

spec = importlib.util.spec_from_file_location("ph01_pipeline", os.path.join(STAGE01, "pipeline.py"))
pl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pl)


def _has_parquet_engine():
    for eng in ("pyarrow", "fastparquet"):
        try:
            __import__(eng)
            return True
        except ImportError:
            pass
    return False


needs_parquet = pytest.mark.skipif(
    not _has_parquet_engine(), reason="no pyarrow/fastparquet engine (present on Databricks)")


# --- _data_quality ------------------------------------------------------------

def test_data_quality_counts_rows_dupes_and_nulls():
    df = pd.DataFrame({
        "number": ["A", "A", "B", "C"],                 # 1 duplicate key of 4 -> 25%
        "short_description": ["x", "", "y", None],       # 2 blank of 4 -> 50%
        "description": ["d", "d", "d", "d"],             # 0 blank
    })
    m = pl._data_quality(df, "number")
    assert m["input_rows"] == 4
    assert m["dup_key_pct"] == 25.0
    assert m["null_short_description_pct"] == 50.0
    assert m["null_description_pct"] == 0.0


def test_data_quality_never_raises_on_bad_input():
    # missing key column + no text columns must not raise, just skip those metrics
    m = pl._data_quality(pd.DataFrame({"other": [1, 2]}), "number")
    assert m["input_rows"] == 2
    assert "dup_key_pct" not in m


def test_data_quality_handles_empty_frame():
    m = pl._data_quality(pd.DataFrame({"number": []}), "number")
    assert m["input_rows"] == 0                          # n==0 guards the divisions


# --- _save_parquet ------------------------------------------------------------

@needs_parquet
def test_save_parquet_writes_file(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3]})
    pl._save_parquet(df, str(tmp_path), "out.parquet", "Test")
    written = tmp_path / "out.parquet"
    assert written.exists()
    assert pd.read_parquet(written)["a"].tolist() == [1, 2, 3]


@needs_parquet
def test_save_parquet_creates_missing_dir(tmp_path):
    target = tmp_path / "nested" / "dir"
    pl._save_parquet(pd.DataFrame({"a": [1]}), str(target), "out.parquet", "Test")
    assert (target / "out.parquet").exists()


def test_save_parquet_never_raises_on_bad_path(capsys):
    # a NUL byte in the path makes makedirs/to_parquet fail — helper must swallow it
    pl._save_parquet(pd.DataFrame({"a": [1]}), "/proc/\0bad", "out.parquet", "Test")
    assert "WARNING" in capsys.readouterr().out
