"""Delta small-file compaction is enabled once, centrally, for every stage's writes.

Every stage overwrites its output table each run, so without optimizeWrite/autoCompact
the file count tracks the partition count and the tables degrade into many small files
as the data grows. Setting it in get_spark() covers all stages; an older runtime that
rejects the flags must not break the run.
"""

import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

spec = importlib.util.spec_from_file_location("ph_run", os.path.join(ROOT, "run.py"))
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)

OPTIMIZE_WRITE = "spark.databricks.delta.optimizeWrite.enabled"
AUTO_COMPACT = "spark.databricks.delta.autoCompact.enabled"


class _Conf:
    def __init__(self):
        self.set_keys = {}

    def set(self, k, v):
        self.set_keys[k] = v


class _Session:
    def __init__(self):
        self.conf = _Conf()


def test_both_compaction_flags_are_enabled():
    s = run._enable_delta_compaction(_Session())
    assert s.conf.set_keys[OPTIMIZE_WRITE] == "true"
    assert s.conf.set_keys[AUTO_COMPACT] == "true"


def test_returns_the_session_so_get_spark_can_chain():
    s = _Session()
    assert run._enable_delta_compaction(s) is s


def test_an_unsupported_flag_never_breaks_the_run(capsys):
    class _Rejecting(_Session):
        def __init__(self):
            super().__init__()
            self.conf = self                       # conf.set -> our set below
            self.set_keys = {}

        def set(self, k, v):
            if k == OPTIMIZE_WRITE:
                raise Exception("unsupported on this runtime")
            self.set_keys[k] = v

    s = run._enable_delta_compaction(_Rejecting())
    assert s.set_keys[AUTO_COMPACT] == "true"      # the other flag still applied
    assert "could not set" in capsys.readouterr().out
