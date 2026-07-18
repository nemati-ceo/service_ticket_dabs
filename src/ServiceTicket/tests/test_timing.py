"""Stage 01 timing — Timer laps + summary.

Pure stdlib. perf_counter is monkeypatched to a fixed sequence so durations are exact.
"""

import importlib.util
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE01 = os.path.join(ROOT, "01_problem_health")
sys.path.insert(0, STAGE01)

spec = importlib.util.spec_from_file_location("ph01_timing", os.path.join(STAGE01, "timing.py"))
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)


def test_lap_records_label_and_duration(monkeypatch):
    seq = iter([100.0, 103.0, 108.5])         # init start, lap now, summary now
    monkeypatch.setattr(timing.time, "perf_counter", lambda: next(seq))
    t = timing.Timer()                        # start = 100.0
    dt = t.lap("step1")                       # now = 103.0 -> dt 3.0
    assert dt == 3.0
    assert t.laps == [("step1", 3.0)]


def test_summary_returns_total_elapsed(monkeypatch):
    seq = iter([10.0, 12.0, 15.0, 20.0])      # init, lap1, lap2, summary
    monkeypatch.setattr(timing.time, "perf_counter", lambda: next(seq))
    t = timing.Timer()                        # start 10
    t.lap("a")                                # 12 -> 2.0
    t.lap("b")                                # 15 -> 3.0
    total = t.summary()                       # 20 - 10 = 10.0
    assert total == 10.0
    assert [d for _, d in t.laps] == [2.0, 3.0]


def test_ts_is_timestamp_string():
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", timing.ts())
