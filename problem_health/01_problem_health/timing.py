"""
timing.py — lightweight per-step + total timing for the pipeline.
"""

import time
from datetime import datetime


def ts():
    """Wall-clock timestamp for log lines."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Timer:
    """Per-step + total timing. Call lap() after each step; summary() at the end."""

    def __init__(self):
        self.start = time.perf_counter()
        self.mark = self.start
        self.laps = []  # list of (label, seconds)
        print(f"[time] pipeline started at {ts()}")

    def lap(self, label):
        now = time.perf_counter()
        dt = now - self.mark
        self.mark = now
        self.laps.append((label, dt))
        print(f"[time] {label}: {dt:.2f}s  (elapsed {now - self.start:.2f}s)  @ {ts()}")
        return dt

    def summary(self):
        total = time.perf_counter() - self.start
        print("-" * 60)
        print(f"[time] STEP TIMINGS (finished {ts()})")
        for label, dt in self.laps:
            pct = (dt / total * 100) if total else 0
            print(f"[time]   {label:<28} {dt:8.2f}s  {pct:5.1f}%")
        print(f"[time]   {'TOTAL':<28} {total:8.2f}s  100.0%")
        print("-" * 60)
        return total
