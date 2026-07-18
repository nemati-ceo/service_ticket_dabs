"""Stage 01 servicenow_source — per-incident fetch resilience.

fetch_incidents must not let one bad incident kill the batch: failures are skipped and
logged, and only an all-empty result raises. `requests` is stubbed so no network is hit.
"""

import importlib.util
import os
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE01 = os.path.join(ROOT, "01_problem_health")
sys.path.insert(0, STAGE01)

spec = importlib.util.spec_from_file_location("ph01_snow", os.path.join(STAGE01, "servicenow_source.py"))
snow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snow)

CFG = {"servicenow": {"base_url": "https://gw.example/api", "timeout": 5}, "secrets": {}}


class _Resp:
    def __init__(self, ok, data=None):
        self._ok, self._data = ok, data or {}

    def raise_for_status(self):
        if not self._ok:
            raise RuntimeError("HTTP 500")

    def json(self):
        return self._data


def _install_requests(monkeypatch, responder):
    fake = types.ModuleType("requests")
    fake.get = lambda url, headers=None, timeout=None: responder(url)
    monkeypatch.setitem(sys.modules, "requests", fake)


def test_one_bad_incident_is_skipped_not_fatal(monkeypatch, capsys):
    def responder(url):
        return _Resp(True, {"result": {"number": "INC1"}}) if url.endswith("INC1") else _Resp(False)
    _install_requests(monkeypatch, responder)

    df = snow.fetch_incidents(CFG, numbers=["INC1", "INC2"])
    assert len(df) == 1                                  # INC2 skipped, INC1 kept
    out = capsys.readouterr().out
    assert "skipped INC2" in out


def test_all_failing_raises(monkeypatch):
    _install_requests(monkeypatch, lambda url: _Resp(False))
    with pytest.raises(RuntimeError, match="no incidents fetched"):
        snow.fetch_incidents(CFG, numbers=["INC1", "INC2"])
