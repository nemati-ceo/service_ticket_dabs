"""Stage 04 routes on config `mode`. Train must never write the production table."""

import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE04 = os.path.join(ROOT, "04_gradient_boost_inference")
sys.path.insert(0, STAGE04)


def _pipeline():
    spec = importlib.util.spec_from_file_location(
        "ph04_pipeline", os.path.join(STAGE04, "pipeline.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def pl():
    return _pipeline()


@pytest.mark.parametrize("mode,expected", [
    ("train", "train"),
    ("production", "inference"),
    ("PRODUCTION", "inference"),
    (None, "inference"),          # missing mode must default to production, not train
])
def test_run_gbm_routes_on_mode(pl, monkeypatch, mode, expected):
    called = []

    monkeypatch.setattr(pl, "build_features", lambda spark, cfg: ("FEATURES", None, None))
    monkeypatch.setattr(pl, "run_gbm_inference", lambda spark, cfg: called.append("inference"))

    import train as tr
    monkeypatch.setattr(tr, "run_gbm_train",
                        lambda spark, cfg, feature_df: called.append("train"))

    cfg = {} if mode is None else {"mode": mode}
    pl.run_gbm(spark=None, cfg=cfg)

    assert called == [expected]
