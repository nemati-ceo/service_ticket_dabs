"""model_cache.load_cached — the no-redownload guarantee, shared by stages 03 and 05.

A populated Volume path must be LOADED, never fetched. Stage 05's clustering embedder
used to call SentenceTransformer(model_name) directly with no Volume at all, so it pulled
a few hundred MB on every single run.
"""

import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("model_cache",
                                              os.path.join(ROOT, "model_cache.py"))
mc = importlib.util.module_from_spec(spec)
sys.modules["model_cache"] = mc
spec.loader.exec_module(mc)


class _FakeModel:
    """Records how it was constructed and whether it was saved."""

    def __init__(self, source, **kwargs):
        self.source = source
        self.kwargs = kwargs
        self.saved_to = None

    def save(self, path):
        self.saved_to = path


def test_loads_from_volume_when_populated(tmp_path, capsys):
    vol = tmp_path / "model"
    vol.mkdir()
    (vol / "config.json").write_text("{}")            # non-empty dir = cached

    model = mc.load_cached("hf/name", str(vol), _FakeModel)

    assert model.source == str(vol)                    # loaded from the Volume path...
    assert model.saved_to is None                      # ...and never re-saved
    assert "from Volume" in capsys.readouterr().out


def test_downloads_and_caches_when_volume_empty(tmp_path, capsys):
    vol = tmp_path / "model"                           # does not exist yet
    model = mc.load_cached("hf/name", str(vol), _FakeModel)

    assert model.source == "hf/name"                   # fell back to the HF name
    assert model.saved_to == str(vol)                  # and cached it for next run
    assert "downloaded and cached" in capsys.readouterr().out


def test_empty_volume_dir_is_not_treated_as_cached(tmp_path):
    vol = tmp_path / "model"
    vol.mkdir()                                        # exists but empty
    assert mc.load_cached("hf/name", str(vol), _FakeModel).source == "hf/name"


def test_no_volume_path_warns_that_it_redownloads(capsys):
    """Silence here is how stage 05 re-downloaded every run without anyone noticing."""
    model = mc.load_cached("hf/name", None, _FakeModel)
    assert model.source == "hf/name" and model.saved_to is None
    assert "re-downloads every run" in capsys.readouterr().out


def test_caching_failure_never_breaks_the_load(tmp_path, capsys):
    class _Unsaveable(_FakeModel):
        def save(self, path):
            raise OSError("volume read-only")

    model = mc.load_cached("hf/name", str(tmp_path / "m"), _Unsaveable)
    assert model.source == "hf/name"                   # still usable
    assert "WARNING" in capsys.readouterr().out


def test_kwargs_are_passed_through():
    assert mc.load_cached("hf/name", None, _FakeModel, max_length=256).kwargs == {
        "max_length": 256}


def test_both_stages_use_this_helper():
    """Stage 03 kept a private copy; stage 05 had none. One implementation now."""
    for stage, mod in (("03_cross_encoder_rerank", "rerank"), ("05_clustering", "clustering")):
        src = open(os.path.join(ROOT, stage, f"{mod}.py")).read()
        assert "load_cached" in src, f"{stage}/{mod}.py no longer uses the shared cache"
        assert "def _load_cached" not in src, f"{stage}/{mod}.py reintroduced a private copy"
