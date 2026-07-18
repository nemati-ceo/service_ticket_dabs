"""Stage 01 embeddings — best-effort guarantees.

The Volume load path must NOT depend on mlflow, and the two governance helpers
(_load_from_registry, _register) must never raise. Here mlflow is absent (not stubbed),
so `import mlflow` inside those helpers fails — proving they degrade gracefully. A fake
sentence_transformers is injected so the module imports without the heavy dependency.
"""

import importlib.util
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE01 = os.path.join(ROOT, "01_problem_health")
sys.path.insert(0, STAGE01)

_st = types.ModuleType("sentence_transformers")
_st.SentenceTransformer = object
sys.modules.setdefault("sentence_transformers", _st)

# Ensure mlflow is genuinely unavailable for these tests.
sys.modules.pop("mlflow", None)

spec = importlib.util.spec_from_file_location("ph01_embeddings", os.path.join(STAGE01, "embeddings.py"))
emb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(emb)


def test_load_from_registry_returns_none_without_mlflow():
    # import mlflow inside the helper raises ImportError -> caught -> None
    assert emb._load_from_registry("catalog.schema.model") is None


def test_register_never_raises_without_mlflow(capsys):
    emb._register(object(), "catalog.schema.model")      # must not raise
    assert "Warning: could not register" in capsys.readouterr().out


def test_save_to_volume_is_noop_without_path():
    emb._save_to_volume(object(), None)                  # returns immediately, no crash


def test_save_to_volume_never_raises_on_bad_path(capsys):
    class _Model:
        def save(self, path):
            raise OSError("boom")
    emb._save_to_volume(_Model(), "/proc/\0bad")
    assert "Warning: could not save" in capsys.readouterr().out
