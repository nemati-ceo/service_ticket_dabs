"""Shared pytest fixtures.

The pipeline stages themselves need Spark / Databricks / an LLM endpoint and so
can't run off-cluster. What IS unit-testable is the dependency-free logic:
`mlflow_utils.py` (loaded by path, with a fake `mlflow` injected) and the pure
`evaluate.py` metric functions. These fixtures wire that up.
"""

import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_by_path(module_name, relpath):
    """Import a module from a path under problem_health/ (stages share basenames)."""
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(ROOT, relpath))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeMlflow:
    """Minimal stand-in for the mlflow module that records what was logged."""

    def __init__(self):
        self.runs_started = []
        self.params = {}
        self.metrics = {}
        self.figures = []
        self.experiment = None
        self._active = None
        self.fail_on_log = False        # flip on to test best-effort error swallowing

    def set_experiment(self, name):
        self.experiment = name

    def active_run(self):
        return self._active

    def start_run(self, run_name=None, nested=False):
        self.runs_started.append({"run_name": run_name, "nested": nested})
        self._active = run_name
        outer = self

        class _Ctx:
            def __enter__(self_):
                return self_

            def __exit__(self_, *exc):
                outer._active = None
                return False

        return _Ctx()

    def log_params(self, d):
        if self.fail_on_log:
            raise RuntimeError("boom")
        self.params.update(d)

    def log_metrics(self, d):
        if self.fail_on_log:
            raise RuntimeError("boom")
        self.metrics.update(d)

    def log_figure(self, fig, artifact_name):
        if self.fail_on_log:
            raise RuntimeError("boom")
        self.figures.append(artifact_name)


@pytest.fixture
def mlflow_utils():
    """Fresh import of the module under test."""
    return load_by_path("mlflow_utils", "mlflow_utils.py")


@pytest.fixture
def fake_mlflow():
    """Inject a recording FakeMlflow as `import mlflow`, restore afterwards."""
    fake = FakeMlflow()
    saved = sys.modules.get("mlflow")
    sys.modules["mlflow"] = fake
    try:
        yield fake
    finally:
        if saved is not None:
            sys.modules["mlflow"] = saved
        else:
            sys.modules.pop("mlflow", None)


@pytest.fixture
def no_mlflow():
    """Make `import mlflow` fail, to exercise the not-installed path."""
    saved = sys.modules.get("mlflow")
    sys.modules["mlflow"] = None        # forces ImportError on `import mlflow`
    try:
        yield
    finally:
        if saved is not None:
            sys.modules["mlflow"] = saved
        else:
            sys.modules.pop("mlflow", None)
