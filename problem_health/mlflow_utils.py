"""mlflow_utils.py — shared MLflow logging for the whole pipeline.

The FULL pipeline logs to ONE MLflow run: run.py opens a single parent run and
every stage logs its params/metrics/artifacts into that same run, namespaced by a
short stage tag (ph01_, ph02_, ...) so keys never collide. Run a stage on its own
and it opens its own one-off run instead — same logging, no orchestration needed.

Best-effort by design: each call swallows its own errors, so a tracking problem
(no MLflow server, bad creds, missing artifact dep) prints a warning but never
breaks a run. Toggle everything with `mlflow.enabled` in config.yml.

    # run.py (full pipeline) — one run for all stages:
    with mu.pipeline_run(cfg):
        stage01(...); stage02(...); ...

    # inside a stage's pipeline.py — logs into the active run, or its own if solo:
    with mu.stage_run(cfg, "ph02_summarization") as ml:
        ml.log_params({"model": model})
        ml.log_metrics({"wall_clock_s": total})
"""

from contextlib import contextmanager


def _mlflow_cfg(cfg):
    return cfg.get("mlflow") or {}


def enabled(cfg):
    """True unless mlflow.enabled is explicitly set false (default on)."""
    return bool(_mlflow_cfg(cfg).get("enabled", True))


def topk_metrics(d, prefix="top"):
    """Flatten a {k: accuracy} dict into metric names: top_1_accuracy, top_5_accuracy, ..."""
    return {f"{prefix}_{k}_accuracy": v for k, v in (d or {}).items()}


def _import_mlflow(label):
    try:
        import mlflow
        return mlflow
    except Exception as e:
        print(f"[mlflow] not available, tracking off for {label} ({e})")
        return None


def _set_experiment(mlflow, cfg):
    experiment = _mlflow_cfg(cfg).get("experiment")
    if experiment:
        try:
            mlflow.set_experiment(experiment)
        except Exception as e:
            print(f"[mlflow] could not set experiment {experiment} ({e})")


class _Logger:
    """Best-effort logger that namespaces every key/artifact with a stage tag.

    `tag` (e.g. "ph03") prefixes params/metrics (ph03_model, ph03_top_5_accuracy)
    and artifact paths (ph03/plot.html) so many stages can share ONE run without
    colliding. A None mlflow makes every method a no-op.
    """

    def __init__(self, mlflow, tag):
        self._mlflow = mlflow
        self._tag = tag

    def log_params(self, params):
        if not self._mlflow:
            return
        try:
            self._mlflow.log_params(
                {f"{self._tag}_{k}": v for k, v in params.items() if v is not None})
        except Exception as e:
            print(f"[mlflow] params skipped ({e})")

    def log_metrics(self, metrics):
        if not self._mlflow:
            return
        try:
            self._mlflow.log_metrics(
                {f"{self._tag}_{k}": float(v) for k, v in metrics.items() if v is not None})
        except Exception as e:
            print(f"[mlflow] metrics skipped ({e})")

    def log_figure(self, fig, artifact_name):
        if not self._mlflow:
            return
        try:
            self._mlflow.log_figure(fig, f"{self._tag}/{artifact_name}")
        except Exception as e:
            print(f"[mlflow] figure {artifact_name} skipped ({e})")


@contextmanager
def pipeline_run(cfg, run_name="problem_health_pipeline"):
    """Open the ONE run that the whole pipeline (run.py) logs into.

    No-op (body still runs) when mlflow is disabled, missing, or fails to start —
    stages then each open their own run instead, so logging degrades gracefully.
    """
    if not enabled(cfg):
        yield
        return
    mlflow = _import_mlflow(run_name)
    if mlflow is None:
        yield
        return
    _set_experiment(mlflow, cfg)
    try:
        run = mlflow.start_run(run_name=run_name)
    except Exception as e:
        print(f"[mlflow] could not start run {run_name} ({e})")
        yield
        return
    with run:
        yield


@contextmanager
def stage_run(cfg, run_name):
    """Yield a best-effort logger for a stage; tag = run_name's leading token (ph03).

    Logs into the pipeline's active run when one exists (the full-pipeline case),
    otherwise opens its own one-off run (the run-this-stage-alone case). Either way
    the stage never has to know which situation it's in.
    """
    tag = run_name.split("_", 1)[0]
    if not enabled(cfg):
        yield _Logger(None, tag)
        return
    mlflow = _import_mlflow(run_name)
    if mlflow is None:
        yield _Logger(None, tag)
        return

    if mlflow.active_run() is not None:
        # Pipeline already opened the parent run — log straight into it.
        yield _Logger(mlflow, tag)
        return

    _set_experiment(mlflow, cfg)
    try:
        run = mlflow.start_run(run_name=run_name)
    except Exception as e:
        print(f"[mlflow] could not start run {run_name} ({e})")
        yield _Logger(None, tag)
        return
    with run:
        yield _Logger(mlflow, tag)
