"""mlflow_utils.py — shared MLflow logging for the whole pipeline.

The FULL pipeline logs to ONE parent MLflow run: run.py opens it and every stage
opens its OWN nested child run underneath, so the UI shows an expandable tree —
parent (pipeline) -> child (ph01) , child (ph02) , ... — each child carrying its
own status, duration, params, metrics, tags and artifacts. Run a stage on its own
and it opens a single top-level run instead — same logging, no orchestration needed.

Keys are still stage-namespaced (ph01_, ph02_, ...) so that even in the degraded
"log straight into the parent" fallback nothing collides.

Best-effort by design: each call swallows its own errors, so a tracking problem
(no MLflow server, bad creds, missing artifact dep) prints a warning but never
breaks a run. Toggle everything with `mlflow.enabled` in config.yml.

    # run.py (full pipeline) — one parent run, a nested child per stage:
    with mu.pipeline_run(cfg):
        stage01(...); stage02(...); ...

    # inside a stage's pipeline.py — nested under the pipeline run, or solo if alone:
    with mu.stage_run(cfg, "ph02_summarization") as ml:
        ml.log_params({"model": model})
        ml.log_metrics({"wall_clock_s": total})
        ml.log_dict(sample_rows, "samples.json")     # artifacts too
        ml.set_tags({"input_table": inp})
"""

import os
import re
import subprocess
from contextlib import contextmanager


def _mlflow_cfg(cfg):
    return cfg.get("mlflow") or {}


def enabled(cfg):
    """True unless mlflow.enabled is explicitly set false (default on)."""
    return bool(_mlflow_cfg(cfg).get("enabled", True))


# --- metric helpers -----------------------------------------------------------

def topk_metrics(d, prefix="top"):
    """Flatten a {k: accuracy} dict into metric names: top_1_accuracy, top_5_accuracy, ..."""
    return {f"{prefix}_{k}_accuracy": v for k, v in (d or {}).items()}


def baseline_delta_metrics(topk, baselines):
    """Turn config baselines into comparison metrics next to the measured top-k.

    For every baseline {name: {k: value}} emit `baseline_<name>_top_<k>`, and when
    the same k was actually measured (`topk[k]`) also emit `delta_<name>_top_<k>` =
    measured - baseline. Lets the MLflow UI show regressions against PH02/PH05 at a
    glance without leaving the run.
    """
    out = {}
    for name, base in (baselines or {}).items():
        for k, bval in (base or {}).items():
            if bval is None:
                continue
            out[f"baseline_{name}_top_{k}"] = bval
            measured = (topk or {}).get(k)
            if measured is not None:
                out[f"delta_{name}_top_{k}"] = measured - bval
    return out


def _metric_key(label):
    """Sanitize a free-text step label into an MLflow-safe metric key."""
    return re.sub(r"_+", "_", re.sub(r"[^0-9A-Za-z]+", "_", str(label))).strip("_").lower()


def step_timings(laps, prefix="step"):
    """Turn a Timer's [(label, seconds), ...] laps into per-step `step_<label>_s` metrics."""
    return {f"{prefix}_{_metric_key(label)}_s": dt for label, dt in (laps or [])}


# --- runtime context (tags + config snapshot) ---------------------------------

def _import_mlflow(label):
    try:
        import mlflow
        return mlflow
    except Exception as e:
        print(f"[mlflow] not available, tracking off for {label} ({e})")
        return None


def _git_info():
    """Best-effort {git_commit, git_branch} for the deployed code (empty on failure)."""
    info = {}
    try:
        root = os.path.dirname(os.path.abspath(__file__))
        for key, args in (("git_commit", ["rev-parse", "HEAD"]),
                          ("git_branch", ["rev-parse", "--abbrev-ref", "HEAD"])):
            out = subprocess.run(["git", "-C", root, *args],
                                 capture_output=True, text=True, timeout=5)
            if out.returncode == 0 and out.stdout.strip():
                info[key] = out.stdout.strip()
    except Exception:
        pass
    return info


def _spark_tags():
    """Best-effort Databricks context (cluster id, user, runtime version)."""
    tags = {}
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession()
        if spark is None:
            return tags
        for key, prop in (
                ("cluster_id", "spark.databricks.clusterUsageTags.clusterId"),
                ("user", "spark.databricks.clusterUsageTags.user"),
                ("runtime_version", "spark.databricks.clusterUsageTags.sparkVersion")):
            try:
                val = spark.conf.get(prop)
                if val:
                    tags[key] = val
            except Exception:
                pass
    except Exception:
        pass
    return tags


def runtime_tags(cfg):
    """Standard run-level tags: pipeline name, git, Databricks context, run mode."""
    tags = {"pipeline": "problem_health"}
    tags.update(_git_info())
    tags.update(_spark_tags())
    limit = (cfg.get("run") or {}).get("limit")
    tags["run_mode"] = "test" if limit else "full"
    if limit:
        tags["run_limit"] = str(limit)
    for env_key, tag_key in (("USER", "os_user"), ("DB_CLUSTER_ID", "cluster_id")):
        v = os.environ.get(env_key)
        if v and tag_key not in tags:
            tags[tag_key] = v
    return tags


def _set_tracking_uri(mlflow, cfg):
    uri = _mlflow_cfg(cfg).get("tracking_uri")
    if uri:
        try:
            mlflow.set_tracking_uri(uri)
        except Exception as e:
            print(f"[mlflow] could not set tracking uri {uri} ({e})")


def _set_experiment(mlflow, cfg):
    experiment = _mlflow_cfg(cfg).get("experiment")
    if experiment:
        try:
            mlflow.set_experiment(experiment)
        except Exception as e:
            print(f"[mlflow] could not set experiment {experiment} ({e})")
            print(f"[mlflow] -> the run-as service principal needs CAN_EDIT on "
                  f"'{experiment}'. Nothing is logged until it does.")


def _enable_system_metrics(mlflow, cfg):
    """Turn on CPU/GPU/memory sampling (default on; needs `psutil`/`pynvml`)."""
    if not _mlflow_cfg(cfg).get("log_system_metrics", True):
        return
    try:
        mlflow.enable_system_metrics_logging()
    except Exception as e:
        print(f"[mlflow] system metrics off ({e})")


def _log_run_context(mlflow, cfg):
    """Stamp the active run with runtime tags + a full config snapshot artifact."""
    try:
        mlflow.set_tags(runtime_tags(cfg))
    except Exception as e:
        print(f"[mlflow] run tags skipped ({e})")
    try:
        mlflow.log_dict(cfg, "config_snapshot.yaml")
    except Exception as e:
        print(f"[mlflow] config snapshot skipped ({e})")


# --- the per-stage best-effort logger -----------------------------------------

class _Logger:
    """Best-effort logger that namespaces every key/artifact with a stage tag.

    `tag` (e.g. "ph03") prefixes params/metrics/tags (ph03_model, ph03_top_5_accuracy)
    and artifact paths (ph03/plot.html) so logging is unambiguous even when many
    stages share ONE run. A None mlflow makes every method a no-op.
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

    def log_metrics(self, metrics, step=None):
        if not self._mlflow:
            return
        clean = {f"{self._tag}_{k}": float(v) for k, v in metrics.items() if v is not None}
        try:
            if step is None:
                self._mlflow.log_metrics(clean)
            else:
                self._mlflow.log_metrics(clean, step=step)
        except Exception as e:
            print(f"[mlflow] metrics skipped ({e})")

    def set_tags(self, tags):
        if not self._mlflow:
            return
        try:
            self._mlflow.set_tags(
                {f"{self._tag}_{k}": v for k, v in tags.items() if v is not None})
        except Exception as e:
            print(f"[mlflow] tags skipped ({e})")

    def log_figure(self, fig, artifact_name):
        if not self._mlflow:
            return
        try:
            self._mlflow.log_figure(fig, f"{self._tag}/{artifact_name}")
        except Exception as e:
            print(f"[mlflow] figure {artifact_name} skipped ({e})")

    def log_dict(self, d, artifact_name):
        """Log a dict/list as a json/yaml artifact (e.g. merge logs, eval breakdowns)."""
        if not self._mlflow:
            return
        try:
            self._mlflow.log_dict(d, f"{self._tag}/{artifact_name}")
        except Exception as e:
            print(f"[mlflow] dict {artifact_name} skipped ({e})")

    def log_text(self, text, artifact_name):
        """Log free text as an artifact (e.g. the input SQL, a traceback)."""
        if not self._mlflow:
            return
        try:
            self._mlflow.log_text(text, f"{self._tag}/{artifact_name}")
        except Exception as e:
            print(f"[mlflow] text {artifact_name} skipped ({e})")

    def log_table(self, data, artifact_name):
        """Log a pandas DataFrame / dict-of-columns as an MLflow table artifact."""
        if not self._mlflow:
            return
        try:
            self._mlflow.log_table(data=data, artifact_file=f"{self._tag}/{artifact_name}")
        except Exception as e:
            print(f"[mlflow] table {artifact_name} skipped ({e})")


# --- run lifecycle ------------------------------------------------------------

@contextmanager
def pipeline_run(cfg, run_name="problem_health_pipeline"):
    """Open the ONE parent run that the whole pipeline (run.py) logs under.

    Sets tracking uri / experiment, turns on system metrics, then stamps the run
    with runtime tags + a config snapshot. No-op (body still runs) when mlflow is
    disabled, missing, or fails to start — stages then each open their own run.
    """
    if not enabled(cfg):
        yield
        return
    mlflow = _import_mlflow(run_name)
    if mlflow is None:
        yield
        return
    _set_tracking_uri(mlflow, cfg)
    _set_experiment(mlflow, cfg)
    _enable_system_metrics(mlflow, cfg)
    try:
        run = mlflow.start_run(run_name=run_name)
    except Exception as e:
        print(f"[mlflow] could not start run {run_name} ({e})")
        yield
        return
    with run:
        _log_run_context(mlflow, cfg)
        yield


@contextmanager
def stage_run(cfg, run_name):
    """Yield a best-effort logger for a stage; tag = run_name's leading token (ph03).

    Opens a NESTED child run when the pipeline's parent run is active (full-pipeline
    case), otherwise a top-level run of its own (run-this-stage-alone case). Either
    way the stage records its own status/duration; on an exception inside the block
    the run is left FAILED (mlflow's run context does this) and a `status=failed`
    tag is set before the error propagates.
    """
    tag = run_name.split("_", 1)[0]
    if not enabled(cfg):
        yield _Logger(None, tag)
        return
    mlflow = _import_mlflow(run_name)
    if mlflow is None:
        yield _Logger(None, tag)
        return

    parent_active = mlflow.active_run() is not None
    if not parent_active:
        _set_tracking_uri(mlflow, cfg)
        _set_experiment(mlflow, cfg)
    try:
        run = mlflow.start_run(run_name=run_name, nested=parent_active)
    except Exception as e:
        print(f"[mlflow] could not start run {run_name} ({e})")
        yield _Logger(None, tag)
        return

    logger = _Logger(mlflow, tag)
    with run:
        if not parent_active:
            _log_run_context(mlflow, cfg)   # a solo stage run still gets tags + config
        logger.set_tags({"status": "running"})
        try:
            yield logger
        except Exception:
            logger.set_tags({"status": "failed"})
            raise
        logger.set_tags({"status": "ok"})


def log_stage_failure(cfg, run_name, exc):
    """Record a stage failure that happened OUTSIDE its own stage_run block.

    The heavy work in a stage runs before it reaches `stage_run` (which logs at the
    end), so a crash there would otherwise leave MLflow looking clean. run.py calls
    this from its per-stage `except` to drop a nested FAILED run carrying the error
    message + traceback. Best-effort and silent on its own failures.
    """
    if not enabled(cfg):
        return
    mlflow = _import_mlflow(run_name)
    if mlflow is None:
        return
    tag = run_name.split("_", 1)[0]
    import traceback as _tb
    parent_active = mlflow.active_run() is not None
    try:
        mlflow.start_run(run_name=run_name, nested=parent_active)
    except Exception as e:
        print(f"[mlflow] could not start failure run {run_name} ({e})")
        return
    try:
        mlflow.set_tags({f"{tag}_status": "failed",
                         f"{tag}_error": f"{type(exc).__name__}: {exc}"[:500]})
        mlflow.log_text(_tb.format_exc(), f"{tag}/traceback.txt")
    except Exception as e:
        print(f"[mlflow] failure details skipped ({e})")
    finally:
        try:
            mlflow.end_run(status="FAILED")
        except Exception:
            pass
