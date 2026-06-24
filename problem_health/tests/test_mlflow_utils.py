"""Tests for mlflow_utils — the shared best-effort MLflow logging layer.

Covers: the enabled toggle, topk flattening, key/artifact namespacing, the
no-op paths (disabled / mlflow missing), best-effort error swallowing, and the
core contract that the FULL pipeline logs to exactly ONE run while a stage run
on its own opens its own run.
"""


# --- small pure helpers -------------------------------------------------------

def test_enabled_defaults_true(mlflow_utils):
    assert mlflow_utils.enabled({}) is True
    assert mlflow_utils.enabled({"mlflow": {}}) is True


def test_enabled_explicit_false(mlflow_utils):
    assert mlflow_utils.enabled({"mlflow": {"enabled": False}}) is False


def test_topk_metrics_flattens(mlflow_utils):
    assert mlflow_utils.topk_metrics({1: 0.4, 5: 0.8}) == {
        "top_1_accuracy": 0.4, "top_5_accuracy": 0.8}


def test_topk_metrics_handles_empty(mlflow_utils):
    assert mlflow_utils.topk_metrics(None) == {}
    assert mlflow_utils.topk_metrics({}) == {}


# --- _Logger ------------------------------------------------------------------

def test_logger_prefixes_and_drops_none(mlflow_utils, fake_mlflow):
    log = mlflow_utils._Logger(fake_mlflow, "ph03")
    log.log_params({"model": "ce", "skip": None})
    log.log_metrics({"acc": 0.5, "skip": None})
    assert fake_mlflow.params == {"ph03_model": "ce"}
    assert fake_mlflow.metrics == {"ph03_acc": 0.5}


def test_logger_metrics_cast_to_float(mlflow_utils, fake_mlflow):
    mlflow_utils._Logger(fake_mlflow, "ph01").log_metrics({"n": 7})
    assert fake_mlflow.metrics["ph01_n"] == 7.0
    assert isinstance(fake_mlflow.metrics["ph01_n"], float)


def test_logger_figure_namespaced(mlflow_utils, fake_mlflow):
    mlflow_utils._Logger(fake_mlflow, "ph05").log_figure(object(), "clusters_2d.html")
    assert fake_mlflow.figures == ["ph05/clusters_2d.html"]


def test_logger_noop_when_mlflow_none(mlflow_utils):
    log = mlflow_utils._Logger(None, "ph02")          # must not raise
    log.log_params({"a": 1})
    log.log_metrics({"b": 2})
    log.log_figure(object(), "x.html")


def test_logger_swallows_backend_errors(mlflow_utils, fake_mlflow):
    fake_mlflow.fail_on_log = True
    log = mlflow_utils._Logger(fake_mlflow, "ph04")
    log.log_params({"a": 1})                           # backend raises -> swallowed
    log.log_metrics({"b": 2})
    log.log_figure(object(), "x.html")


# --- stage_run ----------------------------------------------------------------

def test_stage_run_disabled_is_noop(mlflow_utils, fake_mlflow):
    cfg = {"mlflow": {"enabled": False}}
    with mlflow_utils.stage_run(cfg, "ph02_summarization") as ml:
        ml.log_metrics({"a": 1})
    assert fake_mlflow.runs_started == []              # never touched mlflow
    assert fake_mlflow.metrics == {}


def test_stage_run_missing_mlflow_is_noop(mlflow_utils, no_mlflow):
    with mlflow_utils.stage_run({}, "ph02_summarization") as ml:
        ml.log_metrics({"a": 1})                       # body still runs, no crash


def test_stage_run_standalone_opens_own_run(mlflow_utils, fake_mlflow):
    cfg = {"mlflow": {"enabled": True, "experiment": "/Users/me/PH"}}
    with mlflow_utils.stage_run(cfg, "ph03_reranking") as ml:
        ml.log_params({"model": "ce"})
        ml.log_metrics({"wall_clock_s": 1.0})
    assert len(fake_mlflow.runs_started) == 1
    assert fake_mlflow.runs_started[0]["run_name"] == "ph03_reranking"
    assert fake_mlflow.experiment == "/Users/me/PH"
    assert fake_mlflow.params == {"ph03_model": "ce"}
    assert fake_mlflow.metrics == {"ph03_wall_clock_s": 1.0}


def test_stage_run_tag_from_run_name(mlflow_utils, fake_mlflow):
    with mlflow_utils.stage_run({}, "ph04_gbm_inference") as ml:
        ml.log_metrics({"n": 3})
    assert "ph04_n" in fake_mlflow.metrics


# --- pipeline_run + stage_run = ONE run for the whole pipeline ----------------

def test_pipeline_run_is_single_run_for_all_stages(mlflow_utils, fake_mlflow):
    cfg = {"mlflow": {"enabled": True}}
    with mlflow_utils.pipeline_run(cfg):
        with mlflow_utils.stage_run(cfg, "ph01_problem_health") as ml:
            ml.log_metrics({"wall_clock_s": 1.0})
            ml.log_params({"model": "minilm"})
        with mlflow_utils.stage_run(cfg, "ph03_reranking") as ml:
            ml.log_metrics({**mlflow_utils.topk_metrics({1: 0.4, 5: 0.8})})
            ml.log_params({"model": "cross-encoder"})

    # exactly one run, and stage keys coexist without collision
    assert len(fake_mlflow.runs_started) == 1
    assert fake_mlflow.runs_started[0]["run_name"] == "problem_health_pipeline"
    assert fake_mlflow.params == {"ph01_model": "minilm", "ph03_model": "cross-encoder"}
    assert fake_mlflow.metrics == {
        "ph01_wall_clock_s": 1.0, "ph03_top_1_accuracy": 0.4, "ph03_top_5_accuracy": 0.8}


def test_pipeline_run_disabled_starts_nothing(mlflow_utils, fake_mlflow):
    with mlflow_utils.pipeline_run({"mlflow": {"enabled": False}}):
        pass
    assert fake_mlflow.runs_started == []


def test_pipeline_run_missing_mlflow_then_stages_solo(mlflow_utils, no_mlflow):
    # mlflow import fails: pipeline_run is a no-op and stages just skip logging
    with mlflow_utils.pipeline_run({}):
        with mlflow_utils.stage_run({}, "ph05_clustering") as ml:
            ml.log_metrics({"silhouette": 0.3})        # no crash, nothing logged
