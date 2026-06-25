"""Tests for mlflow_utils — the shared best-effort MLflow logging layer.

Covers: the enabled toggle, topk/baseline/step-timing helpers, key/artifact/tag
namespacing, the no-op paths (disabled / mlflow missing), best-effort error
swallowing, the run-context stamping (tags + config snapshot + system metrics),
and the core contract that the FULL pipeline opens ONE parent run with a NESTED
child per stage, while a stage run on its own opens a single top-level run.
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


def test_baseline_delta_metrics(mlflow_utils):
    topk = {5: 0.72, 10: 0.81}
    baselines = {"PH02": {5: 0.70, 10: 0.80, 1: 0.40}}
    out = mlflow_utils.baseline_delta_metrics(topk, baselines)
    assert out["baseline_PH02_top_5"] == 0.70
    assert out["baseline_PH02_top_1"] == 0.40            # baseline logged even if k not measured
    assert round(out["delta_PH02_top_5"], 4) == 0.02      # measured - baseline
    assert "delta_PH02_top_1" not in out                  # k=1 wasn't measured


def test_baseline_delta_metrics_handles_empty(mlflow_utils):
    assert mlflow_utils.baseline_delta_metrics(None, None) == {}
    assert mlflow_utils.baseline_delta_metrics({1: 0.4}, {}) == {}


def test_step_timings_sanitizes_labels(mlflow_utils):
    out = mlflow_utils.step_timings([("[1/8] load", 1.5), ("[2/8] incremental", 0.5)])
    assert out == {"step_1_8_load_s": 1.5, "step_2_8_incremental_s": 0.5}


def test_step_timings_handles_empty(mlflow_utils):
    assert mlflow_utils.step_timings(None) == {}
    assert mlflow_utils.step_timings([]) == {}


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


def test_logger_tags_namespaced_and_drops_none(mlflow_utils, fake_mlflow):
    mlflow_utils._Logger(fake_mlflow, "ph05").set_tags({"input": "t", "skip": None})
    assert fake_mlflow.tags == {"ph05_input": "t"}


def test_logger_artifacts_namespaced(mlflow_utils, fake_mlflow):
    log = mlflow_utils._Logger(fake_mlflow, "ph05")
    log.log_dict({"a": 1}, "merge_log.json")
    log.log_text("SELECT 1", "input.sql")
    log.log_table({"k": [1]}, "eval.json")
    assert fake_mlflow.dicts == {"ph05/merge_log.json": {"a": 1}}
    assert fake_mlflow.texts == {"ph05/input.sql": "SELECT 1"}
    assert fake_mlflow.tables == {"ph05/eval.json": {"k": [1]}}


def test_logger_noop_when_mlflow_none(mlflow_utils):
    log = mlflow_utils._Logger(None, "ph02")          # must not raise
    log.log_params({"a": 1})
    log.log_metrics({"b": 2})
    log.set_tags({"c": 3})
    log.log_figure(object(), "x.html")
    log.log_dict({"d": 4}, "x.json")
    log.log_text("t", "x.txt")
    log.log_table({"e": [5]}, "x.json")


def test_logger_swallows_backend_errors(mlflow_utils, fake_mlflow):
    fake_mlflow.fail_on_log = True
    log = mlflow_utils._Logger(fake_mlflow, "ph04")
    log.log_params({"a": 1})                           # backend raises -> swallowed
    log.log_metrics({"b": 2})
    log.set_tags({"c": 3})
    log.log_figure(object(), "x.html")
    log.log_dict({"d": 4}, "x.json")
    log.log_text("t", "x.txt")
    log.log_table({"e": [5]}, "x.json")


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
    assert fake_mlflow.runs_started[0] == {"run_name": "ph03_reranking", "nested": False}
    assert fake_mlflow.experiment == "/Users/me/PH"
    assert fake_mlflow.params == {"ph03_model": "ce"}
    assert fake_mlflow.metrics == {"ph03_wall_clock_s": 1.0}
    assert fake_mlflow.tags["ph03_status"] == "ok"      # block completed cleanly
    assert "config_snapshot.yaml" in fake_mlflow.dicts  # solo run still stamps context


def test_stage_run_marks_failed_and_reraises(mlflow_utils, fake_mlflow):
    import pytest
    with pytest.raises(ValueError):
        with mlflow_utils.stage_run({}, "ph02_summarization") as ml:
            ml.log_metrics({"a": 1})
            raise ValueError("boom")
    assert fake_mlflow.tags["ph02_status"] == "failed"


def test_stage_run_tag_from_run_name(mlflow_utils, fake_mlflow):
    with mlflow_utils.stage_run({}, "ph04_gbm_inference") as ml:
        ml.log_metrics({"n": 3})
    assert "ph04_n" in fake_mlflow.metrics


# --- pipeline_run + stage_run = ONE parent run, a NESTED child per stage -------

def test_pipeline_run_nests_each_stage(mlflow_utils, fake_mlflow):
    cfg = {"mlflow": {"enabled": True}}
    with mlflow_utils.pipeline_run(cfg):
        with mlflow_utils.stage_run(cfg, "ph01_problem_health") as ml:
            ml.log_metrics({"wall_clock_s": 1.0})
            ml.log_params({"model": "minilm"})
        with mlflow_utils.stage_run(cfg, "ph03_reranking") as ml:
            ml.log_metrics({**mlflow_utils.topk_metrics({1: 0.4, 5: 0.8})})
            ml.log_params({"model": "cross-encoder"})

    # one top-level parent run, then a nested child per stage
    assert len(fake_mlflow.runs_started) == 3
    assert fake_mlflow.runs_started[0] == {"run_name": "problem_health_pipeline", "nested": False}
    assert fake_mlflow.runs_started[1] == {"run_name": "ph01_problem_health", "nested": True}
    assert fake_mlflow.runs_started[2] == {"run_name": "ph03_reranking", "nested": True}
    # stage keys still coexist without collision
    assert fake_mlflow.params == {"ph01_model": "minilm", "ph03_model": "cross-encoder"}
    assert fake_mlflow.metrics == {
        "ph01_wall_clock_s": 1.0, "ph03_top_1_accuracy": 0.4, "ph03_top_5_accuracy": 0.8}


def test_pipeline_run_stamps_context(mlflow_utils, fake_mlflow):
    with mlflow_utils.pipeline_run({"mlflow": {"enabled": True}}):
        pass
    assert fake_mlflow.system_metrics is True               # CPU/GPU/mem sampling on
    assert fake_mlflow.dicts.get("config_snapshot.yaml") is not None
    assert fake_mlflow.tags.get("pipeline") == "problem_health"
    assert fake_mlflow.tags.get("run_mode") == "full"       # no run.limit -> full run


def test_pipeline_run_respects_tracking_uri_and_test_mode(mlflow_utils, fake_mlflow):
    cfg = {"mlflow": {"enabled": True, "tracking_uri": "databricks",
                      "log_system_metrics": False},
           "run": {"limit": 50}}
    with mlflow_utils.pipeline_run(cfg):
        pass
    assert fake_mlflow.tracking_uri == "databricks"
    assert fake_mlflow.system_metrics is False              # opt-out honored
    assert fake_mlflow.tags.get("run_mode") == "test"
    assert fake_mlflow.tags.get("run_limit") == "50"


def test_pipeline_run_disabled_starts_nothing(mlflow_utils, fake_mlflow):
    with mlflow_utils.pipeline_run({"mlflow": {"enabled": False}}):
        pass
    assert fake_mlflow.runs_started == []


def test_pipeline_run_missing_mlflow_then_stages_solo(mlflow_utils, no_mlflow):
    # mlflow import fails: pipeline_run is a no-op and stages just skip logging
    with mlflow_utils.pipeline_run({}):
        with mlflow_utils.stage_run({}, "ph05_clustering") as ml:
            ml.log_metrics({"silhouette": 0.3})        # no crash, nothing logged


# --- log_stage_failure --------------------------------------------------------

def test_log_stage_failure_records_nested_failed_run(mlflow_utils, fake_mlflow):
    with mlflow_utils.pipeline_run({"mlflow": {"enabled": True}}):
        try:
            raise RuntimeError("kaboom")
        except RuntimeError as e:
            mlflow_utils.log_stage_failure({"mlflow": {"enabled": True}},
                                           "ph03_reranking", e)

    # a nested run was opened for the failed stage and ended FAILED
    assert {"run_name": "ph03_reranking", "nested": True} in fake_mlflow.runs_started
    assert "FAILED" in fake_mlflow.ended_status
    assert fake_mlflow.tags.get("ph03_status") == "failed"
    assert "RuntimeError: kaboom" in fake_mlflow.tags.get("ph03_error", "")
    assert "ph03/traceback.txt" in fake_mlflow.texts


def test_log_stage_failure_disabled_is_noop(mlflow_utils, fake_mlflow):
    mlflow_utils.log_stage_failure({"mlflow": {"enabled": False}},
                                   "ph03_reranking", RuntimeError("x"))
    assert fake_mlflow.runs_started == []
