"""run.py — single entry point for ALL pipeline stages."""

import os
import sys
import importlib.util
import traceback

try:
    ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # __file__ is undefined only when run.py's source is pasted into a cell.
    # Prefer an env override; fall back to the default deploy location.
    ROOT = os.environ.get(
        "PROBLEM_HEALTH_ROOT",
        "/Workspace/Users/nancyhuang@northwesternmutual.com/TCS/script")

STAGE01_DIR = os.path.join(ROOT, "01_problem_health")
STAGE02_DIR = os.path.join(ROOT, "02_llm_summarization")
STAGE03_DIR = os.path.join(ROOT, "03_cross_encoder_rerank")
STAGE04_DIR = os.path.join(ROOT, "04_gradient_boost_inference")
STAGE05_DIR = os.path.join(ROOT, "05_clustering")


def load_config(config_path=None):
    """Load the shared root config.yml."""
    import yaml
    path = config_path or os.path.join(ROOT, "config.yml")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    print(f"[config] loaded from {path}")
    _bootstrap_secrets(cfg)
    return cfg


_SECRETS_DONE = False


def _bootstrap_secrets(cfg):
    """Export external API tokens from the Databricks secret scope into the env ONCE.

    Production firewalls anonymous HuggingFace downloads, so every
    SentenceTransformer/CrossEncoder/hf_hub_download call must be authenticated.
    Setting HF_TOKEN here (before any stage downloads a model) covers all of them
    without threading a token through each loader. Idempotent and best-effort:
    runs once, and never breaks a run if secrets/dbutils are absent (local dev).
    """
    global _SECRETS_DONE
    if _SECRETS_DONE:
        return
    _SECRETS_DONE = True
    sec = (cfg or {}).get("secrets") or {}
    scope = sec.get("scope")
    if not scope:
        return
    try:
        token = dbutils.secrets.get(scope, sec.get("hf_token_key", "hf-token"))
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)  # older hub versions
        print(f"[auth] HF token loaded from secret scope '{scope}'.")
    except Exception as e:
        print(f"[auth] WARNING: could not load HF token ({e}); model downloads may fail in prod.")


def load_mlflow_utils():
    """Load the shared root-level mlflow_utils.py (best-effort logging helpers)."""
    spec = importlib.util.spec_from_file_location(
        "mlflow_utils", os.path.join(ROOT, "mlflow_utils.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _log_failure(config_path, run_name, exc):
    """Record a stage crash as a nested FAILED MLflow run (best-effort, never raises).

    The stages log at the END of their work, so a crash mid-stage never reaches
    their own `stage_run`. This catches that case from run.py's except blocks so a
    failed stage shows up in MLflow (with traceback) instead of silently missing.
    """
    try:
        cfg = load_config(config_path)
        load_mlflow_utils().log_stage_failure(cfg, run_name, exc)
    except Exception:
        pass


def get_spark():
    """Return active Spark session (Databricks) or create one."""
    try:
        return spark
    except NameError:
        from pyspark.sql import SparkSession
        return SparkSession.builder.getOrCreate()


def _import_pipeline(stage_dir):
    """Load a stage's pipeline.py FRESH, with its own helper modules.

    Stage folders share helper names (e.g. every stage has `evaluate`), and a
    notebook keeps the Python process alive across reruns. So we (1) put THIS
    stage_dir first on sys.path and (2) purge the stage's own modules from
    sys.modules before importing — otherwise `import evaluate`/`features`/...
    resolves to another stage's cached copy, or to stale code from a prior run.
    """
    if stage_dir in sys.path:
        sys.path.remove(stage_dir)
    sys.path.insert(0, stage_dir)

    for fn in os.listdir(stage_dir):
        if fn.endswith(".py") and fn != "run.py":
            sys.modules.pop(fn[:-3], None)

    mod_name = "pipeline_" + os.path.basename(stage_dir)
    sys.modules.pop(mod_name, None)
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(stage_dir, "pipeline.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def stage01(config_path=None):
    """Run stage 01 — Problem Health scoring. Returns (incidents, problems)."""
    try:
        cfg = load_config(config_path)
        spark = get_spark()
        pl = _import_pipeline(STAGE01_DIR)
        result = pl.run_problem_health(spark, cfg)
        print("[run] STAGE 01 SUCCESS")
        return result
    except Exception as e:
        print("=" * 60)
        print(f"[run] STAGE 01 FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("=" * 60)
        _log_failure(config_path, "ph01_problem_health", e)
        return None, None


def stage02(config_path=None):
    """Run stage 02 — LLM summarization. Returns (problems_total, incidents_total)."""
    try:
        cfg = load_config(config_path)
        spark = get_spark()
        pl = _import_pipeline(STAGE02_DIR)
        result = pl.run_summarization(spark, cfg)
        print("[run] STAGE 02 SUCCESS")
        return result
    except Exception as e:
        print("=" * 60)
        print(f"[run] STAGE 02 FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("=" * 60)
        _log_failure(config_path, "ph02_summarization", e)
        return None


def stage03(config_path=None):
    """Run stage 03 — Cross-encoder reranking. Returns (raw_scores, sigmoid_scores)."""
    try:
        cfg = load_config(config_path)
        spark = get_spark()
        pl = _import_pipeline(STAGE03_DIR)
        result = pl.run_reranking(spark, cfg)
        print("[run] STAGE 03 SUCCESS")
        return result
    except Exception as e:
        print("=" * 60)
        print(f"[run] STAGE 03 FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("=" * 60)
        _log_failure(config_path, "ph03_reranking", e)
        return None, None


def stage04(config_path=None):
    """Run stage 04 — Gradient Boosting inference. Returns the linking dataframe."""
    try:
        cfg = load_config(config_path)
        spark = get_spark()
        pl = _import_pipeline(STAGE04_DIR)
        result = pl.run_gbm_inference(spark, cfg)
        print("[run] STAGE 04 SUCCESS")
        return result
    except Exception as e:
        print("=" * 60)
        print(f"[run] STAGE 04 FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("=" * 60)
        _log_failure(config_path, "ph04_gbm_inference", e)
        return None


def stage05(config_path=None):
    """Run stage 05 — Clustering / theme grouping. Returns (df, overlay_df)."""
    try:
        cfg = load_config(config_path)
        spark = get_spark()
        pl = _import_pipeline(STAGE05_DIR)
        result = pl.run_clustering(spark, cfg)
        print("[run] STAGE 05 SUCCESS")
        return result
    except Exception as e:
        print("=" * 60)
        print(f"[run] STAGE 05 FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("=" * 60)
        _log_failure(config_path, "ph05_clustering", e)
        return None, None


def main(config_path=None):
    """Run the FULL pipeline: stage 01 -> stage 02 -> stage 03 -> stage 04 -> stage 05.

    Opens ONE MLflow run for the whole pipeline; every stage logs into it (keys are
    stage-namespaced, e.g. ph01_*, ph03_top_5_accuracy). Best-effort — if MLflow is
    off/unavailable the stages still run and just skip logging.
    """
    cfg = load_config(config_path)
    with load_mlflow_utils().pipeline_run(cfg):
        return _run_all_stages(config_path)


def _run_all_stages(config_path=None):
    print("#" * 60)
    print("# STAGE 01 — Problem Health")
    print("#" * 60)
    r1 = stage01(config_path)
    if not isinstance(r1, tuple) or r1[0] is None:
        print("[run] Stage 01 did not produce output — skipping stages 02-05.")
        return None, None
    df_incidents, problem_health = r1

    print("\n" + "#" * 60)
    print("# STAGE 02 — LLM Summarization")
    print("#" * 60)
    r2 = stage02(config_path)
    if not isinstance(r2, tuple) or r2[0] is None:
        print("[run] Stage 02 did not produce output — skipping stage 03.")
        return df_incidents, problem_health

    print("\n" + "#" * 60)
    print("# STAGE 03 — Cross-encoder Reranking")
    print("#" * 60)
    r3 = stage03(config_path)
    if not isinstance(r3, tuple) or r3[0] is None:
        print("[run] Stage 03 did not produce output — skipping stage 04.")
    else:
        print("\n" + "#" * 60)
        print("# STAGE 04 — Gradient Boosting Inference")
        print("#" * 60)
        stage04(config_path)

    print("\n" + "#" * 60)
    print("# STAGE 05 — Clustering")
    print("#" * 60)
    stage05(config_path)
    return df_incidents, problem_health


if __name__ == "__main__":
    main()
