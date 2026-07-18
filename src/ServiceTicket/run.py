"""run.py — single entry point for ALL pipeline stages."""

import os
import sys
import importlib.util
import traceback

def _root_candidates():
    """Directories that might hold config.yml and the stage packages, best first.

    A Databricks python_file task exec()s the source with __file__ stripped, so
    __file__ raises NameError there; sys.argv[0] still carries the script path the
    task launched. Whichever of the two resolves, the bundle may be deployed with
    the code flat at ${workspace.file_path} or nested under src/ServiceTicket, so
    both shapes are probed relative to each base.
    """
    bases = []
    override = os.environ.get("SERVICE_TICKET_ROOT")
    if override:
        bases.append(override)
    try:
        bases.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    argv0 = sys.argv[0] if sys.argv else ""
    if argv0.endswith(".py"):
        bases.append(os.path.dirname(os.path.abspath(argv0)))
    bases.append(os.getcwd())

    seen, out = set(), []
    for base in bases:
        for cand in (base, os.path.join(base, "src", "ServiceTicket")):
            cand = os.path.normpath(cand)
            if cand not in seen:
                seen.add(cand)
                out.append(cand)
    return out


def _resolve_root():
    """First candidate directory that actually contains config.yml."""
    candidates = _root_candidates()
    for cand in candidates:
        if os.path.isfile(os.path.join(cand, "config.yml")):
            return cand
    raise FileNotFoundError(
        "config.yml not found. Looked in:\n  " + "\n  ".join(candidates) +
        "\nSet SERVICE_TICKET_ROOT to the directory holding run.py and config.yml.")


ROOT = _resolve_root()
print(f"[run] ROOT={ROOT}")

STAGE00_DIR = os.path.join(ROOT, "00_input_sync")
STAGE01_DIR = os.path.join(ROOT, "01_problem_health")
STAGE01B_DIR = os.path.join(ROOT, "01b_pii_redaction")
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

    # Redirect HF downloads to the NM Artifactory proxy (public huggingface.co is
    # firewalled in the redzone). Applies process-wide to every SentenceTransformer/
    # CrossEncoder/hf_hub_download call — must be set before any of them run.
    endpoint = sec.get("hf_endpoint")
    if endpoint:
        os.environ.setdefault("HF_ENDPOINT", endpoint)
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        print("[auth] HF_ENDPOINT set to Artifactory proxy.")


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


def stage00(config_path=None):
    """Run stage 00 — input sync. MERGE full refine snapshot -> consume mirror.

    Best-effort: a sync failure should not silently corrupt data, so it raises and
    stops the pipeline (a bad mirror would poison every downstream stage).
    """
    cfg = load_config(config_path)
    spark = get_spark()
    spec = importlib.util.spec_from_file_location("ph00_sync", os.path.join(STAGE00_DIR, "sync.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ph00_sync"] = mod
    spec.loader.exec_module(mod)
    result = mod.run_input_sync(spark, cfg)
    print("[run] STAGE 00 SUCCESS")
    return result


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


def stage01b(config_path=None):
    """Run stage 01b — PII redaction. Returns the redacted row count.

    Raises on failure instead of returning None: stage 02 sends text to an external
    LLM endpoint, so a silent redaction failure would leak PII off-cluster. A broken
    redaction MUST stop the pipeline.
    """
    cfg = load_config(config_path)
    spark = get_spark()
    pl = _import_pipeline(STAGE01B_DIR)
    result = pl.run_pii_redaction(spark, cfg)
    print("[run] STAGE 01b SUCCESS")
    return result


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
    """Run stage 04. mode: train -> fit and save a GBM. mode: production -> score."""
    try:
        cfg = load_config(config_path)
        spark = get_spark()
        pl = _import_pipeline(STAGE04_DIR)
        result = pl.run_gbm(spark, cfg)
        print("[run] STAGE 04 SUCCESS")
        return result
    except Exception as e:
        print("=" * 60)
        print(f"[run] STAGE 04 FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("=" * 60)
        _log_failure(config_path, "ph04_gbm", e)
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
    mode = (load_config(config_path).get("mode") or "production").lower()
    print("#" * 60)
    print(f"# MODE: {mode.upper()}")
    print("# STAGE 00 — Input Sync (refine snapshot -> consume mirror, full copy)")
    print("#" * 60)
    stage00(config_path)

    print("\n" + "#" * 60)
    print("# STAGE 01 — Problem Health")
    print("#" * 60)
    r1 = stage01(config_path)
    if not isinstance(r1, tuple) or r1[0] is None:
        print("[run] Stage 01 did not produce output — skipping stages 02-05.")
        return None, None
    df_incidents, problem_health = r1

    print("\n" + "#" * 60)
    print("# STAGE 01b — PII Redaction (before ANY text leaves the cluster)")
    print("#" * 60)
    stage01b(config_path)

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

    # TEMP GATE: stop after 03 for Databricks verify. Delete to re-enable 04-05.
    print("\n[run] TEMP GATE: stopping after stage 03 (04-05 disabled).")
    return df_incidents, problem_health

    if not isinstance(r3, tuple) or r3[0] is None:
        print("[run] Stage 03 did not produce output — skipping stage 04.")
    else:
        print("\n" + "#" * 60)
        print(f"# STAGE 04 — Gradient Boosting ({mode})")
        print("#" * 60)
        stage04(config_path)

    print("\n" + "#" * 60)
    print("# STAGE 05 — Clustering")
    print("#" * 60)
    stage05(config_path)
    return df_incidents, problem_health


if __name__ == "__main__":
    main()
