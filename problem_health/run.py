"""
run.py — single entry point for ALL pipeline stages.

Usage in a Databricks notebook (point sys.path at the folder holding run.py):
    import sys
    sys.path.insert(0, "/Workspace/Users/nancyhuang@northwesternmutual.com/TCS/script")
    if "run" in sys.modules:          # force re-import so edits are picked up
        del sys.modules["run"]
    import run

    # full pipeline (01 -> 02 -> 03); returns stage-01's primary dataframes
    df_incidents, problem_health = run.main()

    # or run a single stage:
    run.stage01()      # Problem Health scoring   -> ph01_output_*
    run.stage02()      # LLM summarization        -> ph02_output_*
    run.stage03()      # Cross-encoder reranking  -> reranked_scores*.npy

Reads the shared config.yml in this same folder. Each stage's modules live in
its own subfolder (01_problem_health / 02_llm_summarization /
03_cross_encoder_rerank); this loader adds that subfolder to sys.path and imports
its pipeline under a unique name, so the stages' same-named `pipeline.py` files
never collide.
"""

import os
import sys
import importlib.util
import traceback

try:
    ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    ROOT = "/Workspace/Users/nancyhuang@northwesternmutual.com/TCS/script"

STAGE01_DIR = os.path.join(ROOT, "01_problem_health")
STAGE02_DIR = os.path.join(ROOT, "02_llm_summarization")
STAGE03_DIR = os.path.join(ROOT, "03_cross_encoder_rerank")


def load_config(config_path=None):
    """Load the shared root config.yml."""
    import yaml
    path = config_path or os.path.join(ROOT, "config.yml")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    print(f"[config] loaded from {path}")
    return cfg


def get_spark():
    """Return active Spark session (Databricks) or create one."""
    try:
        return spark  # provided in Databricks notebooks
    except NameError:
        from pyspark.sql import SparkSession
        return SparkSession.builder.getOrCreate()


def _import_pipeline(stage_dir):
    """Add a stage folder to sys.path and load its pipeline.py under a unique name."""
    if stage_dir not in sys.path:
        sys.path.insert(0, stage_dir)
    mod_name = "pipeline_" + os.path.basename(stage_dir)
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
        return None, None


def main(config_path=None):
    """Run the FULL pipeline: stage 01 -> stage 02 -> stage 03.

    Returns stage 01's primary business dataframes, so the call unpacks cleanly:
        df_incidents, problem_health = run.main()

    Stages 02 (summaries) and 03 (reranked scores) persist their own outputs to
    Delta tables / Volume; call run.stage02() / run.stage03() directly if you
    need their return values. A stage is skipped if a prior stage failed; on a
    stage-01 failure this returns (None, None).
    """
    print("#" * 60)
    print("# STAGE 01 — Problem Health")
    print("#" * 60)
    r1 = stage01(config_path)
    if not isinstance(r1, tuple) or r1[0] is None:
        print("[run] Stage 01 did not produce output — skipping stages 02 & 03.")
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
    stage03(config_path)
    return df_incidents, problem_health


if __name__ == "__main__":
    main()
