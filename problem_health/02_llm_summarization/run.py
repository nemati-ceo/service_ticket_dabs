"""
run.py — entry point for stage 02 (LLM summarization).

Usage in a notebook:
    import sys
    sys.path.insert(0, "/Workspace/.../TCS/script/problem_health/02_llm_summarization")
    from run import main
    main()

Reads the shared root config.yml (one level up from this stage folder).
"""

import os
import sys
import traceback


def load_config(config_path=None):
    import yaml
    if config_path is None:
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            script_dir = "/Workspace/Users/nancyhuang@northwesternmutual.com/TCS/script/problem_health/02_llm_summarization"
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        # shared config.yml lives one level up (the project root)
        root_dir = os.path.dirname(script_dir)
        config_path = os.path.join(root_dir, "config.yml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    print(f"[config] loaded from {config_path}")
    return cfg


def get_spark():
    try:
        return spark  # provided in Databricks notebooks
    except NameError:
        from pyspark.sql import SparkSession
        return SparkSession.builder.getOrCreate()


def main(config_path=None):
    try:
        cfg = load_config(config_path)
        spark = get_spark()
        import pipeline
        result = pipeline.run_summarization(spark, cfg)
        print("[run] SUCCESS")
        return result
    except Exception as e:
        print("=" * 60)
        print(f"[run] STAGE 02 FAILED: {type(e).__name__}: {e}")
        print("-" * 60)
        traceback.print_exc()
        print("=" * 60)
        return None


if __name__ == "__main__":
    main()
