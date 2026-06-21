"""
run.py — entry point for ProblemHealth 01 (notebook or job).

Usage in a notebook:
    import sys
    sys.path.insert(0, "/Workspace/.../TCS/script/problem_health/01_problem_health")
    from run import main
    df_incidents, problem_health = main()
"""

import os
import sys
import traceback


def load_config(config_path=None):
    """Resolve script dir, load config.yml. Raises clear error if missing."""
    import yaml
    try:
        if config_path is None:
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
            except NameError:
                script_dir = "/Workspace/Users/nancyhuang@northwesternmutual.com/TCS/script/problem_health/01_problem_health"
            if script_dir not in sys.path:
                sys.path.insert(0, script_dir)
            # config.yml lives at the repo/script ROOT (one level up), shared by all stages.
            root_dir = os.path.dirname(script_dir)
            config_path = os.path.join(root_dir, "config.yml")
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        print(f"[config] loaded from {config_path}")
        return cfg
    except FileNotFoundError:
        print(f"[config] ERROR: config.yml not found at {config_path}")
        raise
    except Exception as e:
        print(f"[config] ERROR loading config: {e}")
        raise


def get_spark():
    """Return active Spark session (Databricks) or create one."""
    try:
        return spark  # provided in Databricks notebooks
    except NameError:
        from pyspark.sql import SparkSession
        return SparkSession.builder.getOrCreate()


def main(config_path=None):
    """Run the full pipeline with error handling. Returns (incidents, problems) or (None, None)."""
    try:
        cfg = load_config(config_path)
        spark = get_spark()
        import pipeline
        df_incidents, problem_health = pipeline.run_problem_health(spark, cfg)
        print("[run] SUCCESS")
        return df_incidents, problem_health
    except Exception as e:
        print("=" * 60)
        print(f"[run] PIPELINE FAILED: {type(e).__name__}: {e}")
        print("-" * 60)
        traceback.print_exc()
        print("=" * 60)
        return None, None


if __name__ == "__main__":
    df_incidents, problem_health = main()
