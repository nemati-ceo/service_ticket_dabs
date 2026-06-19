"""
run.py — entry point for ProblemHealth 01.

Usage (Databricks notebook or job):
    %run ./run.py
or:
    from run import main; main()
"""

import os
import sys
import yaml


def load_config(config_path=None):
    """Resolve script dir, load config.yml."""
    if config_path is None:
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            # Databricks notebook fallback
            script_dir = "/Workspace/Users/nancyhuang@northwesternmutual.com/TCS/script"
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        config_path = os.path.join(script_dir, "config.yml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_spark():
    try:
        return spark  # provided in Databricks
    except NameError:
        from pyspark.sql import SparkSession
        return SparkSession.builder.getOrCreate()


def main(config_path=None):
    cfg = load_config(config_path)
    spark = get_spark()
    import pipeline
    return pipeline.run_problem_health(spark, cfg)


if __name__ == "__main__":
    df_incidentscore, problem_health = main()
