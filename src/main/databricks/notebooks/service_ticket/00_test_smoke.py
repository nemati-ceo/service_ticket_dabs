# Databricks notebook source
# Smoke test: cluster + UC access before running the real tasks.
# COMMAND ----------
print("cluster up")
spark.sql("SELECT current_catalog(), current_schema()").show()
