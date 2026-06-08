# Databricks notebook source
# Task 1: incident -> problem linking (PH pipeline: embed -> rerank -> classify).
# Reads UDP incidents/problems, writes link table back to UDP.
# TODO: port PH05 + cross-encoder logic here.
# COMMAND ----------
# config: catalog/schema per target
# COMMAND ----------
# load -> Nancy preprocessing -> embed -> rerank -> classify
# COMMAND ----------
# write results to redzone_consume.<schema>.incident_problem_links
print("incident_linking placeholder")
