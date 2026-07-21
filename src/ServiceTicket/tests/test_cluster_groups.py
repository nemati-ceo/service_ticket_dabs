"""Stage 05 — per-assignment-group clustering.

Clustering is partitioned: each assignment group is reduced, clustered and merged on its
own, so a theme never mixes tickets from two groups and cluster ids restart per group.
Production has ONE group of 10 tickets today, so every branch here is exercised with
synthetic groups — a big group that really clusters, a tiny one that must stand alone,
and the null-group_col path that has to keep behaving exactly as before.
"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE05 = os.path.join(ROOT, "05_clustering")
sys.path.insert(0, STAGE05)
sys.path.append(ROOT)

spec = importlib.util.spec_from_file_location("ph05_pipeline_groups",
                                              os.path.join(STAGE05, "pipeline.py"))
p5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p5)

CC = {"min_cluster_rows": 15, "merge_threshold": 0.9,
      "umap_params": {"n_neighbors": 5, "n_components": 2, "random_state": 42},
      "hdbscan_params": {"min_cluster_size": 3, "min_samples": 2}}


def _frame(groups):
    """groups: {name: n_rows} -> a frame with number + assignment_group."""
    rows = [(f"INC{i:04d}", name)
            for name, n in groups.items() for i in range(n)]
    return pd.DataFrame(rows, columns=["number", "assignment_group"])


# --- group key normalization ---------------------------------------------------

def test_group_key_is_forced_to_clean_string():
    s = pd.Series(["ITSM.Field Career.T2", "  ITSM.Field Career.T2  ", 1.0])
    out = p5._normalize_groups(s)
    assert list(out) == ["ITSM.Field Career.T2", "ITSM.Field Career.T2", "1.0"]
    # padded and unpadded must land in the SAME group, not two
    assert out.nunique() == 2


def test_missing_group_keys_collapse_to_unknown():
    s = pd.Series([None, np.nan, "", "  ", "nan", "None", "Real"])
    assert list(p5._normalize_groups(s)) == (["Unknown"] * 6) + ["Real"]


# --- splitting -----------------------------------------------------------------

def test_split_groups_covers_every_row_exactly_once():
    df = _frame({"A": 4, "B": 3, "C": 1})
    groups = p5._split_groups(df, "assignment_group")
    assert [name for name, _ in groups] == ["A", "B", "C"]      # sorted, deterministic
    pos = np.concatenate([p for _, p in groups])
    assert sorted(pos.tolist()) == list(range(len(df)))
    for name, p in groups:
        assert set(df.iloc[p]["assignment_group"]) == {name}


def test_no_group_col_is_a_single_pass_over_everything():
    df = _frame({"A": 4, "B": 3})
    groups = p5._split_groups(df, None)
    assert len(groups) == 1
    name, pos = groups[0]
    assert name is None and pos.tolist() == list(range(7))


# --- per-group clustering ------------------------------------------------------

def test_small_group_stands_alone_without_clustering():
    df = _frame({"A": 6})
    emb = np.random.default_rng(0).normal(size=(6, 4))
    out, stats, merges = p5._cluster_group(df, emb, CC, "A")
    assert set(out["cluster"]) == {-1}
    assert set(out["theme_group"]) == {-1}
    assert set(out["cluster_status"]) == {p5.STANDALONE}
    assert stats == {"group": "A", "rows": 6, "status": p5.STANDALONE, "n_clusters": 0,
                     "n_noise": 6, "noise_pct": 100.0, "silhouette": None,
                     "n_themes": 0, "n_merges": 0}
    assert merges == []


def test_min_rows_is_raised_to_clear_umap_n_neighbors():
    """UMAP errors when n_neighbors >= n_rows. A config floor below n_neighbors + 1 must
    not be taken literally, or the group crashes instead of standing alone."""
    cc = {**CC, "min_cluster_rows": 4, "umap_params": {"n_neighbors": 15}}
    df = _frame({"A": 10})
    emb = np.random.default_rng(0).normal(size=(10, 4))
    _out, stats, _m = p5._cluster_group(df, emb, cc, "A")
    assert stats["status"] == p5.STANDALONE      # 10 < 15 + 1, despite min_cluster_rows=4


def test_big_group_clusters_and_ids_are_group_local():
    pytest.importorskip("umap")
    pytest.importorskip("hdbscan")
    rng = np.random.default_rng(7)
    # two tight, well-separated blobs per group
    emb = np.vstack([rng.normal(0, 0.05, size=(15, 8)), rng.normal(5, 0.05, size=(15, 8))])
    df = _frame({"A": 30})
    out, stats, _m = p5._cluster_group(df, emb, CC, "A")
    assert stats["status"] == p5.CLUSTERED
    assert stats["n_clusters"] >= 1
    assert out["cluster"].min() >= -1
    assert len(out) == 30


# --- rollups -------------------------------------------------------------------

def _stats(group, rows, clusters, noise, themes, sil=None, status=None):
    return {"group": group, "rows": rows, "status": status or p5.CLUSTERED,
            "n_clusters": clusters, "n_noise": noise,
            "noise_pct": noise / rows * 100, "silhouette": sil,
            "n_themes": themes, "n_merges": 0}


def test_rollup_sums_groups_and_weights_noise_by_rows():
    rows = [_stats("A", 90, 3, 9, 2), _stats("B", 10, 0, 10, 0, status=p5.STANDALONE)]
    r = p5._rollup(rows)
    assert r["n_groups"] == 2
    assert r["groups_clustered"] == 1 and r["groups_too_small"] == 1
    assert r["total_clusters"] == 3 and r["total_themes"] == 2
    # 19 noise of 100 rows — NOT the 55% a plain mean of the two percentages would give
    assert r["n_noise"] == 19
    assert round(r["noise_pct"], 2) == 19.0


def test_rollup_silhouette_is_row_weighted_and_skips_missing():
    rows = [_stats("A", 90, 3, 0, 3, sil=0.5), _stats("B", 10, 0, 10, 0, sil=None,
                                                      status=p5.STANDALONE)]
    r = p5._rollup(rows)
    assert r["silhouette"] == 0.5      # group B has none; it must not count as 0.0


def test_rollup_with_no_silhouette_anywhere_is_none():
    r = p5._rollup([_stats("A", 5, 0, 5, 0, status=p5.STANDALONE)])
    assert r["silhouette"] is None


# --- overlay -------------------------------------------------------------------

def _overlay_frame():
    return pd.DataFrame({
        "number": [f"INC{i}" for i in range(6)],
        "assignment_group": ["A", "A", "A", "B", "B", "B"],
        "theme_group": [0, 0, -1, 0, 0, 1],
        "business_service": ["Email", "Email", "Email", "VPN", "VPN", "VPN"],
    })


def test_overlay_is_reported_per_group():
    out = p5._build_overlay(_overlay_frame(), ["business_service"], "assignment_group")
    assert list(out.columns)[0] == "assignment_group"
    # theme 0 exists in BOTH groups and must stay two separate rows
    theme0 = out[out.theme_group == 0]
    assert sorted(theme0["assignment_group"]) == ["A", "B"]
    assert sorted(theme0["incident_count"]) == [2, 2]
    assert set(out[out.assignment_group == "A"]["top_business_service"]) == {"Email"}


def test_overlay_without_group_col_keeps_the_old_shape():
    out = p5._build_overlay(_overlay_frame(), ["business_service"], None)
    assert "assignment_group" not in out.columns
    # themes pooled across groups: theme 0 is 4 tickets, theme 1 is 1
    assert sorted(out["incident_count"]) == [1, 4]


def test_overlay_survives_a_group_with_no_themes():
    """A stand-alone group is all noise, so it contributes no overlay rows — it must not
    take the other groups' rows down with it."""
    df = _overlay_frame()
    df.loc[df.assignment_group == "B", "theme_group"] = -1
    out = p5._build_overlay(df, ["business_service"], "assignment_group")
    assert list(out["assignment_group"]) == ["A"]


# --- writing -------------------------------------------------------------------

class _Writer:
    def __init__(self, sink):
        self.sink, self.opts = sink, {}

    def format(self, _):
        return self

    def option(self, k, v):
        self.opts[k] = v
        return self

    def mode(self, m):
        self.opts["mode"] = m
        return self

    def saveAsTable(self, table):
        self.sink[table] = self.opts


class _Spark:
    """createDataFrame(pdf) for real frames, createDataFrame([], schema=...) for empty."""

    def __init__(self):
        self.saved, self.schemas = {}, {}

    def createDataFrame(self, data, schema=None):
        self.last_schema = schema
        df = type("SDF", (), {})()
        df.write = _Writer(self.saved)
        return df


def test_run_clustering_end_to_end_keeps_every_row_with_its_own_group(monkeypatch):
    """Whole-stage wiring with the heavy parts faked: two clusterable groups and one
    stand-alone group must come back as ONE frame, each row still carrying its own
    group's labels, and the plot must see embeddings aligned to that frame."""
    pytest.importorskip("sklearn")     # cluster_stats
    # INTERLEAVED on purpose: rows do not arrive grouped, so the per-group embedding
    # slices are scattered and have to be realigned to the regrouped frame afterwards.
    order = [g for i in range(20) for g in (["B", "A"] + (["C"] if i < 5 else []))]
    df_in = pd.DataFrame({"assignment_group": order})
    df_in["number"] = [f"{g}{i}" for i, g in enumerate(df_in.assignment_group)]
    df_in["summary_final"] = "text " + df_in["number"]
    df_in["business_service"] = "Email"
    # embedding per row = a group-specific offset, so nothing can silently cross groups
    offsets = {"A": 0.0, "B": 10.0, "C": 20.0}
    emb = np.array([[offsets[g], float(i)] for i, g in enumerate(df_in.assignment_group)])

    monkeypatch.setattr(p5, "load_frame", lambda *a, **k: df_in.copy())
    monkeypatch.setattr(p5.cl, "embed", lambda texts, *a, **k: emb)
    monkeypatch.setattr(p5.cl, "reduce_umap", lambda e, params: np.asarray(e))
    # first half of a group -> cluster 0, second half -> cluster 1
    monkeypatch.setattr(p5.cl, "cluster_hdbscan",
                        lambda e, params: np.array([0] * (len(e) // 2) + [1] * (len(e) - len(e) // 2)))
    seen = {}
    monkeypatch.setattr(p5.vz, "project_2d", lambda e, params: np.asarray(e)[:, :2])
    monkeypatch.setattr(p5.vz, "build_scatter",
                        lambda d, proj, color, hover, title=None: seen.update(df=d, proj=proj,
                                                                              color=color))

    cfg = {"mlflow": {"enabled": False}, "run": {},
           "clustering": {**CC, "embed_model": "fake", "group_col": "assignment_group",
                          "number_col": "number", "text_col": "summary_final",
                          "cat_cols": ["business_service"], "summarize_gap": False,
                          "input_table": "cat.sch.in", "min_cluster_rows": 10,
                          "umap_params": {"n_neighbors": 5},
                          "output_table": "cat.sch.clusters",
                          "overlay_table": "cat.sch.overlay",
                          "export_cols": ["number", "assignment_group", "cluster",
                                          "theme_group", "cluster_status"]}}
    result, overlay = p5.run_clustering(_Spark(), cfg)

    assert len(result) == 45                                   # no row lost or duplicated
    assert list(result.columns) == cfg["clustering"]["export_cols"]
    per_group = result.groupby("assignment_group")
    # A and B clustered independently — ids restart, so both hold clusters 0 and 1
    assert set(per_group.get_group("A")["cluster"]) == {0, 1}
    assert set(per_group.get_group("B")["cluster"]) == {0, 1}
    # C is 5 rows, below min_cluster_rows -> stands alone, all noise
    c = per_group.get_group("C")
    assert set(c["cluster"]) == {-1} and set(c["cluster_status"]) == {p5.STANDALONE}
    assert set(per_group.get_group("A")["cluster_status"]) == {p5.CLUSTERED}
    # overlay is per group; C contributes nothing (no themes)
    assert sorted(set(overlay["assignment_group"])) == ["A", "B"]
    # the plot's projection must line up with the frame it was handed: each point's
    # embedding still carries its own group's offset
    plotted = seen["df"].reset_index(drop=True)
    assert list(seen["proj"][:, 0]) == [offsets[g] for g in plotted["assignment_group"]]
    assert seen["color"] == "theme_label"
    assert set(plotted.loc[plotted.assignment_group == "C", "theme_label"]) == {"noise"}


def test_empty_frame_is_still_written_so_stale_rows_cannot_survive():
    """Every group too small -> no themes -> empty overlay. Spark cannot infer a schema
    from an empty pandas frame, and skipping the write would leave last run's overlay
    sitting there looking current."""
    spark = _Spark()
    empty = pd.DataFrame(columns=["assignment_group", "theme_group", "incident_count"])
    rows = p5._write(spark, empty, "cat.sch.overlay", "overlay")
    assert rows == 0
    assert "cat.sch.overlay" in spark.saved                     # written, not skipped
    assert spark.saved["cat.sch.overlay"]["mode"] == "overwrite"
    assert spark.last_schema == "`assignment_group` string, `theme_group` string, `incident_count` string"
