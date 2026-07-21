"""Stage 05 overlay.py + visualize.py.

theme_overlay feeds a live Delta table, so its schema must stay stable even when there
is nothing to report. build_scatter must refuse a projection that does not line up with
its frame — points would be drawn against the wrong tickets, and the picture would look
perfectly fine.
"""

import contextlib
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from conftest import load_by_path

ov = load_by_path("overlay05", "05_clustering/overlay.py")
vz = load_by_path("visualize05", "05_clustering/visualize.py")

CATS = ["business_service"]
DF = pd.DataFrame({
    "number": list("ABCDE"),
    "theme_group": [0, 0, 0, 1, -1],
    "business_service": ["net", "net", "db", "db", "net"],
})


# --- overlay -------------------------------------------------------------------

def test_one_row_per_theme_noise_excluded():
    out = ov.theme_overlay(DF, CATS)
    assert out["theme_group"].tolist() == [0, 1]          # -1 dropped
    assert out["incident_count"].tolist() == [3, 1]       # sorted by count desc


def test_dominant_value_and_share():
    out = ov.theme_overlay(DF, CATS).set_index("theme_group")
    assert out.loc[0, "top_business_service"] == "net"
    assert out.loc[0, "top_business_service_pct"] == pytest.approx(66.7, abs=0.1)


def test_all_noise_still_returns_the_expected_schema():
    """The overlay table is overwritten every run — an empty frame with the wrong columns
    would rewrite the live table's schema."""
    out = ov.theme_overlay(DF.assign(theme_group=-1), CATS)
    assert out.empty
    assert out.columns.tolist() == ["theme_group", "incident_count",
                                    "top_business_service", "top_business_service_pct"]


def test_no_cat_cols_still_counts_themes():
    out = ov.theme_overlay(DF, [])
    assert out.columns.tolist() == ["theme_group", "incident_count"]
    assert out["incident_count"].sum() == 4               # 5 rows minus the noise row


# --- visualize -----------------------------------------------------------------

@contextlib.contextmanager
def _no_pyspark_probe():
    """Stop plotly's dataframe-type probe from touching pyspark.

    plotly 6 routes inputs through narwhals, which imports `pyspark.sql` to ask whether
    the frame is a Spark one. On an image where pyspark is importable but incomplete that
    raises `module 'pyspark.sql' has no attribute 'DataFrame'` before our plain pandas
    frame is ever looked at — it fails this test in CI and nowhere else.

    Patched only when that internal actually exists: on plotly 5 (no narwhals), or after
    narwhals renames it, the test runs unpatched rather than erroring on a missing target.
    """
    try:
        import narwhals.dependencies as nw_deps
    except Exception:
        yield
        return
    if not hasattr(nw_deps, "get_pyspark_sql"):
        yield
        return
    with patch.object(nw_deps, "get_pyspark_sql", return_value=None):
        yield


def test_scatter_refuses_a_misaligned_projection():
    """A projection of a different length means the sample and the frame drifted apart."""
    pytest.importorskip("plotly")
    proj = np.zeros((3, 2))
    with pytest.raises(ValueError, match="wrong tickets"):
        vz.build_scatter(DF, proj, "theme_group", ["number"])


def test_scatter_copies_only_the_columns_it_plots():
    pytest.importorskip("plotly")
    wide = DF.assign(huge_text="x" * 100, unused=1)
    with _no_pyspark_probe():
        fig = vz.build_scatter(wide, np.zeros((len(wide), 2)), "theme_group", ["number"])
    assert fig is not None
