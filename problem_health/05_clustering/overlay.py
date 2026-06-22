"""overlay.py — per-theme incident counts and dominant categorical values."""

import pandas as pd


def theme_overlay(df, cat_cols, theme_col="theme_group", count_col="incident_count"):
    """One row per theme: incident_count + top value/share for each categorical column."""
    rows = []
    for theme, sub in df[df[theme_col] != -1].groupby(theme_col):
        row = {theme_col: theme, count_col: len(sub)}
        for col in cat_cols:
            top = sub[col].value_counts().head(1)
            row[f"top_{col}"] = top.index[0] if len(top) else "N/A"
            row[f"top_{col}_pct"] = round(top.iloc[0] / len(sub) * 100, 1) if len(top) else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(count_col, ascending=False)
