"""visualize.py — 2D UMAP projection and the Plotly cluster scatter (logged via MLflow by the pipeline)."""


def project_2d(embeddings, params):
    """UMAP 2D projection for plotting."""
    import umap
    return umap.UMAP(**params).fit_transform(embeddings)


def build_scatter(df, proj_2d, color_col, hover_cols, title="Clusters"):
    """Build (don't save) a Plotly scatter of the 2D projection."""
    import plotly.express as px
    if len(df) != len(proj_2d):
        raise ValueError(f"projection has {len(proj_2d)} rows but the frame has {len(df)} "
                         f"— points would be plotted against the wrong tickets")
    # Copy only the columns the plot uses; df carries full ticket text.
    keep = [c for c in dict.fromkeys([*hover_cols, color_col]) if c in df.columns]
    viz = df[keep].copy()
    viz["x"], viz["y"] = proj_2d[:, 0], proj_2d[:, 1]
    fig = px.scatter(viz, x="x", y="y", color=color_col, hover_data=hover_cols, title=title)
    fig.update_traces(marker=dict(size=6, opacity=0.75))
    return fig
