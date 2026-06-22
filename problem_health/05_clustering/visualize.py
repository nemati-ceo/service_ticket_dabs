"""visualize.py — 2D UMAP projection and an interactive cluster scatter saved to Volume."""

import os


def project_2d(embeddings, params):
    """UMAP 2D projection for plotting."""
    import umap
    return umap.UMAP(**params).fit_transform(embeddings)


def scatter_html(df, proj_2d, color_col, hover_cols, out_path, title="Clusters"):
    """Plotly scatter of the 2D projection, written to out_path as HTML and PNG."""
    import plotly.express as px
    viz = df.copy()
    viz["x"], viz["y"] = proj_2d[:, 0], proj_2d[:, 1]
    fig = px.scatter(viz, x="x", y="y", color=color_col, hover_data=hover_cols, title=title)
    fig.update_traces(marker=dict(size=6, opacity=0.75))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    base = out_path.rsplit(".", 1)[0]
    try:
        fig.write_html(f"{base}.html")
        print(f"[ph05] cluster scatter saved: {base}.html")
    except Exception as e:
        print(f"[ph05] WARNING: could not save HTML ({e})")
    try:
        fig.write_image(f"{base}.png", width=1200, height=800, scale=2)
        print(f"[ph05] cluster scatter saved: {base}.png")
    except Exception as e:
        print(f"[ph05] WARNING: could not save PNG (needs 'kaleido': {e})")
    return fig
