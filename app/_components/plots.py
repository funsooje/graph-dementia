# app/_components/plots.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import geopandas as gpd
from sklearn.decomposition import PCA

# ---------- 2D index scatter (no edges) ----------
def plot_zip_scatter(out_df: pd.DataFrame,
                              x_col: str = "environment_index",
                              y_col: str = "ses_index",
                              comm_col: str = "zip_community",
                              size_col: str = "zip_pagerank",
                              title: str = "ZIP communities in index space"):
    """
    Scatter of ZIPs in (environment_index, ses_index) colored by community.
    Node size optionally scaled by size_col (e.g., PageRank).
    """
    X = out_df[[x_col, y_col]].to_numpy()
    comm = out_df[comm_col].to_numpy() if comm_col in out_df else np.zeros(len(out_df))
    if size_col in out_df:
        s = out_df[size_col].to_numpy()
        s_min, s_max = np.quantile(s, [0.05, 0.95])
        s_range = max(s_max - s_min, 1e-12)
        sizes = 2.0 + 100.0 * (np.clip(s, s_min, s_max) - s_min) / s_range
    else:
        sizes = np.full(len(out_df), 30.0)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(X[:, 0], X[:, 1], c=comm, s=sizes, cmap="tab20", edgecolors="black", linewidths=0.2)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    # cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    # cbar.set_label("community id")
    return fig

# ---------- NetworkX graph plot ----------
def plot_networkx_graph(
    G: nx.Graph,
    out_df: pd.DataFrame | None = None,
    layout: str = "spring",
    node_size: int = 30,                # fallback if no size_col
    edge_width: float = 0.6,
    edge_alpha: float = 0.35,
    edge_color: str = "gray",
    show_node_labels: bool = False,
    show_edge_labels: bool = False,
    community_col: str = "zip_community",
    title: str = "Graph",
    size_col: str | None = None,        # <-- NEW
    size_min_px: float = 2.0,           # <-- NEW (matches your scatter defaults)
    size_max_px: float = 100.0,         # <-- NEW
    scale_factor: float = 5.0
):
    """
    Minimal, reliable NetworkX graph plot:
    - edges first (so they are visible)
    - fixed node size by default (node_size)
    - optional coloring by community if present in out_df
    - optional sizing by a column (size_col) using 5–95% quantile clipping
    """

    # Choose layout
    layouts = {
        "spring": nx.spring_layout(G, seed=42),
        "kamada": nx.kamada_kawai_layout(G),
        "circular": nx.circular_layout(G),
        "random": nx.random_layout(G),
        "shell": nx.shell_layout(G),
    }
    pos = layouts.get(layout, layouts["spring"])

    # Node colors
    if out_df is not None and community_col in out_df.columns:
        node_colors = out_df[community_col].to_numpy()
        cmap = "tab20"
    else:
        node_colors = "lightblue"
        cmap = None

    # Node sizes (column-based if provided)
    if out_df is not None and size_col and size_col in out_df.columns:
        try:
            order = list(G.nodes())
            s = out_df.iloc[order][size_col].to_numpy(dtype=float)
        except Exception:
            s = out_df[size_col].to_numpy(dtype=float)

        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        if s.size:
            q5, q95 = np.quantile(s, [0.05, 0.95])
            rng = max(q95 - q5, 1e-12)
            s_clipped = np.clip(s, q5, q95)
            sizes = node_size * (1.0 + (scale_factor - 1.0) * (s_clipped - q5) / rng)
        else:
            sizes = np.full(G.number_of_nodes(), float(node_size))
    else:
        sizes = np.full(G.number_of_nodes(), float(node_size))

    # Labels (optional)
    labels = None
    if show_node_labels and out_df is not None and "ZIPCODE" in out_df.columns:
        labels = {i: str(out_df.iloc[i]["ZIPCODE"]) for i in G.nodes()}

    edge_labels = None
    if show_edge_labels:
        edge_labels = {(i, j): d.get("weight", 1.0) for i, j, d in G.edges(data=True)}

    # Figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # --- Draw edges first ---
    if G.number_of_edges() > 0:
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color=edge_color,
            width=edge_width,
            alpha=edge_alpha,
        )

    # --- Draw nodes on top ---
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=sizes,
        node_color=node_colors,
        cmap=cmap,
        edgecolors="black",
        linewidths=0.2,
    )

    if labels:
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=6, ax=ax)

    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, ax=ax)

    ax.set_title(title)
    ax.set_axis_off()
    return fig

# ---------- Geographic plot ----------
def plot_geographic_communities(
    community_df: pd.DataFrame,
    zip_coordinates: pd.DataFrame,
    washington_boundary,
    comm_col: str = "zip_community",
    zip_col_df: str = "ZIPCODE",
    zip_col_xy: str = "zip",
    isolated_node_color: str = "lightgray",
    community_color_map: dict = None,
    base_markersize: float = 30.0,
    scale_factor: float = 4.0,
    size_col: str | None = None,
):
    """
    Plot ZIP communities on a geographic map (requires lon/lat for ZIPs and a boundary GeoDataFrame).
    - zip_coordinates must have columns [zip, lat, lng] (strings for zip).
    - If size_col is provided and exists in community_df, scale point sizes by its values.
    """

    # --- Prep community + coords ---
    df = community_df.copy()
    df[zip_col_df] = df[zip_col_df].astype(str)
    coords = zip_coordinates.copy()
    coords[zip_col_xy] = coords[zip_col_xy].astype(str)

    merged = df.merge(coords, left_on=zip_col_df, right_on=zip_col_xy, how="left")
    merged = merged.dropna(subset=["lat", "lng"])
    merged["lat"] = pd.to_numeric(merged["lat"], errors="coerce")
    merged["lng"] = pd.to_numeric(merged["lng"], errors="coerce")
    merged = merged.dropna(subset=["lat", "lng"])

    gdf = gpd.GeoDataFrame(
        merged,
        geometry=gpd.points_from_xy(merged["lng"], merged["lat"]),
        crs="EPSG:4326"
    )

    # --- Marker sizes ---
    if size_col and size_col in gdf.columns:
        s = gdf[size_col].to_numpy()
        s_min, s_max = np.quantile(s, [0.05, 0.95])
        s_range = max(s_max - s_min, 1e-12)
        sizes = base_markersize * (1.0 + (scale_factor - 1.0) * (np.clip(s, s_min, s_max) - s_min) / s_range)
    else:
        sizes = np.full(len(gdf), base_markersize)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 8))
    washington_boundary.plot(ax=ax, color="lightgray", edgecolor="black")

    if community_color_map:
        gdf["__color__"] = gdf[comm_col].map(community_color_map).fillna(isolated_node_color)
        gdf.plot(ax=ax, color=gdf["__color__"], markersize=sizes, legend=False)
    else:
        gdf.plot(ax=ax, column=comm_col, cmap="tab20", markersize=sizes, legend=False)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("ZIP communities (geographic)")
    ax.grid(True, alpha=0.3)
    return fig

# def plot_profile_scatter_embed(
#     X: np.ndarray,
#     features_df,
#     community_col: str = "profile_community",
#     size_col: str = "profile_count",
#     title: str = "Profiles (PCA 2D)",
#     scale_min_px: float = 8.0,
#     scale_max_px: float = 120.0,
# ):
#     """
#     2D PCA projection of X (n x d). Colors by community, sizes by profile_count (quantile-scaled).
#     """
#     pca = PCA(n_components=2, random_state=42)
#     X2 = pca.fit_transform(X)

#     if size_col in features_df.columns:
#         s = features_df[size_col].to_numpy(dtype=float)
#         s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
#         q5, q95 = np.quantile(s, [0.05, 0.95]) if s.size else (0.0, 1.0)
#         rng = max(q95 - q5, 1e-12)
#         s_clipped = np.clip(s, q5, q95)
#         sizes = scale_min_px + (scale_max_px - scale_min_px) * (s_clipped - q5) / rng
#     else:
#         sizes = np.full(len(X2), 30.0)

#     if community_col in features_df.columns:
#         colors = features_df[community_col].to_numpy()
#         cmap = "tab20"
#     else:
#         colors, cmap = "lightblue", None

#     fig, ax = plt.subplots(figsize=(8, 6))
#     ax.scatter(X2[:, 0], X2[:, 1], s=sizes, c=colors, cmap=cmap, alpha=0.75, linewidths=0.2, edgecolors="black")
#     ax.set_title(title)
#     ax.set_xlabel("PC1")
#     ax.set_ylabel("PC2")
#     ax.grid(True, alpha=0.3)
#     return fig

def plot_profile_scatter_embed(
    X_weighted: np.ndarray,
    out_df: pd.DataFrame,
    community_col: str = "profile_community",
    size_col: str | None = "profile_count",
    color_by: str | None = None,     # NEW
    title: str = "Profiles (PCA 2D)",
    max_cats_legend: int = 20,
):
    """
    Projects X_weighted to 2D via PCA and scatters points.
    - size_col: controls marker sizes (quantile-scaled)
    - color_by: column in out_df for coloring (categorical or numeric)
        • categorical (<= max_cats_legend unique): discrete colors + legend
        • numeric: continuous colormap
        • None: fallback to community_col
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    n = X_weighted.shape[0]
    pca = PCA(n_components=2, random_state=42)
    XY = pca.fit_transform(X_weighted)

    # --- sizes (same scaling you use elsewhere) ---
    if size_col and size_col in out_df.columns:
        s = pd.to_numeric(out_df[size_col], errors="coerce").fillna(0.0).to_numpy()
        q5, q95 = np.quantile(s, [0.05, 0.95]) if s.size else (0.0, 1.0)
        rng = max(q95 - q5, 1e-12)
        s_clipped = np.clip(s, q5, q95)
        sizes = 12.0 + (90.0) * (s_clipped - q5) / rng
    else:
        sizes = np.full(n, 30.0)

    # --- choose color column ---
    col = color_by if color_by else community_col
    fig, ax = plt.subplots(figsize=(10, 7))

    if col in out_df.columns:
        series = out_df[col]
        # Detect numeric vs categorical
        if pd.api.types.is_numeric_dtype(series):
            c = pd.to_numeric(series, errors="coerce").fillna(series.median())
            sc = ax.scatter(XY[:, 0], XY[:, 1], s=sizes, c=c, cmap="viridis", alpha=0.75, edgecolors="none")
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(col)
        else:
            # Treat as categorical-like
            cats = series.astype("string").fillna("Unknown")
            uniq = cats.unique().tolist()
            # cap legend size for readability
            if len(uniq) > max_cats_legend:
                # map to integer codes and color by codes (no legend)
                code_map = {u: i for i, u in enumerate(sorted(uniq))}
                codes = cats.map(code_map).astype(int).to_numpy()
                sc = ax.scatter(XY[:, 0], XY[:, 1], s=sizes, c=codes, cmap="tab20", alpha=0.75, edgecolors="none")
            else:
                # draw per-category with legend
                cmap = plt.get_cmap("tab20")
                for i, u in enumerate(sorted(uniq)):
                    mask = (cats == u).to_numpy()
                    ax.scatter(XY[mask, 0], XY[mask, 1], s=sizes[mask], alpha=0.75,
                               edgecolors="none", color=cmap(i % 20), label=str(u))
                ax.legend(title=col, loc="best", fontsize=8, ncol=1)
    else:
        # fallback: color by community
        if community_col in out_df.columns:
            cats = out_df[community_col].astype("string").fillna("Unknown")
            uniq = sorted(cats.unique().tolist())
            cmap = plt.get_cmap("tab20")
            for i, u in enumerate(uniq):
                mask = (cats == u).to_numpy()
                ax.scatter(XY[mask, 0], XY[mask, 1], s=sizes[mask],
                           alpha=0.75, edgecolors="none", color=cmap(i % 20), label=str(u))
            ax.legend(title=community_col, loc="best", fontsize=8, ncol=1)
        else:
            ax.scatter(XY[:, 0], XY[:, 1], s=sizes, alpha=0.75, edgecolors="none", color="steelblue")

    ax.set_title(title + f"\nPCA var exp: {pca.explained_variance_ratio_[0]:.2f}, {pca.explained_variance_ratio_[1]:.2f}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.2)
    return fig