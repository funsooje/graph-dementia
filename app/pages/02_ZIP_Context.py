# app/pages/02_ZIP_Context.py
import os
from pathlib import Path
from io import BytesIO

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

import networkx as nx
import community as community_louvain  # python-louvain
import matplotlib.pyplot as plt

from app._components.plots import (
    plot_zip_scatter,            # returns a Matplotlib fig
    plot_networkx_graph,         # returns a Matplotlib fig (network)
    plot_geographic_communities, # keep commented until geo fixes are ready
)

# ---------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------
st.set_page_config(page_title="ZIP Context", layout="wide")
st.title("02 - ZIP Context Mini-Analysis")

# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
zipc = st.session_state.get("zip_df")
zip_coords = st.session_state.get("zip_coords")
wa_boundary = st.session_state.get("wa_boundary")

if zipc is None or zip_coords is None or wa_boundary is None:
    st.error("Required dataframes not found in session_state. Ensure Home.py loads them at startup.")
    st.stop()

zipc = zipc.reset_index(drop=True)

# ---------------------------------------------------------------------
# Sidebar controls (form with two buttons)
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    with st.form("zip_graph_controls", clear_on_submit=False):
        graph_choice_label = st.selectbox(
            "Graph type",
            ["All features", "Environmental only", "Sociodemographic only"],
            index={"all": 0, "env": 1, "soc": 2}.get(st.session_state.get("graph_mode", "all"), 0),
        )
        layout_choice = st.selectbox(
            "Layout",
            ["spring", "kamada", "circular", "random", "shell"],
            index=["spring", "kamada", "circular", "random", "shell"].index(
                st.session_state.get("graph_layout", "spring")
            ),
        )
        k = st.number_input(
            "k (k-NN)",
            min_value=1, max_value=20, value=int(st.session_state.get("graph_k", 3)), step=1
        )
        knn_type_label = st.selectbox(
            "k-NN graph type",
            ["Mutual k-NN (undirected)", "Directed k-NN"],
            index={"mutual":0, "directed":1}.get(st.session_state.get("knn_type","mutual"), 0),
        )

        colb1, colb2 = st.columns(2)
        with colb1:
            update_plots_clicked = st.form_submit_button("Load Plots")
        with colb2:
            recompute_clicked = st.form_submit_button("Recompute")

        set_zip_clicked = st.form_submit_button("Set Current as ZIP features")

# Normalize selected mode for downstream code
GRAPH_MODES = {
    "All features": "all",
    "Environmental only": "env",
    "Sociodemographic only": "soc",
}
graph_mode = GRAPH_MODES[graph_choice_label]

# Persist current selections so they’re remembered on rerun
st.session_state["graph_mode"] = graph_mode
st.session_state["graph_layout"] = layout_choice
st.session_state["graph_k"] = int(k)
st.session_state["knn_type"] = "mutual" if knn_type_label.startswith("Mutual") else "directed"

# ---------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------
ENV_NORM = [
    "EPL_OZONE","EPL_PM","EPL_DSLPM","EPL_NPL","EPL_TRI","EPL_TSD","EPL_RMP","EPL_COAL",
    "EPL_LEAD","EPL_PARK","EPL_HOUAGE","EPL_WLKIND","EPL_ROAD","EPL_RAIL","EPL_AIRPRT",
    "EPL_IMPWTR"
]
SOC_NORM = [
    "EPL_MINRTY","EPL_POV200","EPL_NOHSDP","EPL_UNEMP","EPL_RENTER","EPL_HOUBDN",
    "EPL_UNINSUR","EPL_NOINT","EPL_AGE17","EPL_AGE65","EPL_DISABL","EPL_LIMENG",
    "EPL_MOBILE","EPL_GROUPQ"
]

def present(df: pd.DataFrame, cols: list) -> list:
    return [c for c in cols if c in df.columns]

def pca_first_component(df: pd.DataFrame, cols: list):
    cols = present(df, cols)
    if not cols:
        return None, None, []
    X = df[cols].astype(float).values
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=1, random_state=42)
    pc1 = pca.fit_transform(X).ravel()
    return pc1, float(pca.explained_variance_ratio_[0]), cols

def mutual_knn_graph(features: np.ndarray, k_neighbors: int) -> nx.Graph:
    sim = cosine_similarity(features)
    np.fill_diagonal(sim, -np.inf)
    n = sim.shape[0]

    topk = []
    for i in range(n):
        idx = np.argpartition(sim[i], -k_neighbors)[-k_neighbors:]
        idx = idx[np.argsort(sim[i, idx])[::-1]]
        topk.append(set(idx.tolist()))

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in topk[i]:
            if i < j and i in topk[j]:
                w = float(sim[i, j])
                if np.isfinite(w):
                    G.add_edge(i, j, weight=w)
    return G


def build_knn_graph(features: np.ndarray, k_neighbors: int, knn_type: str):
    """
    knn_type: "mutual" -> undirected mutual k-NN
              "directed" -> directed k-NN (i -> top-k neighbors of i)
    """
    sim = cosine_similarity(features)
    np.fill_diagonal(sim, -np.inf)
    n = sim.shape[0]

    # top-k neighbor sets per node
    topk = []
    for i in range(n):
        idx = np.argpartition(sim[i], -k_neighbors)[-k_neighbors:]
        idx = idx[np.argsort(sim[i, idx])[::-1]]
        topk.append(idx.tolist())

    if knn_type == "directed":
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in topk[i]:
                w = float(sim[i, j])
                if np.isfinite(w):
                    G.add_edge(i, j, weight=w)
        return G

    # mutual (intersection) undirected
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        nbrs_i = set(topk[i])
        for j in nbrs_i:
            if i < j and i in set(topk[j]):
                w = float(sim[i, j])
                if np.isfinite(w):
                    G.add_edge(i, j, weight=w)
    return G


def compute_graph_metrics(G):
    """
    - Communities (Louvain) on an undirected view
    - Betweenness: directed if DiGraph, undirected otherwise
    - PageRank: directed if DiGraph, undirected otherwise
    """
    if G.is_directed():
        G_u = G.to_undirected(reciprocal=False)
        partition = community_louvain.best_partition(G_u, weight="weight", random_state=42)
        btw = nx.betweenness_centrality(G, weight="weight", normalized=True)
        pr = nx.pagerank(G, alpha=0.85, weight="weight")
    else:
        partition = community_louvain.best_partition(G, weight="weight", random_state=42)
        btw = nx.betweenness_centrality(G, weight="weight", normalized=True)
        pr = nx.pagerank(G, alpha=0.85, weight="weight")
    return partition, btw, pr

# ---------------------------------------------------------------------
# Disk figure cache helpers
# ---------------------------------------------------------------------
CACHE_DIR = Path("data/cache/zip_figs")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def fig_to_png_file(fig, path: Path, dpi: int = 110) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return str(path)

# ---------------------------------------------------------------------
# Indices (compute once and reuse)
# ---------------------------------------------------------------------
if "zip_indices" not in st.session_state:
    env_idx, env_var, _ = pca_first_component(zipc, ENV_NORM)
    ses_idx, ses_var, _ = pca_first_component(zipc, SOC_NORM)
    st.session_state["zip_indices"] = {
        "environment_index": env_idx,
        "ses_index": ses_idx,
        "env_var": env_var,
        "ses_var": ses_var,
    }
indices = st.session_state["zip_indices"]
env_var = indices["env_var"]
ses_var = indices["ses_var"]

# ---------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------
# Graph cache: {(mode, k): {"graph": G, "features": out}}
if "zip_graph_cache" not in st.session_state:
    st.session_state["zip_graph_cache"] = {}
graph_cache = st.session_state["zip_graph_cache"]

# Figure cache (paths): {"network": {(mode,k,layout): {...}}, "scatter": {(mode,): {...}}}
if "zip_fig_cache" not in st.session_state:
    st.session_state["zip_fig_cache"] = {"network": {}, "scatter": {}, "geo": {}}
fig_cache = st.session_state["zip_fig_cache"]

knn_type = st.session_state["knn_type"]  # "mutual" or "directed"
graph_key   = (graph_mode, int(k), knn_type)
network_key = (graph_mode, int(k), knn_type, st.session_state["graph_layout"])
scatter_key = (graph_mode,)  # unchanged: scatter depends only on feature set
geo_key     = (graph_mode, int(k), knn_type)  # depends on graph, not layout

# ---------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------
if recompute_clicked:
    # Build feature matrix based on selected mode
    if graph_mode == "all":
        feats = np.hstack([
            StandardScaler().fit_transform(zipc[present(zipc, ENV_NORM)]),
            StandardScaler().fit_transform(zipc[present(zipc, SOC_NORM)])
        ])
    elif graph_mode == "env":
        feats = StandardScaler().fit_transform(zipc[present(zipc, ENV_NORM)])
    else:  # "soc"
        feats = StandardScaler().fit_transform(zipc[present(zipc, SOC_NORM)])

    # Graph + metrics
    G = build_knn_graph(feats, k_neighbors=int(k), knn_type=knn_type)
    partition, betweenness, pagerank = compute_graph_metrics(G)

    # Features table (always include indices)
    n = len(zipc)
    out = pd.DataFrame({
        "ZIPCODE": zipc["ZIPCODE"].astype(str).values,
        "environment_index": indices["environment_index"],
        "ses_index": indices["ses_index"],
        "zip_community": [partition.get(i, -1) for i in range(n)],
        "zip_betweenness": [betweenness.get(i, np.nan) for i in range(n)],
        "zip_pagerank": [pagerank.get(i, np.nan) for i in range(n)],
    })

    # Save graph data
    graph_cache[graph_key] = {"graph": G, "features": out}

    # Render and cache Network plot (by mode,k,layout)
    net_path = CACHE_DIR / f"net_{graph_mode}_k{int(k)}_{knn_type}_{st.session_state['graph_layout']}.png"
    fig_net = plot_networkx_graph(
        G,
        out_df=out,
        node_size=20,
        edge_width=0.9,
        edge_alpha=0.5,
        edge_color="gray",
        community_col="zip_community",
        size_col="zip_pagerank",
        title=f"Network plot: {graph_choice_label} (k={int(k)}, {knn_type}, layout={st.session_state['graph_layout']})",
        layout=st.session_state["graph_layout"],
        scale_factor = 4.0
    )
    fig_cache["network"][network_key] = {
        "network_png_path": fig_to_png_file(fig_net, net_path)
    }

    # Render and cache PCA scatter once per feature set (mode)
    if scatter_key not in fig_cache["scatter"]:
        scatter_path = CACHE_DIR / f"scatter_{graph_mode}.png"
        fig_scatter = plot_zip_scatter(out)
        fig_cache["scatter"][scatter_key] = {
            "scatter_png_path": fig_to_png_file(fig_scatter, scatter_path)
        }

    # --- Geographic plot (cache by mode,k,knn_type) ---
    geo_path = CACHE_DIR / f"geo_{graph_mode}_k{int(k)}_{knn_type}.png"
    fig_geo = plot_geographic_communities(
        out,            
        zip_coords,     
        wa_boundary,
        size_col="zip_pagerank",
        base_markersize=30
    )
    fig_cache["geo"][geo_key] = {
        "geo_png_path": fig_to_png_file(fig_geo, geo_path)
    }

    st.success("Graph recomputed and plots updated.")

elif update_plots_clicked:
    # Only re-render PNGs if graph data exists for this (mode, k)
    if graph_key in graph_cache:
        G = graph_cache[graph_key]["graph"]
        out = graph_cache[graph_key]["features"]

        # Regenerate Network plot for current layout from cached graph
        net_path = CACHE_DIR / f"net_{graph_mode}_k{int(k)}_{knn_type}_{st.session_state['graph_layout']}.png"
        fig_net = plot_networkx_graph(
            G,
            out_df=out,
            node_size=30,
            edge_width=0.9,
            edge_alpha=0.5,
            edge_color="gray",
            community_col="zip_community",
            title=f"Network plot: {graph_choice_label} (k={int(k)}, {knn_type}, layout={st.session_state['graph_layout']})",
            layout=st.session_state["graph_layout"],
        )
        fig_cache["network"][network_key] = {
            "network_png_path": fig_to_png_file(fig_net, net_path)
        }

        # Ensure scatter exists for this feature set (mode); create once if missing
        if scatter_key not in fig_cache["scatter"]:
            scatter_path = CACHE_DIR / f"scatter_{graph_mode}.png"
            fig_scatter = plot_zip_scatter(out)
            fig_cache["scatter"][scatter_key] = {
                "scatter_png_path": fig_to_png_file(fig_scatter, scatter_path)
            }

        # Ensure geographic plot exists for (mode,k,knn_type); create once if missing
        if geo_key not in fig_cache["geo"]:
            geo_path = CACHE_DIR / f"geo_{graph_mode}_k{int(k)}_{knn_type}.png"
            fig_geo = plot_geographic_communities(out, zip_coords, wa_boundary)
            fig_cache["geo"][geo_key] = {
                "geo_png_path": fig_to_png_file(fig_geo, geo_path)
            }
            
        st.success("Plots updated from cached graph.")
    else:
        st.info("No cached graph for this configuration. Click 'Recompute graph' first.")

# ---------------------------------------------------------------------
# Persist current graph outputs as canonical zip_features
# ---------------------------------------------------------------------
if 'set_zip_clicked' in locals() and set_zip_clicked:
    # Save the CURRENT settings' outputs if present in cache
    if graph_key in graph_cache:
        out = graph_cache[graph_key]["features"].copy()

        # --- One-hot expansion preview for zip_community ---
        onehot_preview = {}
        if "zip_community" in out.columns:
            # count distinct valid communities; exclude unknown sentinel (-1) if present
            vals = pd.Series(out["zip_community"].unique())
            valid = vals[vals >= 0]
            n_added = int(valid.nunique())
            onehot_preview["zip_community"] = n_added
            st.info(f"One-hot preview: zip_community would add {n_added} column(s) (excluding 'unknown').")
        else:
            st.info("One-hot preview: zip_community not found in current outputs.")

        # Ensure required columns exist
        needed = {"ZIPCODE", "environment_index", "ses_index", "zip_community", "zip_betweenness", "zip_pagerank"}
        missing = needed.difference(out.columns)
        if missing:
            st.warning(f"Cannot set zip_features: missing columns {sorted(missing)}. Recompute first.")
        else:
            st.session_state["zip_features"] = out
            st.session_state["zip_features_meta"] = {
                "mode": graph_mode,
                "k": int(k),
                "knn_type": st.session_state["knn_type"],
                "layout": st.session_state["graph_layout"],
                "env_var": env_var,
                "ses_var": ses_var,
                "onehot_preview": onehot_preview,
            }
            st.success("Saved current outputs as session_state['zip_features'].")
    else:
        st.warning("No cached graph for these settings. Click 'Recompute graph' first.")

# ---------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------
if graph_key in graph_cache:
    G = graph_cache[graph_key]["graph"]
    out = graph_cache[graph_key]["features"]

    # Status: which ZIP features are currently active for fusion page?
    if "zip_features" in st.session_state and "zip_features_meta" in st.session_state:
        meta = st.session_state["zip_features_meta"]
        st.info(
            f"Active ZIP features set from: mode='{meta['mode']}', k={meta['k']}, "
            f"knn_type='{meta['knn_type']}', layout='{meta['layout']}'."
        )

    st.subheader("Derived ZIP features (head)")
    st.dataframe(out.head(12), width='content')

    st.subheader("PCA variance explained")
    st.write({
        "environment_index_variance_ratio": env_var,
        "ses_index_variance_ratio": ses_var
    })

    st.subheader("Graph summary")
    connected_flag = nx.is_weakly_connected(G) if G.is_directed() else nx.is_connected(G)

    st.write({
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "k": int(k),
        "is_connected": connected_flag,
        "mode": graph_mode,
        "layout": st.session_state["graph_layout"],
        "graph_cache_hit": True,
        "network_fig_cache_hit": network_key in fig_cache["network"],
        "scatter_fig_cache_hit": scatter_key in fig_cache["scatter"],
    })

    # PCA scatter (cached by feature set only)
    st.subheader("PCA indices scatter (cached image)")
    scatter_png_path = None
    if scatter_key in fig_cache["scatter"]:
        scatter_png_path = fig_cache["scatter"][scatter_key]["scatter_png_path"]
    if scatter_png_path and os.path.exists(scatter_png_path):
        st.image(scatter_png_path, width='content')
    else:
        st.info("Scatter image not cached yet. Click 'Recompute graph' or 'Update plots'.")

    # Network graph (cached by mode, k, layout)
    st.subheader("Network graph view (cached image)")
    if network_key in fig_cache["network"]:
        net_png_path = fig_cache["network"][network_key]["network_png_path"]
        if os.path.exists(net_png_path):
            st.image(net_png_path, width='content')
        else:
            st.info("Network image missing on disk. Click 'Update plots'.")
    else:
        st.info("Network image not cached yet for this layout. Click 'Update plots'.")

    # Geographic map 
    st.subheader("Geographic map of communities (cached image)")
    if geo_key in fig_cache["geo"]:
        geo_png_path = fig_cache["geo"][geo_key].get("geo_png_path")
        if geo_png_path and os.path.exists(geo_png_path):
            st.image(geo_png_path, width='content')
        else:
            st.info("Geographic image missing on disk. Click 'Update plots' or 'Recompute graph'.")
    else:
        st.info("Geographic plot not cached yet. Click 'Update plots' or 'Recompute graph'.")

else:
    st.info("No cached graph for this configuration yet. Choose settings and click 'Recompute graph'.")

    # Still show PCA indices preview (computed once)
    preview = pd.DataFrame({
        "ZIPCODE": zipc["ZIPCODE"].astype(str).values,
        "environment_index": indices["environment_index"],
        "ses_index": indices["ses_index"],
    }).head(12)
    st.subheader("PCA indices (preview)")
    st.dataframe(preview, width='content')