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

from app._components.zip_context_utils import (
    present,
    pca_first_component,
    build_knn_graph,
    compute_graph_metrics,
    compute_adaptive_pca_indices,
    assemble_zip_features,
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


# --- Feature Groups: Load default from session_state or disk, merge with custom ---
import json
default_fg = None
if "default_feature_groups" in st.session_state:
    default_fg = st.session_state["default_feature_groups"]
else:
    # fallback: try to load from disk (as in 02_FeatureGroups.py)
    default_fg_path = Path("data/feature_groups_default.json")
    if default_fg_path.exists():
        with open(default_fg_path, "r") as f:
            default_fg = json.load(f)
    else:
        st.error("Default feature groups not found in session_state or on disk. Please run 02_FeatureGroups.py first.")
        st.stop()

# Ensure keys/values are as expected (dict of str:list-of-str)
if not isinstance(default_fg, dict):
    st.error("Default feature groups are not a dictionary. Please check 02_FeatureGroups.py output.")
    st.stop()

DEFAULT_FEATURE_GROUPS = default_fg
custom_feature_groups = st.session_state.get("feature_groups", {})
if not isinstance(custom_feature_groups, dict):
    custom_feature_groups = {}
feature_groups = {**DEFAULT_FEATURE_GROUPS, **custom_feature_groups}

# Sidebar controls (feature group selection)
with st.sidebar:
    st.header("Settings")
    with st.form("zip_graph_controls", clear_on_submit=False):
        group_names = list(feature_groups.keys())
        # Try to select previously used group if present, else default to "All features"
        prev_group = st.session_state.get("selected_feature_group", "All features")
        group_index = group_names.index(prev_group) if prev_group in group_names else 0
        selected_group_name = st.selectbox(
            "Feature group",
            group_names,
            index=group_index,
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
        # colb1, colb2 = st.columns(2)
        # with colb1:
        #     update_plots_clicked = st.form_submit_button("Load Plots")
        # with colb2:
        recompute_clicked = st.form_submit_button("Recompute Table and Graphs")
        set_zip_clicked = st.form_submit_button("Set Current as ZIP features")

# Persist selections
st.session_state["selected_feature_group"] = selected_group_name
st.session_state["graph_layout"] = layout_choice
st.session_state["graph_k"] = int(k)
st.session_state["knn_type"] = "mutual" if knn_type_label.startswith("Mutual") else "directed"

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
# Indices (computed adaptively per selected feature group)
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------
# Graph cache: {(group_name, k, knn_type): {"graph": G, "features": out}}
if "zip_graph_cache" not in st.session_state:
    st.session_state["zip_graph_cache"] = {}
graph_cache = st.session_state["zip_graph_cache"]

# Figure cache (paths): {"network": {(group_name,k,layout): {...}}, "scatter": {(group_name,): {...}}}
if "zip_fig_cache" not in st.session_state:
    st.session_state["zip_fig_cache"] = {"network": {}, "scatter": {}, "geo": {}}
fig_cache = st.session_state["zip_fig_cache"]

knn_type = st.session_state["knn_type"]  # "mutual" or "directed"
selected_group_key = (selected_group_name, int(k), knn_type)
network_key = (selected_group_name, int(k), knn_type, st.session_state["graph_layout"])
scatter_key = (selected_group_name,)  # scatter depends only on feature set
geo_key     = (selected_group_name, int(k), knn_type)

# ---------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------

def get_group_columns(group_data):
    """Extract columns from a feature group (handles dict or list)."""
    if isinstance(group_data, dict) and "columns" in group_data:
        return group_data["columns"]
    return group_data

# Helper: get columns from DEFAULT_FEATURE_GROUPS by key, handling dict/list
def get_cols_from_default(group_name):
    group = DEFAULT_FEATURE_GROUPS.get(group_name, [])
    if isinstance(group, dict) and "columns" in group:
        return group["columns"]
    return group

selected_features = present(zipc, get_group_columns(feature_groups[selected_group_name]))

if recompute_clicked:
    # --- Early validation: ensure all selected group columns exist in zipc ---
    req_cols = get_group_columns(feature_groups[selected_group_name])
    missing_cols = [c for c in req_cols if c not in zipc.columns]
    if missing_cols:
        st.error(f"Cannot compute graph: the following columns from the selected feature group are missing in the data: {missing_cols}")
        st.stop()

    # Build feature matrix based on selected group
    if selected_features:
        feats = StandardScaler().fit_transform(zipc[selected_features])
    else:
        feats = np.zeros((len(zipc), 1))
    # Graph + metrics
    G = build_knn_graph(feats, k_neighbors=int(k), knn_type=knn_type)
    partition, betweenness, pagerank = compute_graph_metrics(G)

    # --- PCA indices computed adaptively for this group ---
    # Determine which of the default env/ses features are present in the selected features
    env_cols_all = present(zipc, get_cols_from_default("env"))
    ses_cols_all = present(zipc, get_cols_from_default("ses"))
    selected_set = set(selected_features)
    env_cols = [c for c in env_cols_all if c in selected_set]
    ses_cols = [c for c in ses_cols_all if c in selected_set]

    environment_index, env_var, env_used = None, None, []
    ses_index, ses_var, ses_used = None, None, []
    if env_cols:
        environment_index, env_var, env_used = pca_first_component(zipc, env_cols)
    if ses_cols:
        ses_index, ses_var, ses_used = pca_first_component(zipc, ses_cols)

    # Save indices for this group in session_state (for current group only)
    st.session_state["zip_indices"] = {
        "environment_index": environment_index,
        "ses_index": ses_index,
        "env_var": env_var,
        "ses_var": ses_var,
        "env_cols": env_used,
        "ses_cols": ses_used,
    }

    # Features table (include computed indices)
    n = len(zipc)
    out = pd.DataFrame({
        "ZIPCODE": zipc["ZIPCODE"].astype(str).values,
        "environment_index": environment_index if environment_index is not None else [np.nan]*n,
        "ses_index": ses_index if ses_index is not None else [np.nan]*n,
        "zip_community": [partition.get(i, -1) for i in range(n)],
        "zip_betweenness": [betweenness.get(i, np.nan) for i in range(n)],
        "zip_pagerank": [pagerank.get(i, np.nan) for i in range(n)],
    })
    # Optionally, also include explained variance
    out["environment_index_var"] = env_var if env_var is not None else np.nan
    out["ses_index_var"] = ses_var if ses_var is not None else np.nan

    # Save graph data
    graph_cache[selected_group_key] = {"graph": G, "features": out}

    # Render and cache Network plot (by group,k,layout)
    net_path = CACHE_DIR / f"net_{selected_group_name}_k{int(k)}_{knn_type}_{st.session_state['graph_layout']}.png"
    fig_net = plot_networkx_graph(
        G,
        out_df=out,
        node_size=20,
        edge_width=0.9,
        edge_alpha=0.5,
        edge_color="gray",
        community_col="zip_community",
        size_col="zip_pagerank",
        title=f"Network plot: {selected_group_name} (k={int(k)}, {knn_type}, layout={st.session_state['graph_layout']})",
        layout=st.session_state["graph_layout"],
        scale_factor = 4.0
    )
    fig_cache["network"][network_key] = {
        "network_png_path": fig_to_png_file(fig_net, net_path)
    }

    # Always regenerate and cache PCA scatter for this feature set (group)
    scatter_path = CACHE_DIR / f"scatter_{selected_group_name}.png"
    fig_scatter = plot_zip_scatter(out)
    fig_cache["scatter"][scatter_key] = {
        "scatter_png_path": fig_to_png_file(fig_scatter, scatter_path)
    }

    # --- Geographic plot (cache by group,k,knn_type) ---
    geo_path = CACHE_DIR / f"geo_{selected_group_name}_k{int(k)}_{knn_type}.png"
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


# ---------------------------------------------------------------------
# Persist current graph outputs as canonical zip_features
# ---------------------------------------------------------------------
if 'set_zip_clicked' in locals() and set_zip_clicked:
    # Save the CURRENT settings' outputs if present in cache
    if selected_group_key in graph_cache:
        out = graph_cache[selected_group_key]["features"].copy()

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
        # Get adaptive PCA explained variance from session_state["zip_indices"]
        zip_indices = st.session_state.get("zip_indices", {})
        env_var = zip_indices.get("env_var", None)
        ses_var = zip_indices.get("ses_var", None)
        if missing:
            st.warning(f"Cannot set zip_features: missing columns {sorted(missing)}. Recompute first.")
        else:
            st.session_state["zip_features"] = out
            st.session_state["zip_features_meta"] = {
                "feature_group": selected_group_name,
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
if selected_group_key in graph_cache:
    G = graph_cache[selected_group_key]["graph"]
    out = graph_cache[selected_group_key]["features"]

    # Status: which ZIP features are currently active for fusion page?
    if "zip_features" in st.session_state and "zip_features_meta" in st.session_state:
        meta = st.session_state["zip_features_meta"]
        st.info(
            f"Active ZIP features set from: feature_group='{meta.get('feature_group', '')}', k={meta['k']}, "
            f"knn_type='{meta['knn_type']}', layout='{meta['layout']}'."
        )

    st.subheader("Derived ZIP features")
    # Ensure required columns are present or computed: environment_index, ses_index, environment_index_var, ses_index_var, zip_degree, zip_betweenness, zip_pagerank, zip_community, isolated
    columns_to_include = [
        "environment_index",
        "ses_index",
        "environment_index_var",
        "ses_index_var",
        "zip_degree",
        "zip_betweenness",
        "zip_pagerank",
        "zip_community",
        "isolated",
    ]
    # Compute zip_degree if missing
    if "zip_degree" not in out.columns:
        # Degree for each node (number of neighbors)
        degree_dict = dict(G.degree())
        out["zip_degree"] = out.index.map(lambda i: degree_dict.get(i, 0))
    # Compute isolated if missing
    if "isolated" not in out.columns:
        isolate_set = set(nx.isolates(G))
        out["isolated"] = out.index.map(lambda i: i in isolate_set)
    # Add environment_index_var and ses_index_var if not present
    if "environment_index_var" not in out.columns:
        env_var = st.session_state.get("zip_indices", {}).get("env_var", np.nan)
        out["environment_index_var"] = env_var if env_var is not None else np.nan
    if "ses_index_var" not in out.columns:
        ses_var = st.session_state.get("zip_indices", {}).get("ses_var", np.nan)
        out["ses_index_var"] = ses_var if ses_var is not None else np.nan
    # Select only columns that are present in out
    display_cols = ["ZIPCODE"] + [c for c in columns_to_include if c in out.columns]
    st.dataframe(out[display_cols], use_container_width=False, hide_index=True)

    # Show PCA variance explained for this group (from session_state["zip_indices"])
    zip_indices = st.session_state.get("zip_indices", {})
    env_var = zip_indices.get("env_var", None)
    ses_var = zip_indices.get("ses_var", None)
    st.subheader("PCA variance explained")
    pca_var_df = pd.DataFrame(
        {
            "Metric": ["environment_index_variance_ratio", "ses_index_variance_ratio"],
            "Value": [env_var, ses_var],
        }
    )
    st.dataframe(pca_var_df, use_container_width=False, hide_index=True)

    st.subheader("Graph summary")
    connected_flag = nx.is_weakly_connected(G) if G.is_directed() else nx.is_connected(G)
    # number of detected communities (Louvain)
    num_communities = int(out["zip_community"].nunique()) if "zip_community" in out.columns else 0
    # number of isolated nodes
    num_isolates = sum(1 for _ in nx.isolates(G))
    # Only display selected statistics (exclude cache and status flags)
    graph_summary_dict = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "num_communities": num_communities,
        "isolated_nodes": num_isolates,
        "k": int(k),
        "is_connected": connected_flag,
        "feature_group": selected_group_name,
    }
    graph_summary_df = pd.DataFrame(
        {
            "Statistic": list(graph_summary_dict.keys()),
            "Value": list(graph_summary_dict.values()),
        }
    )
    st.dataframe(graph_summary_df, use_container_width=False, hide_index=True)

    # Community Summaries: count, percent, and mean of numeric metrics
    st.subheader("Community Summaries")
    try:
        total_zip_rows = len(out)
        # Identify numeric columns for aggregation (excluding ZIPCODE, zip_community)
        exclude_cols = {"ZIPCODE", "zip_community"}
        numeric_cols = [col for col in out.columns if pd.api.types.is_numeric_dtype(out[col]) and col not in exclude_cols]
        # Group by community
        comm_tbl = (
            out.assign(zip_community=out["zip_community"].fillna(-1))
               .groupby("zip_community", dropna=False)
        )
        # Count and percent
        zip_count = comm_tbl.size().rename("zip_count")
        pct_of_total = (zip_count / total_zip_rows).round(4).rename("pct_of_total")
        # Means of numeric metrics
        mean_metrics = comm_tbl[numeric_cols].mean().add_prefix("mean_")
        # Merge all
        summary_tbl = pd.concat([zip_count, pct_of_total, mean_metrics], axis=1).reset_index()
        summary_tbl = summary_tbl.sort_values("zip_count", ascending=False, kind="mergesort")
        st.dataframe(summary_tbl, use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Could not compute community summary table: {e}")

    # PCA scatter (cached by feature set only)
    st.subheader("PCA indices scatter (cached image)")
    scatter_png_path = None
    if scatter_key in fig_cache["scatter"]:
        scatter_png_path = fig_cache["scatter"][scatter_key]["scatter_png_path"]
    if scatter_png_path and os.path.exists(scatter_png_path):
        st.image(scatter_png_path, width='content')
    else:
        st.info("Scatter image not cached yet. Click 'Recompute graph' or 'Update plots'.")

    # Network graph (cached by group, k, layout)
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

    # Still show PCA indices preview (computed for all features at startup)
    # For preview, use all env/ses features from DEFAULT_FEATURE_GROUPS (using keys "env" and "ses")
    env_idx, env_var, _ = pca_first_component(zipc, get_cols_from_default("env"))
    ses_idx, ses_var, _ = pca_first_component(zipc, get_cols_from_default("ses"))
    preview = pd.DataFrame({
        "ZIPCODE": zipc["ZIPCODE"].astype(str).values,
        "environment_index": env_idx,
        "ses_index": ses_idx,
    }).head(12)
    st.subheader("PCA indices (preview)")
    st.dataframe(preview, width='content')