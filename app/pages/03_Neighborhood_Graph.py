# app/pages/03_Neighborhood_Graph.py
import os
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

import networkx as nx
import community as community_louvain  # python-louvain (exact path)
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
    process_zip_group,
    compute_adaptive_pca_indices,
)

# ---------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------
st.set_page_config(page_title="Neighborhood Graph", layout="wide")
st.title("Neighborhood Graph")

# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
from app._logic.loader import ensure_data_loaded

if not ensure_data_loaded():
    st.stop()

zipc = st.session_state.get("zip_df")
zip_coords = st.session_state.get("zip_coords")
wa_boundary = st.session_state.get("wa_boundary")

zipc = zipc.reset_index(drop=True)

# ---------------------------------------------------------------------
# Feature Groups: Load default from session_state or disk, merge with custom
# ---------------------------------------------------------------------
import json
default_fg = None
if "default_feature_groups" in st.session_state:
    default_fg = st.session_state["default_feature_groups"]
else:
    default_fg_path = Path("data/feature_groups_default.json")
    if default_fg_path.exists():
        with open(default_fg_path, "r") as f:
            default_fg = json.load(f)
    else:
        st.error("Default feature groups not found. Please visit the Neighborhood Features page first.")
        st.stop()

if not isinstance(default_fg, dict):
    st.error("Default feature groups are not a dictionary. Please check the Neighborhood Features page.")
    st.stop()

DEFAULT_FEATURE_GROUPS = default_fg
custom_feature_groups = st.session_state.get("feature_groups", {})
if not isinstance(custom_feature_groups, dict):
    custom_feature_groups = {}
feature_groups = {**DEFAULT_FEATURE_GROUPS, **custom_feature_groups}

# ---------------------------------------------------------------------
# Cache setup
# ---------------------------------------------------------------------
CACHE_DIR = Path("data/cache/zip_figs")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Import shared graph cache utilities
from app._logic.graph_cache import (
    save_graph_to_cache,
    get_cached_graph,
    clear_all_cache,
    initialize_session_cache,
)

# Initialize graph cache from disk
initialize_session_cache()

# ---------------------------------------------------------------------
# Settings (top of page)
# ---------------------------------------------------------------------
st.subheader("Graph Settings")

# --- Method selector ---
method = st.radio(
    "Compute method",
    ["Exact (NetworkX)", "Cached Index (igraph)"],
    index=0 if st.session_state.get("nbr_graph_method", "exact") == "exact" else 1,
    horizontal=True,
    help=(
        "**Exact**: cosine similarity on all pairs — fine for small datasets (<5k nodes). "
        "**Cached Index**: uses the PyNNDescent index from page 02 — required for large datasets."
    ),
)
use_igraph = (method == "Cached Index (igraph)")
st.session_state["nbr_graph_method"] = "igraph" if use_igraph else "exact"

st.markdown("---")

# --- Cached index info (igraph path only) ---
nbr_meta = st.session_state.get("nbr_index_meta")
nbr_neighbors = st.session_state.get("nbr_index_neighbors")
active_context_key = st.session_state.get("active_context_key", "default")
if use_igraph:
    if nbr_meta is None or nbr_neighbors is None:
        st.warning(
            "No cached index loaded. Go to **Build Neighborhood Index** (page 02) "
            "to build and load an index first."
        )
        st.stop()

    # Warn if the index was built for a different dataset
    index_dataset_key = nbr_meta.get("dataset_key")
    if index_dataset_key and index_dataset_key != active_context_key:
        index_label = nbr_meta.get("dataset_label", index_dataset_key)
        active_label = (
            st.session_state.get("context_datasets", {})
            .get(active_context_key, {})
            .get("label", active_context_key)
        )
        st.warning(
            f"Index mismatch: this index was built for **{index_label}** "
            f"(`{index_dataset_key}`), but the active dataset is "
            f"**{active_label}** (`{active_context_key}`). "
            "Results may be incorrect — rebuild the index for the active dataset."
        )

    st.info(
        f"Active index: **{st.session_state.get('nbr_index_name', '?')}** — "
        f"group=`{nbr_meta.get('group_name')}`, "
        f"k_max={nbr_meta.get('k_max')}, "
        f"nodes={nbr_meta.get('n_nodes', 0):,}, "
        f"dataset=`{nbr_meta.get('dataset_key', '—')}`, "
        f"built {nbr_meta.get('built_at', '?')}"
    )

# --- Feature group (exact path only) ---
group_names = list(feature_groups.keys())
prev_group = st.session_state.get("selected_feature_group", "All features")
group_index = group_names.index(prev_group) if prev_group in group_names else 0

col1, col2, col3 = st.columns(3)
with col1:
    if not use_igraph:
        selected_group_name = st.selectbox("Feature group", group_names, index=group_index)
    else:
        selected_group_name = nbr_meta.get("group_name", group_names[0])
        st.markdown(f"**Feature group:** `{selected_group_name}` *(from cached index)*")

    k_max_allowed = int(nbr_meta.get("k_max", 20)) if use_igraph else 50
    k = st.number_input(
        "k (k-NN)",
        min_value=1,
        max_value=k_max_allowed,
        value=min(int(st.session_state.get("graph_k", 3)), k_max_allowed),
        step=1,
        help=f"Max k = {k_max_allowed}" + (" (limited by cached index)" if use_igraph else ""),
    )
with col2:
    if not use_igraph:
        layout_choice = st.selectbox(
            "Layout",
            ["spring", "kamada", "circular", "random", "shell"],
            index=["spring", "kamada", "circular", "random", "shell"].index(
                st.session_state.get("graph_layout", "spring")
            ),
        )
    else:
        layout_choice = st.session_state.get("graph_layout", "spring")
        st.markdown("*Network plot not available for large graphs.*")

    knn_type_label = st.selectbox(
        "k-NN graph type",
        ["Mutual k-NN (undirected)", "Directed k-NN"],
        index={"mutual": 0, "directed": 1}.get(st.session_state.get("knn_type", "mutual"), 0),
    )
with col3:
    resolution = st.number_input(
        "Community resolution",
        min_value=0.01, max_value=100.0,
        value=float(st.session_state.get("resolution", 1.0)),
        step=0.1, format="%.2f",
        help="Higher values produce more/smaller communities (recommended: 0.1-3.0)"
    )

btn_col1, btn_col2, _ = st.columns([1, 1, 2])
with btn_col1:
    recompute_clicked = st.button("Compute Graph", type="primary")
with btn_col2:
    clear_cache_clicked = st.button("Clear Cache")

st.markdown("---")

# Handle cache clearing
if clear_cache_clicked:
    # Clear session state caches
    if "zip_fig_cache" in st.session_state:
        st.session_state["zip_fig_cache"] = {"network": {}, "scatter": {}, "geo": {}}
    # Clear all graph caches (session state + disk)
    clear_all_cache()
    # Clear figure cache on disk
    import shutil
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    st.success("Cache cleared.")
    st.rerun()

# Persist selections
st.session_state["selected_feature_group"] = selected_group_name
st.session_state["graph_layout"] = layout_choice
st.session_state["graph_k"] = int(k)
st.session_state["knn_type"] = "mutual" if knn_type_label.startswith("Mutual") else "directed"
st.session_state["resolution"] = float(resolution)

# ---------------------------------------------------------------------
# Disk figure cache helpers
# ---------------------------------------------------------------------
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
resolution_val = st.session_state["resolution"]  # community detection resolution
selected_group_key = (selected_group_name, int(k), knn_type, resolution_val)
network_key = (selected_group_name, int(k), knn_type, resolution_val, st.session_state["graph_layout"])
scatter_key = (selected_group_name,)  # scatter depends only on feature set
geo_key     = (selected_group_name, int(k), knn_type, resolution_val)

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

    # =========================================================
    # PATH A: Cached Index + igraph  (large datasets)
    # =========================================================
    if use_igraph:
        import igraph as ig
        import scipy.sparse as sp

        nbr_distances = st.session_state["nbr_index_distances"]
        n = len(nbr_neighbors)

        with st.spinner(
            f"Building igraph from cached index (k={int(k)}, {knn_type}) "
            f"and running Leiden + PageRank on {n:,} nodes..."
        ):
            # Slice neighbor arrays to chosen k
            nbrs = nbr_neighbors[:, :int(k)].astype(np.int32)
            dists = nbr_distances[:, :int(k)].astype(np.float32)
            sims = (1.0 - dists).clip(0.0, 1.0)

            # Build edge arrays without Python loops
            valid_mask = (nbrs >= 0) & (nbrs != np.arange(n, dtype=np.int32)[:, None])
            row_idx = np.repeat(np.arange(n, dtype=np.int32), int(k))[valid_mask.ravel()]
            col_idx = nbrs.ravel()[valid_mask.ravel()]
            sim_vals = sims.ravel()[valid_mask.ravel()]

            if knn_type == "mutual":
                # scipy sparse: keep only edges present in both directions
                A = sp.csr_matrix(
                    (np.ones(len(row_idx), dtype=np.float32), (row_idx, col_idx)),
                    shape=(n, n),
                )
                A_sim = sp.csr_matrix(
                    (sim_vals, (row_idx, col_idx)), shape=(n, n)
                )
                A_mutual_triu = sp.triu(A.multiply(A.T), k=1)
                rows_m, cols_m = A_mutual_triu.nonzero()
                w_fwd = np.asarray(A_sim[rows_m, cols_m]).ravel()
                w_bwd = np.asarray(A_sim[cols_m, rows_m]).ravel()
                edge_weights = np.maximum(w_fwd, w_bwd).tolist()
                edges = list(zip(rows_m.tolist(), cols_m.tolist()))
                g = ig.Graph(n=n, edges=edges, directed=False)
            else:
                edges = list(zip(row_idx.tolist(), col_idx.tolist()))
                edge_weights = sim_vals.tolist()
                g = ig.Graph(n=n, edges=edges, directed=True)

            g.es["weight"] = edge_weights

            # Undirected view for community detection + degree
            g_ud = g if not g.is_directed() else g.as_undirected(combine_edges="max")

            # Leiden community detection
            partition = g_ud.community_leiden(
                weights="weight",
                resolution_parameter=float(resolution_val),
                objective_function="modularity",
                n_iterations=10,
            )
            community_labels = partition.membership
            modularity_val = partition.modularity

            # PageRank
            pagerank_vals = g.pagerank(weights="weight", directed=g.is_directed())

            # Degree
            degree_vals = g_ud.degree()

            # PCA indices (env / ses) for visualization and downstream fusion
            cols_in_data = [
                c for c in nbr_meta.get("columns", []) if c in zipc.columns
            ]
            indices = compute_adaptive_pca_indices(
                zipc, cols_in_data, DEFAULT_FEATURE_GROUPS
            )

            out = pd.DataFrame({
                "ZIPCODE": zipc["ZIPCODE"].astype(str).values,
                "environment_index": (
                    indices["environment_index"]
                    if indices["environment_index"] is not None
                    else np.nan
                ),
                "ses_index": (
                    indices["ses_index"]
                    if indices["ses_index"] is not None
                    else np.nan
                ),
                "zip_community": community_labels,
                "zip_betweenness": np.nan,   # not computed — infeasible at scale
                "zip_pagerank": pagerank_vals,
                "zip_degree": degree_vals,
                "isolated": [d == 0 for d in degree_vals],
                "environment_index_var": indices.get("env_var"),
                "ses_index_var": indices.get("ses_var"),
                "modularity": modularity_val,
            })

            # Graph-level summary stored on G placeholder (igraph)
            igraph_summary = {
                "nodes": g.vcount(),
                "edges": g.ecount(),
                "is_directed": g.is_directed(),
                "n_communities": len(set(community_labels)),
                "modularity": modularity_val,
                "isolated_nodes": int(sum(d == 0 for d in degree_vals)),
            }
            st.session_state["igraph_summary"] = igraph_summary

        # Save to session state (same keys as exact path)
        st.session_state["zip_indices"] = {
            "environment_index": out["environment_index"].to_numpy(),
            "ses_index": out["ses_index"].to_numpy(),
            "env_var": indices.get("env_var"),
            "ses_var": indices.get("ses_var"),
            "env_cols": indices.get("env_cols", []),
            "ses_cols": indices.get("ses_cols", []),
        }

        # Geographic plot (no network plot for large graphs)
        geo_path = CACHE_DIR / (
            f"geo_igraph_{selected_group_name}_k{int(k)}_{knn_type}"
            f"_res{resolution_val:.1f}.png"
        )
        fig_geo = plot_geographic_communities(
            out, zip_coords, wa_boundary,
            size_col="zip_pagerank", base_markersize=30,
        )
        fig_cache["geo"][geo_key] = {
            "geo_png_path": fig_to_png_file(fig_geo, geo_path)
        }

        onehot_preview = {}
        if "zip_community" in out.columns:
            vals = pd.Series(out["zip_community"].unique())
            onehot_preview["zip_community"] = int(vals.nunique())

        zip_indices = st.session_state.get("zip_indices", {})
        st.session_state["zip_features"] = out
        st.session_state["zip_features_meta"] = {
            "feature_group": selected_group_name,
            "k": int(k),
            "knn_type": knn_type,
            "resolution": float(resolution_val),
            "layout": layout_choice,
            "method": "igraph",
            "index_name": st.session_state.get("nbr_index_name"),
            "env_var": zip_indices.get("env_var"),
            "ses_var": zip_indices.get("ses_var"),
            "onehot_preview": onehot_preview,
        }
        # igraph path does NOT save to graph_cache (no NetworkX G object)
        st.success(
            f"igraph graph computed: {igraph_summary['nodes']:,} nodes, "
            f"{igraph_summary['edges']:,} edges, "
            f"{igraph_summary['n_communities']} communities "
            f"(modularity={modularity_val:.4f})."
        )

    # =========================================================
    # PATH B: Exact cosine similarity + NetworkX  (small datasets)
    # =========================================================
    else:
        # --- Early validation ---
        req_cols = get_group_columns(feature_groups[selected_group_name])
        missing_cols = [c for c in req_cols if c not in zipc.columns]
        if missing_cols:
            st.error(f"Cannot compute graph: missing columns {missing_cols}")
            st.stop()

        cache_key = (selected_group_name, int(k), knn_type, resolution_val)
        cached_result = get_cached_graph(cache_key)

        if cached_result is not None:
            G, out = cached_result
            st.info("Loaded from cache!")
        else:
            with st.spinner("Computing graph..."):
                results = process_zip_group(
                    zipc=zipc,
                    group_name=selected_group_name,
                    feature_groups=feature_groups,
                    k=int(k),
                    knn_type=knn_type,
                    default_groups=DEFAULT_FEATURE_GROUPS,
                    resolution=resolution_val,
                )

                if results is None:
                    st.error("Failed to process group - no valid features found.")
                    st.stop()

                feats = zipc[selected_features].astype(float).values
                feats = StandardScaler().fit_transform(feats)
                G = build_knn_graph(feats, k_neighbors=int(k), knn_type=knn_type)

                env_var_key = f"environment_index_var_{selected_group_name}"
                ses_var_key = f"ses_index_var_{selected_group_name}"

                out = pd.DataFrame({
                    "ZIPCODE": results["ZIPCODE"],
                    "environment_index": results[f"environment_index_{selected_group_name}"],
                    "ses_index": results[f"ses_index_{selected_group_name}"],
                    "zip_community": results[f"zip_community_{selected_group_name}"],
                    "zip_betweenness": results[f"zip_betweenness_{selected_group_name}"],
                    "zip_pagerank": results[f"zip_pagerank_{selected_group_name}"],
                    "zip_degree": results.get(f"degree_{selected_group_name}"),
                    "isolated": results.get(f"isolated_{selected_group_name}"),
                    "environment_index_var": (
                        results[env_var_key] if env_var_key in results.columns else None
                    ),
                    "ses_index_var": (
                        results[ses_var_key] if ses_var_key in results.columns else None
                    ),
                    "modularity": results.get(f"modularity_{selected_group_name}"),
                })

        st.session_state["zip_indices"] = {
            "environment_index": out["environment_index"].to_numpy(),
            "ses_index": out["ses_index"].to_numpy(),
            "env_var": (
                out["environment_index_var"].iloc[0]
                if "environment_index_var" in out.columns else None
            ),
            "ses_var": (
                out["ses_index_var"].iloc[0]
                if "ses_index_var" in out.columns else None
            ),
            "env_cols": [c for c in selected_features if c in get_cols_from_default("env")],
            "ses_cols": [c for c in selected_features if c in get_cols_from_default("ses")],
        }

        graph_cache[selected_group_key] = {"graph": G, "features": out}

        net_path = CACHE_DIR / (
            f"net_{selected_group_name}_k{int(k)}_{knn_type}"
            f"_res{resolution_val:.1f}_{st.session_state['graph_layout']}.png"
        )
        fig_net = plot_networkx_graph(
            G,
            out_df=out,
            node_size=20,
            edge_width=0.9,
            edge_alpha=0.5,
            edge_color="gray",
            community_col="zip_community",
            size_col="zip_pagerank",
            title=(
                f"Network plot: {selected_group_name} "
                f"(k={int(k)}, {knn_type}, res={resolution_val:.1f}, "
                f"layout={st.session_state['graph_layout']})"
            ),
            layout=st.session_state["graph_layout"],
            scale_factor=4.0,
        )
        fig_cache["network"][network_key] = {
            "network_png_path": fig_to_png_file(fig_net, net_path)
        }

        scatter_path = CACHE_DIR / f"scatter_{selected_group_name}.png"
        fig_scatter = plot_zip_scatter(out)
        fig_cache["scatter"][scatter_key] = {
            "scatter_png_path": fig_to_png_file(fig_scatter, scatter_path)
        }

        geo_path = CACHE_DIR / (
            f"geo_{selected_group_name}_k{int(k)}_{knn_type}"
            f"_res{resolution_val:.1f}.png"
        )
        fig_geo = plot_geographic_communities(
            out, zip_coords, wa_boundary,
            size_col="zip_pagerank", base_markersize=30,
        )
        fig_cache["geo"][geo_key] = {
            "geo_png_path": fig_to_png_file(fig_geo, geo_path)
        }

        onehot_preview = {}
        if "zip_community" in out.columns:
            vals = pd.Series(out["zip_community"].unique())
            valid = vals[vals >= 0]
            onehot_preview["zip_community"] = int(valid.nunique())

        zip_indices = st.session_state.get("zip_indices", {})
        st.session_state["zip_features"] = out
        st.session_state["zip_features_meta"] = {
            "feature_group": selected_group_name,
            "k": int(k),
            "knn_type": st.session_state["knn_type"],
            "resolution": st.session_state["resolution"],
            "layout": st.session_state["graph_layout"],
            "method": "exact",
            "env_var": zip_indices.get("env_var"),
            "ses_var": zip_indices.get("ses_var"),
            "onehot_preview": onehot_preview,
        }

        if cached_result is None:
            save_graph_to_cache(
                cache_key, G, out,
                st.session_state["zip_features_meta"],
                st.session_state["zip_indices"],
            )

        if cached_result is not None:
            st.success("Graph loaded from cache and neighborhood features updated.")
        else:
            st.success("Graph computed and neighborhood features updated.")

# ---------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------

# Results are available from either path:
#   exact  -> graph_cache[selected_group_key]
#   igraph -> st.session_state["zip_features"]
_exact_ready = selected_group_key in graph_cache
_igraph_ready = (
    use_igraph
    and st.session_state.get("zip_features") is not None
    and st.session_state.get("zip_features_meta", {}).get("method") == "igraph"
)
_has_results = _exact_ready or _igraph_ready

if _has_results:
    # Load results from the appropriate source
    if _exact_ready and not use_igraph:
        G = graph_cache[selected_group_key]["graph"]
        out = graph_cache[selected_group_key]["features"]
    else:
        out = st.session_state["zip_features"]
        G = None  # not available for igraph path

    # Status banner
    if "zip_features_meta" in st.session_state:
        meta = st.session_state["zip_features_meta"]
        method_label = meta.get("method", "exact")
        st.info(
            f"Active features: group=`{meta.get('feature_group', '')}`, "
            f"k={meta['k']}, knn_type=`{meta['knn_type']}`, "
            f"resolution={meta.get('resolution', 1.0):.1f}, "
            f"method=`{method_label}`"
            + (f", index=`{meta.get('index_name')}`" if method_label == "igraph" else "")
        )

    # --- Derived features table ---
    st.subheader("Derived Neighborhood Features")
    if "environment_index_var" not in out.columns:
        env_var = st.session_state.get("zip_indices", {}).get("env_var", np.nan)
        out["environment_index_var"] = env_var if env_var is not None else np.nan
    if "ses_index_var" not in out.columns:
        ses_var = st.session_state.get("zip_indices", {}).get("ses_var", np.nan)
        out["ses_index_var"] = ses_var if ses_var is not None else np.nan
    columns_to_include = [
        "environment_index", "ses_index", "zip_degree",
        "zip_betweenness", "zip_pagerank", "zip_community", "isolated",
    ]
    display_cols = ["ZIPCODE"] + [c for c in columns_to_include if c in out.columns]
    st.dataframe(out[display_cols], use_container_width=False, hide_index=True)

    # --- PCA variance explained ---
    zip_indices = st.session_state.get("zip_indices", {})
    env_var = zip_indices.get("env_var")
    ses_var = zip_indices.get("ses_var")
    st.subheader("PCA variance explained")
    st.dataframe(
        pd.DataFrame({
            "Metric": ["environment_index_variance_ratio", "ses_index_variance_ratio"],
            "Value": [env_var, ses_var],
        }),
        use_container_width=False, hide_index=True,
    )

    # --- Graph summary ---
    st.subheader("Graph summary")
    modularity_val = out["modularity"].iloc[0] if "modularity" in out.columns else np.nan
    num_communities = int(out["zip_community"].nunique()) if "zip_community" in out.columns else 0

    if G is not None:
        # Exact path: use NetworkX for detailed stats
        connected_flag = nx.is_weakly_connected(G) if G.is_directed() else nx.is_connected(G)
        num_isolates = sum(1 for _ in nx.isolates(G))
        num_components = (
            nx.number_weakly_connected_components(G) if G.is_directed()
            else nx.number_connected_components(G)
        )
        avg_deg = out["zip_degree"].mean() if "zip_degree" in out.columns else None
        graph_summary_dict = {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "avg_degree": f"{avg_deg:.2f}" if avg_deg is not None else "—",
            "n_communities": num_communities,
            "non_isolated_communities": max(num_communities - num_isolates, 0),
            "n_components": num_components,
            "non_isolated_components": max(num_components - num_isolates, 0),
            "isolated_nodes": num_isolates,
            "resolution": resolution_val,
            "modularity": f"{modularity_val:.4f}" if not np.isnan(modularity_val) else "N/A",
            "k": int(k),
            "is_connected": connected_flag,
            "feature_group": selected_group_name,
            "method": "exact (NetworkX)",
        }
    else:
        # igraph path: use stored summary
        ig_sum = st.session_state.get("igraph_summary", {})
        graph_summary_dict = {
            "nodes": ig_sum.get("nodes", len(out)),
            "edges": ig_sum.get("edges", "—"),
            "n_communities": ig_sum.get("n_communities", num_communities),
            "isolated_nodes": ig_sum.get("isolated_nodes", "—"),
            "resolution": resolution_val,
            "modularity": f"{modularity_val:.4f}" if not np.isnan(modularity_val) else "N/A",
            "k": int(k),
            "feature_group": selected_group_name,
            "method": "igraph (Leiden)",
            "betweenness": "not computed (large graph)",
        }
    st.dataframe(
        pd.DataFrame({
            "Statistic": list(graph_summary_dict.keys()),
            "Value": list(graph_summary_dict.values()),
        }),
        use_container_width=False, hide_index=True,
    )

    # --- Community summaries ---
    st.subheader("Community Summaries")
    try:
        total_zip_rows = len(out)
        exclude_cols = {"ZIPCODE", "zip_community"}
        numeric_cols = [
            col for col in out.columns
            if pd.api.types.is_numeric_dtype(out[col]) and col not in exclude_cols
        ]
        comm_tbl = (
            out.assign(zip_community=out["zip_community"].fillna(-1))
               .groupby("zip_community", dropna=False)
        )
        zip_count = comm_tbl.size().rename("zip_count")
        pct_of_total = (zip_count / total_zip_rows).round(4).rename("pct_of_total")
        mean_metrics = comm_tbl[numeric_cols].mean().add_prefix("mean_")
        summary_tbl = pd.concat([zip_count, pct_of_total, mean_metrics], axis=1).reset_index()
        summary_tbl = summary_tbl.sort_values("zip_count", ascending=False, kind="mergesort")
        st.dataframe(summary_tbl, use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Could not compute community summary table: {e}")

    # --- PCA scatter (exact path only) ---
    if not use_igraph:
        st.subheader("PCA Indices Scatter")
        scatter_png_path = fig_cache["scatter"].get(scatter_key, {}).get("scatter_png_path")
        if scatter_png_path and os.path.exists(scatter_png_path):
            st.image(scatter_png_path, width='content')
        else:
            st.info("Scatter image not cached yet. Click 'Compute Graph' to generate.")

    # --- Network graph (exact path only) ---
    if not use_igraph:
        st.subheader("Network Graph")
        if network_key in fig_cache["network"]:
            net_png_path = fig_cache["network"][network_key]["network_png_path"]
            if os.path.exists(net_png_path):
                st.image(net_png_path, width='content')
            else:
                st.info("Network image missing on disk. Click 'Compute Graph' to regenerate.")
        else:
            st.info("Network image not cached yet. Click 'Compute Graph' to generate.")
    else:
        st.subheader("Network Graph")
        st.info(
            "Network graph plot not available for large datasets."
        )

    # --- Geographic community map (both paths) ---
    st.subheader("Geographic Community Map")
    if geo_key in fig_cache["geo"]:
        geo_png_path = fig_cache["geo"][geo_key].get("geo_png_path")
        if geo_png_path and os.path.exists(geo_png_path):
            st.image(geo_png_path, width='content')
        else:
            st.info("Geographic image missing on disk. Click 'Compute Graph' to regenerate.")
    else:
        st.info("Geographic plot not cached yet. Click 'Compute Graph' to generate.")

else:
    st.info("No graph computed yet. Select settings above and click 'Compute Graph'.")

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