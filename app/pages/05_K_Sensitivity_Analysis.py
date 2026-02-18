# app/pages/05_K_Sensitivity_Analysis.py

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import igraph as ig
import streamlit as st
from sklearn.preprocessing import StandardScaler

from app._components.zip_context_utils import (
    process_zip_group,
    present,
    get_group_columns,
    build_knn_graph,
)
from app._logic.loader import ensure_data_loaded
from app._logic.graph_cache import (
    get_cached_graph,
    clear_all_cache,
    initialize_session_cache,
    reconstruct_results_from_cache,
    save_graph_to_cache,
)

# ---------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------
st.set_page_config(page_title="K Sensitivity Analysis", layout="wide")
st.title("K Sensitivity Analysis")

# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
if not ensure_data_loaded():
    st.stop()

zipc = st.session_state.get("zip_df").reset_index(drop=True)
active_context_key = st.session_state.get("active_context_key", "default")

default_fg = st.session_state.get("default_feature_groups")
if not isinstance(default_fg, dict):
    fg_path = Path("data/config/default_feature_groups.json")
    if fg_path.exists():
        with open(fg_path, "r", encoding="utf-8") as fh:
            default_fg = json.load(fh)
    else:
        st.error("Default feature groups not found.")
        st.stop()

DEFAULT_FEATURE_GROUPS = default_fg
custom_feature_groups = st.session_state.get("feature_groups", {})
if not isinstance(custom_feature_groups, dict):
    custom_feature_groups = {}
feature_groups = {**DEFAULT_FEATURE_GROUPS, **custom_feature_groups}

if not feature_groups:
    st.error("No feature groups available.")
    st.stop()

initialize_session_cache()

# ---------------------------------------------------------------------
# Cached index helpers (same as page 04)
# ---------------------------------------------------------------------
INDEX_CACHE_DIR = Path("data/cache/nbr_index")
INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _find_best_index(group_name: str, dataset_key: str, k_required: int):
    """Find the smallest-k_max cached index that covers k_required for a group+dataset."""
    candidates = []
    for meta_file in INDEX_CACHE_DIR.glob("*.json"):
        try:
            with open(meta_file, encoding="utf-8") as fh:
                meta = json.load(fh)
            if (
                meta.get("group_name") == group_name
                and meta.get("dataset_key") == dataset_key
                and int(meta.get("k_max", 0)) >= k_required
            ):
                npz = INDEX_CACHE_DIR / f"{meta_file.stem}.npz"
                if npz.exists():
                    candidates.append((meta_file.stem, meta, npz))
        except Exception:
            pass
    if not candidates:
        return None
    candidates.sort(key=lambda x: int(x[1].get("k_max", 0)))
    return candidates[0]


def _igraph_metrics(npz_path, k, knn_type, resolution):
    """
    Build igraph from cached index at k and return a metrics dict.
    Computes: nodes, edges, n_communities, isolated_nodes, modularity,
              n_components — all via igraph (no NetworkX needed).
    """
    data = np.load(npz_path)
    neighbors = data["neighbors"]
    distances = data["distances"]
    n = len(neighbors)

    nbrs = neighbors[:, :k].astype(np.int32)
    dists = distances[:, :k].astype(np.float32)
    sims = (1.0 - dists).clip(0.0, 1.0)

    valid_mask = (nbrs >= 0) & (nbrs != np.arange(n, dtype=np.int32)[:, None])
    row_idx = np.repeat(np.arange(n, dtype=np.int32), k)[valid_mask.ravel()]
    col_idx = nbrs.ravel()[valid_mask.ravel()]
    sim_vals = sims.ravel()[valid_mask.ravel()]

    if knn_type == "mutual":
        adj = sp.csr_matrix(
            (np.ones(len(row_idx), dtype=np.float32), (row_idx, col_idx)),
            shape=(n, n),
        )
        mutual_triu = sp.triu(adj.multiply(adj.T), k=1)
        rows_m, cols_m = mutual_triu.nonzero()
        adj_sim = sp.csr_matrix((sim_vals, (row_idx, col_idx)), shape=(n, n))
        w_fwd = np.asarray(adj_sim[rows_m, cols_m]).ravel()
        w_bwd = np.asarray(adj_sim[cols_m, rows_m]).ravel()
        edge_weights = np.maximum(w_fwd, w_bwd).tolist()
        edges = list(zip(rows_m.tolist(), cols_m.tolist()))
        graph = ig.Graph(n=n, edges=edges, directed=False)
    else:
        edges = list(zip(row_idx.tolist(), col_idx.tolist()))
        edge_weights = sim_vals.tolist()
        graph = ig.Graph(n=n, edges=edges, directed=True)

    graph.es["weight"] = edge_weights
    g_ud = graph if not graph.is_directed() else graph.as_undirected(combine_edges="max")

    partition = g_ud.community_leiden(
        weights="weight",
        resolution_parameter=float(resolution),
        objective_function="modularity",
        n_iterations=10,
    )
    n_communities = len(partition)
    modularity_val = partition.modularity
    degree_vals = g_ud.degree()
    isolated_nodes = int(sum(d == 0 for d in degree_vals))
    avg_degree = round(float(np.mean(degree_vals)), 4) if degree_vals else None

    # Connected components (igraph)
    clusters = g_ud.clusters()
    n_components = len(clusters)

    n_edges = graph.ecount()
    non_isolated_communities = n_communities - isolated_nodes
    non_isolated_components = n_components - isolated_nodes

    return {
        "nodes": n,
        "edges": n_edges,
        "avg_degree": avg_degree,
        "n_communities": n_communities,
        "non_isolated_communities": max(non_isolated_communities, 0),
        "n_components": n_components,
        "non_isolated_components": max(non_isolated_components, 0),
        "isolated_nodes": isolated_nodes,
        "modularity": round(modularity_val, 4),
    }


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
st.subheader("Analysis Settings")

method = st.radio(
    "Compute method",
    ["Exact (NetworkX)", "Cached Index (igraph)"],
    index=0 if st.session_state.get("nbr_graph_method", "exact") == "exact" else 1,
    horizontal=True,
)
use_igraph = method == "Cached Index (igraph)"

col1, col2 = st.columns(2)

with col1:
    selected_groups = st.multiselect(
        "Feature Groups",
        options=list(feature_groups.keys()),
        default=None,
    )

with col2:
    k_values_input = st.text_input(
        "k values (comma-separated)",
        value="3, 5, 7, 10",
        help="Enter k values separated by commas. For igraph, each k must be ≤ k_max of the cached index.",
    )

col3, col4 = st.columns(2)

with col3:
    knn_type_label = st.selectbox(
        "Graph Type",
        options=["Mutual k-NN (undirected)", "Directed k-NN"],
        index=0,
    )
    knn_type = "mutual" if knn_type_label.startswith("Mutual") else "directed"

with col4:
    resolution = st.number_input(
        "Community resolution",
        min_value=0.01, max_value=100.0,
        value=1.0, step=0.1, format="%.2f",
        help="Higher values produce more/smaller communities",
    )

# Parse k values
try:
    k_max_limit = 100 if use_igraph else 20
    k_values = [int(v.strip()) for v in k_values_input.split(",") if v.strip()]
    k_values = sorted(set(k for k in k_values if 1 <= k <= k_max_limit))
except ValueError:
    k_values = []

if not k_values:
    st.warning(f"Please enter valid k values (integers 1–{k_max_limit}).")

# -----------------------------------------------------------------
# For igraph: show index availability before running
# -----------------------------------------------------------------
if use_igraph and selected_groups and k_values:
    k_max_needed = max(k_values)
    st.markdown(
        f"**Index availability** (dataset: `{active_context_key}`, "
        f"k_max needed ≥ {k_max_needed})"
    )
    avail_rows = []
    missing_groups = []
    for grp in selected_groups:
        match = _find_best_index(grp, active_context_key, k_max_needed)
        if match:
            idx_name, idx_meta, _ = match
            avail_rows.append({
                "Group": grp,
                "Index": idx_name,
                "k_max": idx_meta.get("k_max"),
                "Status": "Ready",
            })
        else:
            avail_rows.append({
                "Group": grp, "Index": "—", "k_max": "—",
                "Status": f"Missing — need k_max ≥ {k_max_needed}",
            })
            missing_groups.append(grp)
    st.dataframe(pd.DataFrame(avail_rows), use_container_width=False, hide_index=True)
    if missing_groups:
        st.error(
            f"No cached index for: **{', '.join(missing_groups)}** "
            f"(dataset=`{active_context_key}`, k_max ≥ {k_max_needed}). "
            "Build them in page 02 first."
        )

col_btn1, col_btn2, _ = st.columns([1, 1, 2])
with col_btn1:
    run_analysis = st.button("Run Analysis", type="primary")
with col_btn2:
    clear_cache_clicked = st.button("Clear Cache")

if clear_cache_clicked:
    clear_all_cache()
    st.success("Cache cleared.")
    st.rerun()

st.markdown("---")

if not selected_groups:
    st.info("Select feature groups above to analyze.")
    st.stop()

# ---------------------------------------------------------------------
# METRICS tracked in the cross-tab
# ---------------------------------------------------------------------
METRICS = [
    "nodes", "edges", "avg_degree", "n_communities", "non_isolated_communities",
    "n_components", "non_isolated_components", "isolated_nodes", "modularity",
]

# ---------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------
if run_analysis and k_values:

    # =========================================================
    # PATH A: Cached Index + igraph
    # =========================================================
    if use_igraph:
        k_max_needed = max(k_values)
        missing = [
            grp for grp in selected_groups
            if _find_best_index(grp, active_context_key, k_max_needed) is None
        ]
        if missing:
            st.error(
                f"Cannot run: missing indexes for **{', '.join(missing)}**. "
                "Build them in page 02 first."
            )
            st.stop()

        results_dict = {}
        total_iterations = len(selected_groups) * len(k_values)
        current_iteration = 0
        progress_bar = st.progress(0)
        status_text = st.empty()

        for group_name in selected_groups:
            idx_name, idx_meta, npz_path = _find_best_index(
                group_name, active_context_key, k_max_needed
            )
            for k_val in k_values:
                status_text.text(
                    f"Processing: {group_name}, k={k_val} "
                    f"(index='{idx_name}', igraph)"
                )
                metrics = _igraph_metrics(
                    npz_path, k_val, knn_type, resolution
                )
                results_dict[(group_name, k_val)] = metrics
                current_iteration += 1
                progress_bar.progress(current_iteration / total_iterations)

        progress_bar.empty()
        status_text.empty()
        st.success(
            f"Analysis complete: {len(selected_groups)} group(s) × "
            f"{len(k_values)} k values (igraph)."
        )

    # =========================================================
    # PATH B: Exact cosine similarity + NetworkX
    # =========================================================
    else:
        results_dict = {}
        total_iterations = len(selected_groups) * len(k_values)
        current_iteration = 0
        cache_hits = 0
        cache_misses = 0
        progress_bar = st.progress(0)
        status_text = st.empty()

        for group_name in selected_groups:
            for k_val in k_values:
                cache_key = (group_name, int(k_val), knn_type, float(resolution))
                cached_result = get_cached_graph(cache_key)

                if cached_result is not None:
                    cache_hits += 1
                    status_text.text(f"Processing: {group_name}, k={k_val} (from cache)")
                    try:
                        graph, features_df = cached_result
                        results = reconstruct_results_from_cache(
                            graph, features_df, group_name
                        )
                    except Exception as exc:
                        st.warning(f"Cache error for {group_name}, k={k_val}: {exc}. Recomputing...")
                        cache_hits -= 1
                        cache_misses += 1
                        results = process_zip_group(
                            zipc=zipc, group_name=group_name,
                            feature_groups=feature_groups, k=k_val,
                            knn_type=knn_type, default_groups=DEFAULT_FEATURE_GROUPS,
                            resolution=resolution,
                        )
                else:
                    cache_misses += 1
                    status_text.text(f"Processing: {group_name}, k={k_val}")
                    try:
                        sel_cols = get_group_columns(feature_groups[group_name])
                        selected_features = present(zipc, sel_cols)
                        if selected_features:
                            feats = zipc[selected_features].astype(float).values
                            feats = StandardScaler().fit_transform(feats)
                            G = build_knn_graph(
                                feats, k_neighbors=k_val, knn_type=knn_type
                            )
                            results = process_zip_group(
                                zipc=zipc, group_name=group_name,
                                feature_groups=feature_groups, k=k_val,
                                knn_type=knn_type, default_groups=DEFAULT_FEATURE_GROUPS,
                                resolution=resolution,
                            )
                            if results is not None:
                                out_df = pd.DataFrame({
                                    "ZIPCODE": results["ZIPCODE"].astype(str),
                                    "zip_community": results.get(f"zip_community_{group_name}"),
                                    "zip_pagerank": results.get(f"zip_pagerank_{group_name}"),
                                    "modularity": results.get(f"modularity_{group_name}"),
                                })
                                save_graph_to_cache(
                                    cache_key, G, out_df,
                                    {"feature_group": group_name, "k": k_val,
                                     "knn_type": knn_type, "resolution": resolution},
                                )
                        else:
                            results = None
                    except Exception as exc:
                        st.error(f"Error processing {group_name}, k={k_val}: {exc}")
                        results = None

                # Extract metrics
                if results is not None:
                    def _get_col(res, g_key, col):
                        full = f"{col}_{g_key}"
                        return res[full].iloc[0] if full in res.columns else None

                    nodes = _get_col(results, group_name, "nodes")
                    edges = _get_col(results, group_name, "edges")
                    num_comm = _get_col(results, group_name, "num_communities")
                    isolated = _get_col(results, group_name, "isolated_nodes")
                    n_comp = _get_col(results, group_name, "n_components")
                    mod = _get_col(results, group_name, "modularity")

                    deg_col = f"degree_{group_name}"
                    avg_deg = (
                        round(float(results[deg_col].mean()), 4)
                        if deg_col in results.columns else None
                    )
                    non_iso_comm = (
                        int(num_comm) - int(isolated)
                        if num_comm is not None and isolated is not None else None
                    )
                    non_iso_comp = (
                        int(n_comp) - int(isolated)
                        if n_comp is not None and isolated is not None else None
                    )

                    results_dict[(group_name, k_val)] = {
                        "nodes": nodes,
                        "edges": edges,
                        "avg_degree": avg_deg,
                        "n_communities": num_comm,
                        "non_isolated_communities": non_iso_comm,
                        "n_components": n_comp,
                        "non_isolated_components": non_iso_comp,
                        "isolated_nodes": isolated,
                        "modularity": round(mod, 4) if mod is not None else None,
                    }
                else:
                    results_dict[(group_name, k_val)] = {m: None for m in METRICS}

                current_iteration += 1
                progress_bar.progress(current_iteration / total_iterations)

        progress_bar.empty()
        status_text.empty()
        if cache_hits:
            st.info(f"{cache_hits} run(s) loaded from cache, {cache_misses} computed.")
        st.success(
            f"Analysis complete: {len(selected_groups)} group(s) × {len(k_values)} k values."
        )

    # -----------------------------------------------------------------
    # Build cross-tab and store
    # -----------------------------------------------------------------
    rows = []
    for idx, group_name in enumerate(selected_groups):
        for metric_idx, metric in enumerate(METRICS):
            row = {
                "Group": group_name if metric_idx == 0 else "",
                "Metric": metric,
            }
            for k_val in k_values:
                row[f"k={k_val}"] = results_dict.get((group_name, k_val), {}).get(metric)
            rows.append(row)
        if idx < len(selected_groups) - 1:
            blank = {"Group": "", "Metric": ""}
            for k_val in k_values:
                blank[f"k={k_val}"] = ""
            rows.append(blank)

    crosstab_df = pd.DataFrame(rows)
    st.session_state["k_sensitivity_results"] = crosstab_df
    st.session_state["k_sensitivity_settings"] = {
        "groups": selected_groups,
        "k_values": k_values,
        "knn_type": knn_type,
        "resolution": resolution,
        "method": "igraph" if use_igraph else "exact",
    }

# ---------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------
if "k_sensitivity_results" in st.session_state:
    crosstab_df = st.session_state["k_sensitivity_results"]
    settings = st.session_state.get("k_sensitivity_settings", {})

    st.subheader("Cross-Tab Results")
    if settings:
        st.caption(
            f"Graph type: {settings.get('knn_type', 'N/A')}, "
            f"Resolution: {settings.get('resolution', 'N/A')}, "
            f"Method: {settings.get('method', 'exact')}"
        )

    table_height = (len(crosstab_df) + 1) * 35 + 10
    st.dataframe(crosstab_df, use_container_width=True, hide_index=True, height=table_height)

    csv = crosstab_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="k_sensitivity_analysis.csv",
        mime="text/csv",
    )
