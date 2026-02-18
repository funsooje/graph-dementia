# app/pages/04_Feature_Group_Comparison.py

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import igraph as ig
import streamlit as st
from sklearn.preprocessing import StandardScaler

from app._components.zip_context_utils import (
    present,
    get_group_columns,
    process_zip_group,
    compute_adaptive_pca_indices,
    build_knn_graph,
)
from app._components.plots import (
    plot_zip_scatter,
    plot_networkx_graph,
    plot_geographic_communities,
)
from app._logic.loader import ensure_data_loaded
from app._logic.graph_cache import (
    save_graph_to_cache,
    get_cached_graph,
    clear_all_cache,
    initialize_session_cache,
    reconstruct_results_from_cache,
)

# ---------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------
st.set_page_config(page_title="Feature Group Comparison", layout="wide")
st.title("Feature Group Comparison")

# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
if not ensure_data_loaded():
    st.stop()

zipc = st.session_state.get("zip_df").reset_index(drop=True)
zip_coords = st.session_state.get("zip_coords")
wa_boundary = st.session_state.get("wa_boundary")
active_context_key = st.session_state.get("active_context_key", "default")

# Feature groups
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
# Cached index helpers
# ---------------------------------------------------------------------
INDEX_CACHE_DIR = Path("data/cache/nbr_index")
INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _find_best_index(group_name: str, dataset_key: str, k_required: int):
    """
    Find the best cached index for a group + dataset.
    Returns (index_name, meta, npz_path) or None.
    Prefers smallest k_max >= k_required.
    """
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


def _process_group_igraph(group_name, meta, npz_path, k, knn_type, resolution):
    """Build igraph from a cached index and return (results_df, stats_dict)."""
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
        A = sp.csr_matrix(
            (np.ones(len(row_idx), dtype=np.float32), (row_idx, col_idx)),
            shape=(n, n),
        )
        A_sim = sp.csr_matrix((sim_vals, (row_idx, col_idx)), shape=(n, n))
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
    g_ud = g if not g.is_directed() else g.as_undirected(combine_edges="max")

    partition = g_ud.community_leiden(
        weights="weight",
        resolution_parameter=float(resolution),
        objective_function="modularity",
        n_iterations=10,
    )
    community_labels = partition.membership
    modularity_val = partition.modularity
    pagerank_vals = g.pagerank(weights="weight", directed=g.is_directed())
    degree_vals = g_ud.degree()

    cols_in_data = [c for c in meta.get("columns", []) if c in zipc.columns]
    indices = compute_adaptive_pca_indices(zipc, cols_in_data, DEFAULT_FEATURE_GROUPS)

    sfx = group_name
    n_nodes = g.vcount()
    n_edges = g.ecount()
    n_communities = len(set(community_labels))
    n_isolated = int(sum(d == 0 for d in degree_vals))

    results = pd.DataFrame({
        "ZIPCODE": zipc["ZIPCODE"].astype(str).values,
        f"environment_index_{sfx}": (
            indices["environment_index"]
            if indices["environment_index"] is not None else np.nan
        ),
        f"ses_index_{sfx}": (
            indices["ses_index"]
            if indices["ses_index"] is not None else np.nan
        ),
        f"zip_community_{sfx}": community_labels,
        f"zip_betweenness_{sfx}": np.nan,   # not computed at scale
        f"zip_pagerank_{sfx}": pagerank_vals,
        f"degree_{sfx}": degree_vals,
        f"isolated_{sfx}": [d == 0 for d in degree_vals],
        f"nodes_{sfx}": n_nodes,
        f"edges_{sfx}": n_edges,
        f"num_communities_{sfx}": n_communities,
        f"isolated_nodes_{sfx}": n_isolated,
        f"modularity_{sfx}": modularity_val,
    })
    if indices.get("env_var") is not None:
        results[f"environment_index_var_{sfx}"] = indices["env_var"]
    if indices.get("ses_var") is not None:
        results[f"ses_index_var_{sfx}"] = indices["ses_var"]

    stats = {
        "group": group_name,
        "index_used": meta.get("cache_name", "?"),
        "nodes": n_nodes,
        "edges": n_edges,
        "n_communities": n_communities,
        "isolated_nodes": n_isolated,
        "modularity": round(modularity_val, 4),
        "env_var": indices.get("env_var"),
        "ses_var": indices.get("ses_var"),
    }
    return results, stats


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
st.subheader("Comparison Settings")

method = st.radio(
    "Compute method",
    ["Exact (NetworkX)", "Cached Index (igraph)"],
    index=0 if st.session_state.get("nbr_graph_method", "exact") == "exact" else 1,
    horizontal=True,
    help=(
        "**Exact**: cosine similarity on all pairs — fine for small datasets (<5k nodes). "
        "**Cached Index**: uses PyNNDescent indexes from page 02 — required for large datasets."
    ),
)
use_igraph = method == "Cached Index (igraph)"

selected_groups = st.multiselect(
    "Select Feature Groups to Compare",
    options=list(feature_groups.keys()),
    default=None,
)

col1, col2 = st.columns(2)
with col1:
    # k_max depends on method; for igraph we allow up to whatever the indexes support
    k_max_ui = 100 if use_igraph else 20
    k = st.number_input(
        "k (k-NN)",
        min_value=1, max_value=k_max_ui,
        value=min(int(st.session_state.get("graph_k", 3)), k_max_ui),
        step=1,
    )
    knn_type_label = st.selectbox(
        "Graph Type",
        options=["Mutual k-NN (undirected)", "Directed k-NN"],
        index=0,
    )
    knn_type = "mutual" if knn_type_label.startswith("Mutual") else "directed"

with col2:
    resolution = st.number_input(
        "Community resolution",
        min_value=0.01, max_value=100.0,
        value=float(st.session_state.get("resolution", 1.0)),
        step=0.1, format="%.2f",
        help="Higher values produce more/smaller communities",
    )
    if not use_igraph:
        layout_choice = st.selectbox(
            "Graph Layout",
            options=["spring", "kamada", "circular", "random", "shell"],
            index=0,
        )
    else:
        layout_choice = "spring"
        st.markdown("*Network plot not available for large datasets.*")

# -----------------------------------------------------------------
# For igraph: show index availability before running
# -----------------------------------------------------------------
if use_igraph and selected_groups:
    st.markdown("**Index availability** (dataset: `" + active_context_key + "`, k=" + str(int(k)) + ")")
    avail_rows = []
    missing_groups = []
    for grp in selected_groups:
        match = _find_best_index(grp, active_context_key, int(k))
        if match:
            idx_name, idx_meta, _ = match
            avail_rows.append({
                "Group": grp,
                "Index": idx_name,
                "k_max": idx_meta.get("k_max"),
                "Nodes": f"{idx_meta.get('n_nodes', 0):,}",
                "Status": "Ready",
            })
        else:
            avail_rows.append({
                "Group": grp,
                "Index": "—",
                "k_max": "—",
                "Nodes": "—",
                "Status": "Missing — build in page 02",
            })
            missing_groups.append(grp)
    st.dataframe(pd.DataFrame(avail_rows), use_container_width=False, hide_index=True)
    if missing_groups:
        st.error(
            f"No cached index found for: **{', '.join(missing_groups)}** "
            f"(dataset=`{active_context_key}`, k_max ≥ {int(k)}). "
            "Go to **Build Neighborhood Index** (page 02) to build them."
        )

col_btn1, col_btn2, _ = st.columns([1, 1, 2])
with col_btn1:
    run_batch = st.button("Run Comparison", type="primary")
with col_btn2:
    clear_cache_clicked = st.button("Clear Cache")

if clear_cache_clicked:
    clear_all_cache()
    st.success("Cache cleared.")
    st.rerun()

st.markdown("---")

if not selected_groups:
    st.info("Select feature groups above to compare.")
    st.stop()

# ---------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------
if run_batch:

    # =========================================================
    # PATH A: Cached Index + igraph
    # =========================================================
    if use_igraph:
        # Final validation — fail if any group is missing an index
        missing = [
            grp for grp in selected_groups
            if _find_best_index(grp, active_context_key, int(k)) is None
        ]
        if missing:
            st.error(
                f"Cannot run: missing indexes for **{', '.join(missing)}**. "
                "Build them in page 02 first."
            )
            st.stop()

        all_results = []
        group_stats = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, group_name in enumerate(selected_groups):
            status_text.text(f"Processing {group_name} with igraph...")
            idx_name, idx_meta, npz_path = _find_best_index(
                group_name, active_context_key, int(k)
            )
            with st.spinner(
                f"Building igraph for '{group_name}' "
                f"(index='{idx_name}', k={int(k)}, {knn_type})..."
            ):
                results, stats = _process_group_igraph(
                    group_name, idx_meta, npz_path,
                    int(k), knn_type, resolution,
                )
            all_results.append(results)
            group_stats.append(stats)
            progress_bar.progress((idx + 1) / len(selected_groups))

        progress_bar.empty()
        status_text.empty()

    # =========================================================
    # PATH B: Exact cosine similarity + NetworkX
    # =========================================================
    else:
        all_results = []
        group_stats = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        cache_hits = 0
        cache_misses = 0

        for idx, group_name in enumerate(selected_groups):
            status_text.text(f"Processing group: {group_name}")
            try:
                cache_key = (group_name, int(k), knn_type, float(resolution))
                cached_result = get_cached_graph(cache_key)

                if cached_result is not None:
                    cache_hits += 1
                    status_text.text(f"Processing group: {group_name} (from cache)")
                    graph, features_df = cached_result
                    results = reconstruct_results_from_cache(graph, features_df, group_name)
                else:
                    cache_misses += 1
                    sel_cols = get_group_columns(feature_groups[group_name])
                    selected_features = present(zipc, sel_cols)
                    if not selected_features:
                        st.warning(f"No valid features for group '{group_name}' — skipping.")
                        continue

                    feats = zipc[selected_features].astype(float).values
                    feats = StandardScaler().fit_transform(feats)
                    G = build_knn_graph(feats, k_neighbors=int(k), knn_type=knn_type)

                    results = process_zip_group(
                        zipc=zipc,
                        group_name=group_name,
                        feature_groups=feature_groups,
                        k=int(k),
                        knn_type=knn_type,
                        default_groups=DEFAULT_FEATURE_GROUPS,
                        resolution=resolution,
                    )

                    if results is not None:
                        out_df = pd.DataFrame({
                            "ZIPCODE": results["ZIPCODE"].astype(str),
                            "environment_index": results.get(f"environment_index_{group_name}"),
                            "ses_index": results.get(f"ses_index_{group_name}"),
                            "zip_community": results.get(f"zip_community_{group_name}"),
                            "zip_betweenness": results.get(f"zip_betweenness_{group_name}"),
                            "zip_pagerank": results.get(f"zip_pagerank_{group_name}"),
                            "zip_degree": results.get(f"degree_{group_name}"),
                            "isolated": results.get(f"isolated_{group_name}"),
                            "modularity": results.get(f"modularity_{group_name}"),
                        })
                        save_graph_to_cache(
                            cache_key, G, out_df,
                            {"feature_group": group_name, "k": int(k),
                             "knn_type": knn_type, "resolution": resolution},
                        )

                if results is not None:
                    def _get_scalar(res, grp, col):
                        full = f"{col}_{grp}"
                        return res[full].iloc[0] if full in res.columns else None

                    group_stats.append({
                        "group": group_name,
                        "nodes": _get_scalar(results, group_name, "nodes"),
                        "edges": _get_scalar(results, group_name, "edges"),
                        "n_communities": _get_scalar(results, group_name, "num_communities"),
                        "isolated_nodes": _get_scalar(results, group_name, "isolated_nodes"),
                        "modularity": _get_scalar(results, group_name, "modularity"),
                        "env_var": _get_scalar(results, group_name, "environment_index_var"),
                        "ses_var": _get_scalar(results, group_name, "ses_index_var"),
                    })
                    all_results.append(results)

            except Exception as e:
                st.error(f"Error processing group {group_name}: {e}")

            progress_bar.progress((idx + 1) / len(selected_groups))

        progress_bar.empty()
        status_text.empty()
        if cache_hits:
            st.info(f"{cache_hits} group(s) loaded from cache, {cache_misses} computed.")

    # -----------------------------------------------------------------
    # Store + display
    # -----------------------------------------------------------------
    if all_results:
        final_results = all_results[0]
        for df in all_results[1:]:
            final_results = final_results.merge(df, on="ZIPCODE", how="outer")

        st.session_state["batch_results"] = final_results
        st.session_state["batch_settings"] = {
            "k": int(k),
            "knn_type": knn_type,
            "layout": layout_choice,
            "resolution": resolution,
            "groups": selected_groups,
            "method": "igraph" if use_igraph else "exact",
        }

        st.success(f"Processed {len(all_results)} feature groups.")

# ---------------------------------------------------------------------
# Outputs (shown from session state so they persist across re-renders)
# ---------------------------------------------------------------------
batch_results = st.session_state.get("batch_results")
batch_settings = st.session_state.get("batch_settings", {})

if batch_results is not None and batch_settings.get("groups"):
    stored_groups = batch_settings["groups"]
    stored_method = batch_settings.get("method", "exact")
    stored_k = batch_settings.get("k", k)
    stored_knn_type = batch_settings.get("knn_type", knn_type)
    stored_resolution = batch_settings.get("resolution", resolution)

    st.info(
        f"Showing results for: {', '.join(f'`{g}`' for g in stored_groups)} "
        f"| k={stored_k}, {stored_knn_type}, res={stored_resolution}, method=`{stored_method}`"
    )

    # --- Summary table ---
    st.subheader("Group Summary")
    def _batch_col(res, grp, name):
        full = f"{name}_{grp}"
        return res[full].iloc[0] if full in res.columns else None

    summary_rows = []
    for grp in stored_groups:
        summary_rows.append({
            "group": grp,
            "nodes": _batch_col(batch_results, grp, "nodes"),
            "edges": _batch_col(batch_results, grp, "edges"),
            "n_communities": _batch_col(batch_results, grp, "num_communities"),
            "isolated_nodes": _batch_col(batch_results, grp, "isolated_nodes"),
            "modularity": _batch_col(batch_results, grp, "modularity"),
            "env_var": _batch_col(batch_results, grp, "environment_index_var"),
            "ses_var": _batch_col(batch_results, grp, "ses_index_var"),
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=False, hide_index=True)

    # --- Full results table ---
    st.subheader("Results")
    drop_prefixes = (
        "environment_index_var_", "ses_index_var_", "nodes_", "edges_",
        "num_communities_", "isolated_nodes_",
    )
    display_cols = [
        c for c in batch_results.columns
        if not any(c.startswith(p) for p in drop_prefixes)
    ]
    st.dataframe(
        batch_results[display_cols],
        use_container_width=False, height=400, hide_index=True,
    )

    # --- Per-group plots ---
    st.subheader("Per-group plots")
    for grp in stored_groups:
        with st.expander(f"Plots: {grp}", expanded=False):
            out = pd.DataFrame({
                "ZIPCODE": batch_results["ZIPCODE"].astype(str),
                "environment_index": batch_results.get(f"environment_index_{grp}"),
                "ses_index": batch_results.get(f"ses_index_{grp}"),
                "zip_community": batch_results.get(f"zip_community_{grp}"),
                "zip_betweenness": batch_results.get(f"zip_betweenness_{grp}"),
                "zip_pagerank": batch_results.get(f"zip_pagerank_{grp}"),
                "zip_degree": batch_results.get(f"degree_{grp}"),
                "isolated": batch_results.get(f"isolated_{grp}"),
            })

            cols = st.columns(3)

            # 1) PCA scatter
            with cols[0]:
                has_env = "environment_index" in out.columns and not out["environment_index"].isna().all()
                has_ses = "ses_index" in out.columns and not out["ses_index"].isna().all()
                if has_env and has_ses:
                    fig_scatter = plot_zip_scatter(
                        out,
                        x_col="environment_index", y_col="ses_index",
                        comm_col="zip_community", size_col="zip_pagerank",
                        title=f"PCA scatter: {grp}",
                    )
                    st.pyplot(fig_scatter)
                else:
                    st.info("PCA scatter requires both env and SES indices — skipping.")

            # 2) Network graph (exact path only — meaningless at 71k nodes)
            with cols[1]:
                if stored_method == "exact":
                    cache_key = (grp, stored_k, stored_knn_type, float(stored_resolution))
                    cached = get_cached_graph(cache_key)
                    if cached is not None:
                        G_plot, _ = cached
                        fig_net = plot_networkx_graph(
                            G_plot, out_df=out,
                            node_size=10, edge_width=0.9, edge_alpha=0.5,
                            edge_color="gray", layout=batch_settings.get("layout", "spring"),
                            community_col="zip_community", size_col="zip_pagerank",
                            title=f"Network graph: {grp}", scale_factor=4.0,
                        )
                        st.pyplot(fig_net)
                    else:
                        st.info("Network graph not in cache. Re-run comparison to generate.")
                else:
                    st.info("Network plot not available for large datasets (igraph path).")

            # 3) Geographic map
            with cols[2]:
                if (zip_coords is not None and wa_boundary is not None
                        and "zip_pagerank" in out.columns
                        and not out["zip_pagerank"].isna().all()):
                    fig_geo = plot_geographic_communities(
                        out, zip_coords, wa_boundary,
                        size_col="zip_pagerank", base_markersize=30,
                    )
                    st.pyplot(fig_geo)
                else:
                    st.info("Geographic map unavailable (missing required data).")
