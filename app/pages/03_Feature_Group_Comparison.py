# app/pages/03_Feature_Group_Comparison.py

from pathlib import Path
import json
import streamlit as st
import pandas as pd
import numpy as np

from app._components.zip_context_utils import (
    present,
    get_group_columns,
    process_zip_group,
    create_results_dataframe,
)
from app._components.plots import (
    plot_zip_scatter,
    plot_networkx_graph,
    plot_geographic_communities,
)
from sklearn.preprocessing import StandardScaler
from app._components.zip_context_utils import build_knn_graph

# Import shared graph cache utilities
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
# Data loading and validation
# ---------------------------------------------------------------------
from app._logic.loader import ensure_data_loaded

if not ensure_data_loaded():
    st.stop()

zipc = st.session_state.get("zip_df")
zip_coords = st.session_state.get("zip_coords")
wa_boundary = st.session_state.get("wa_boundary")

zipc = zipc.reset_index(drop=True)

# Load feature groups (default + custom)
default_fg = None
if "default_feature_groups" in st.session_state:
    default_fg = st.session_state["default_feature_groups"]
else:
    # fallback: try to load from disk
    default_fg_path = Path("data/config/default_feature_groups.json")
    if default_fg_path.exists():
        import json
        with open(default_fg_path, "r") as f:
            default_fg = json.load(f)

# Validate feature groups
if not isinstance(default_fg, dict):
    st.error("Default feature groups are not a dictionary. Please check feature group configuration.")
    st.stop()

DEFAULT_FEATURE_GROUPS = default_fg
custom_feature_groups = st.session_state.get("feature_groups", {})
if not isinstance(custom_feature_groups, dict):
    custom_feature_groups = {}

# Combine default and custom feature groups
feature_groups = {**DEFAULT_FEATURE_GROUPS, **custom_feature_groups}

if not feature_groups:
    st.error("No feature groups available. Please configure feature groups first.")
    st.stop()

# Initialize graph cache from disk
initialize_session_cache()

# ---------------------------------------------------------------------
# Settings (top of page)
# ---------------------------------------------------------------------
st.subheader("Comparison Settings")

# Feature groups multi-select (full width)
selected_groups = st.multiselect(
    "Select Feature Groups to Compare",
    options=list(feature_groups.keys()),
    default=None,
)

col1, col2 = st.columns(2)
with col1:
    k = st.number_input(
        "k (k-NN)",
        min_value=1, max_value=20,
        value=int(st.session_state.get("graph_k", 3)),
        step=1
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
        help="Higher values produce more/smaller communities"
    )
    layout_choice = st.selectbox(
        "Graph Layout",
        options=["spring", "kamada", "circular", "random", "shell"],
        index=0,
    )

col_btn1, col_btn2, _ = st.columns([1, 1, 2])
with col_btn1:
    run_batch = st.button("Run Comparison", type="primary")
with col_btn2:
    clear_cache_clicked = st.button("Clear Cache")

# Handle cache clearing
if clear_cache_clicked:
    clear_all_cache()
    st.success("Cache cleared.")
    st.rerun()

st.markdown("---")

# Store selections in session state
if "batch_settings" not in st.session_state:
    st.session_state.batch_settings = {}

st.session_state.batch_settings.update({
    "selected_groups": selected_groups,
    "k": k,
    "knn_type": knn_type,
    "layout": layout_choice,
    "resolution": resolution
})

# Status message
if not selected_groups:
    st.info("Select feature groups above to compare.")

# ---------------------------------------------------------------------
# Batch Processing Logic
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Batch Processing Section
# ---------------------------------------------------------------------

# Process all selected groups when batch button is clicked
if run_batch and selected_groups:
    
    # Initialize progress tracking
    progress_text = "Processing feature groups..."
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process each group
    all_results = []
    group_stats = []
    cache_hits = 0
    cache_misses = 0

    for idx, group_name in enumerate(selected_groups):
        status_text.text(f"Processing group: {group_name}")

        try:
            # Check cache first
            cache_key = (group_name, int(k), knn_type, float(resolution))
            cached_result = get_cached_graph(cache_key)

            if cached_result is not None:
                # Load from cache - MUCH faster!
                cache_hits += 1
                status_text.text(f"Processing group: {group_name} (from cache)")

                # Reconstruct results from cached data WITHOUT expensive recomputation
                graph, features_df = cached_result
                results = reconstruct_results_from_cache(graph, features_df, group_name)
            else:
                # Compute from scratch
                cache_misses += 1

                # Get features and compute graph to save to cache
                sel_cols = get_group_columns(feature_groups[group_name])
                selected_features = present(zipc, sel_cols)

                if selected_features:
                    # Build graph
                    feats = zipc[selected_features].astype(float).values
                    feats = StandardScaler().fit_transform(feats)
                    G = build_knn_graph(feats, k_neighbors=k, knn_type=knn_type)

                    # Call process_zip_group for full results
                    results = process_zip_group(
                        zipc=zipc,
                        group_name=group_name,
                        feature_groups=feature_groups,
                        k=k,
                        knn_type=knn_type,
                        default_groups=DEFAULT_FEATURE_GROUPS,
                        resolution=resolution
                    )

                    # Save to cache for future use
                    if results is not None:
                        # Create normalized features DataFrame for caching
                        out_df = pd.DataFrame({
                            "ZIPCODE": results["ZIPCODE"].astype(str),
                            "environment_index": results.get(f"environment_index_{group_name}"),
                            "ses_index": results.get(f"ses_index_{group_name}"),
                            "zip_community": results.get(f"zip_community_{group_name}"),
                            "zip_betweenness": results.get(f"zip_betweenness_{group_name}"),
                            "zip_pagerank": results.get(f"zip_pagerank_{group_name}"),
                            "zip_degree": results.get(f"degree_{group_name}"),
                            "isolated": results.get(f"isolated_{group_name}"),
                            "environment_index_var": results.get(f"environment_index_var_{group_name}"),
                            "ses_index_var": results.get(f"ses_index_var_{group_name}"),
                            "modularity": results.get(f"modularity_{group_name}")
                        })

                        meta = {
                            "feature_group": group_name,
                            "k": k,
                            "knn_type": knn_type,
                            "resolution": resolution,
                        }

                        save_graph_to_cache(cache_key, G, out_df, meta)
                else:
                    results = None

            if results is not None:
                # Get some basic stats for this group
                # community count (exclude unknown -1)
                n_communities = results[f"zip_community_{group_name}"].nunique()
                # PCA explained variance scalars (same value per-row)
                env_var = None
                ses_var = None
                if f"environment_index_var_{group_name}" in results.columns:
                    env_var = results[f"environment_index_var_{group_name}"].iloc[0]
                if f"ses_index_var_{group_name}" in results.columns:
                    ses_var = results[f"ses_index_var_{group_name}"].iloc[0]

                # Graph summary scalars (stored per-row by the utility)
                def get_result_col(res, grp, col_name):
                    full_name = f"{col_name}_{grp}"
                    return res[full_name].iloc[0] if full_name in res.columns else None

                nodes = get_result_col(results, group_name, "nodes")
                edges = get_result_col(results, group_name, "edges")
                num_comm = get_result_col(results, group_name, "num_communities")
                isolated_nodes = get_result_col(results, group_name, "isolated_nodes")
                n_components = get_result_col(results, group_name, "n_components")
                modularity = get_result_col(results, group_name, "modularity")

                # Compute derived metrics
                non_isolated_communities = None
                if num_comm is not None and isolated_nodes is not None:
                    non_isolated_communities = int(num_comm) - int(isolated_nodes)

                non_isolated_components = None
                if n_components is not None and isolated_nodes is not None:
                    non_isolated_components = int(n_components) - int(isolated_nodes)

                group_stats.append({
                    "group": group_name,
                    "nodes": nodes,
                    "edges": edges,
                    "n_communities": num_comm,
                    "non_isolated_communities": non_isolated_communities,
                    "n_components": n_components,
                    "non_isolated_components": non_isolated_components,
                    "isolated_nodes": isolated_nodes,
                    "modularity": modularity,
                    "env_var": env_var,
                    "ses_var": ses_var,
                })
                
                all_results.append(results)
                
        except Exception as e:
            st.error(f"Error processing group {group_name}: {str(e)}")
            continue
            
        # Update progress
        progress_bar.progress((idx + 1) / len(selected_groups))
    
    # Merge all results and create summaries
    if all_results:
        # Merge all results
        final_results = all_results[0]
        for df in all_results[1:]:
            final_results = final_results.merge(df, on="ZIPCODE", how="outer")
        
        # Store in session state
        st.session_state["batch_results"] = final_results
        st.session_state["batch_settings"] = {
            "k": k,
            "knn_type": knn_type,
            "layout": layout_choice,
            "resolution": resolution,
            "groups": selected_groups
        }

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        # Show cache statistics
        cache_msg = f"Successfully processed {len(all_results)} feature groups!"
        if cache_hits > 0:
            cache_msg += f" ({cache_hits} from cache, {cache_misses} computed)"
        st.success(cache_msg)
        
        # Display group statistics
        st.subheader("Group Processing Summary")
        stats_df = pd.DataFrame(group_stats)
        st.dataframe(stats_df, use_container_width=False, hide_index=True)
        
        # Preview results
        st.subheader("Results")
        
        # Filter out redundant group name columns
        display_cols = [
            col for col in final_results.columns
            if not (col.startswith('environment_index_var_') or
                   col.startswith('ses_index_var_') or
                   col.startswith('nodes_') or
                   col.startswith('edges_')) 
        ]
        
        st.dataframe(
            final_results[display_cols],
            use_container_width=False,
            height=400,
            hide_index=True
        )

        # Per-group collapsible plots (PCA scatter, network, geographic)
        st.subheader("Per-group plots")
        for results, group_name in zip(all_results, selected_groups):
            with st.expander(f"Plots: {group_name}", expanded=False):
                # Build a normalized out DataFrame (no suffix) for plotting
                out = pd.DataFrame({
                    "ZIPCODE": results["ZIPCODE"].astype(str),
                    "environment_index": results.get(f"environment_index_{group_name}"),
                    "ses_index": results.get(f"ses_index_{group_name}"),
                    "zip_community": results.get(f"zip_community_{group_name}"),
                    "zip_betweenness": results.get(f"zip_betweenness_{group_name}"),
                    "zip_pagerank": results.get(f"zip_pagerank_{group_name}"),
                    "zip_degree": results.get(f"degree_{group_name}"),
                    "isolated": results.get(f"isolated_{group_name}"),
                })

                # Recompute features + graph locally to pass to network plot
                sel_cols = get_group_columns(feature_groups[group_name])
                selected_features = present(zipc, sel_cols)
                if selected_features:
                    feats = zipc[selected_features].astype(float).values
                    feats = StandardScaler().fit_transform(feats)
                    G = build_knn_graph(feats, k_neighbors=k, knn_type=knn_type)
                else:
                    G = None

                cols = st.columns(3)

                # 1) PCA scatter — only show if both indices available
                with cols[0]:
                    if ("environment_index" in out.columns and "ses_index" in out.columns and
                            not out["environment_index"].isna().all() and not out["ses_index"].isna().all()):
                        fig_scatter = plot_zip_scatter(
                            out,
                            x_col="environment_index",
                            y_col="ses_index",
                            comm_col="zip_community",
                            size_col="zip_pagerank",
                            title=f"PCA scatter: {group_name}"
                        )
                        st.pyplot(fig_scatter)
                    else:
                        st.info("PCA scatter requires both environment and SES indices — skipping.")

                # 2) Network graph
                with cols[1]:
                    if G is not None:
                        fig_net = plot_networkx_graph(
                            G,
                            out_df=out,
                            node_size=10,
                            edge_width=0.9,
                            edge_alpha=0.5,
                            edge_color="gray",
                            layout=layout_choice,
                            community_col="zip_community",
                            size_col="zip_pagerank",
                            title=f"Network graph: {group_name}",
                            scale_factor=4.0,
                        )
                        st.pyplot(fig_net)
                    else:
                        st.info("Network graph could not be generated (missing features).")

                # 3) Geographic map
                with cols[2]:
                    if (G is not None and zip_coords is not None and wa_boundary is not None and
                        "zip_pagerank" in out.columns and not out["zip_pagerank"].isna().all()):
                        fig_geo = plot_geographic_communities(
                            out,
                            zip_coords,
                            wa_boundary,
                            size_col="zip_pagerank",
                            base_markersize=30
                        )
                        st.pyplot(fig_geo)
                    else:
                        st.info("Geographic map unavailable (missing required data).")
