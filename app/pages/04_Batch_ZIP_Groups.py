# app/pages/04_Batch_ZIP_Groups.py

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

# ---------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------
st.set_page_config(page_title="Batch ZIP Groups", layout="wide")
st.title("Batch ZIP Groups Analysis")

# ---------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------
# Load required dataframes from session state
zipc = st.session_state.get("zip_df")
zip_coords = st.session_state.get("zip_coords")
wa_boundary = st.session_state.get("wa_boundary")

if zipc is None or zip_coords is None or wa_boundary is None:
    st.error("Required dataframes not found in session_state. Ensure Home.py loads them at startup.")
    st.stop()

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

# ---------------------------------------------------------------------
# User Interface Setup (Sidebar)
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Batch Processing Settings")
    
    # Multi-select for feature groups
    selected_groups = st.multiselect(
        "Select Feature Groups to Process",
        options=list(feature_groups.keys()),
        default=None,
    )
    
    # k-value selection
    k = st.number_input(
            "k (k-NN)",
            min_value=1, max_value=20, value=int(st.session_state.get("graph_k", 3)), step=1
        )
    
    # Graph type selection
    knn_type_label = st.radio(
        "Graph Type",
        options=["Mutual k-NN", "Directed k-NN"],
    )
    knn_type = "mutual" if knn_type_label.startswith("Mutual") else "directed"
    
    # Layout type selection
    layout_choice = st.selectbox(
        "Graph Layout",
        options=["spring", "kamada", "circular", "random", "shell"],
        index=0,
    )
    
    # Run batch button
    run_batch = st.button("Run Batch Processing", type="primary")

# Store selections in session state
if "batch_settings" not in st.session_state:
    st.session_state.batch_settings = {}

st.session_state.batch_settings.update({
    "selected_groups": selected_groups,
    "k": k,
    "knn_type": knn_type,
    "layout": layout_choice
})

# Main area status
if not selected_groups:
    st.warning("Please select at least one feature group to begin batch processing.")
else:
    st.info(f"Ready to process {len(selected_groups)} feature groups with k={k}, {knn_type} graph type, and {layout_choice} layout.")

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
    
    for idx, group_name in enumerate(selected_groups):
        status_text.text(f"Processing group: {group_name}")
        
        try:
            # Process using utility function
            results = process_zip_group(
                zipc=zipc,
                group_name=group_name,
                feature_groups=feature_groups,
                k=k,
                knn_type=knn_type,
                default_groups=DEFAULT_FEATURE_GROUPS
            )
            
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
                nodes = (
                    results[f"nodes_{group_name}"].iloc[0]
                    if f"nodes_{group_name}" in results.columns
                    else None
                )
                edges = (
                    results[f"edges_{group_name}"].iloc[0]
                    if f"edges_{group_name}" in results.columns
                    else None
                )
                num_comm = (
                    results[f"num_communities_{group_name}"].iloc[0]
                    if f"num_communities_{group_name}" in results.columns
                    else None
                )
                isolated_nodes = (
                    results[f"isolated_nodes_{group_name}"].iloc[0]
                    if f"isolated_nodes_{group_name}" in results.columns
                    else None
                )
                is_connected = (
                    results[f"is_connected_{group_name}"].iloc[0]
                    if f"is_connected_{group_name}" in results.columns
                    else None
                )

                group_stats.append({
                    "group": group_name,
                    "n_communities": n_communities,
                    "environment_index_var": env_var,
                    "ses_index_var": ses_var,
                    "nodes": nodes,
                    "edges": edges,
                    "num_communities": num_comm,
                    "isolated_nodes": isolated_nodes,
                    "is_connected": is_connected,
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
            "groups": selected_groups
        }
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        st.success(f"Successfully processed {len(all_results)} feature groups!")
        
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
