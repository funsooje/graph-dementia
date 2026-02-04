# app/pages/04_K_Sensitivity_Analysis.py

from pathlib import Path
import json
import streamlit as st
import pandas as pd

from app._components.zip_context_utils import (
    process_zip_group,
    present,
    get_group_columns,
    build_knn_graph,
)
from sklearn.preprocessing import StandardScaler

# Import shared graph cache utilities
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
# Data loading and validation
# ---------------------------------------------------------------------
from app._logic.loader import ensure_data_loaded

if not ensure_data_loaded():
    st.stop()

zipc = st.session_state.get("zip_df")

zipc = zipc.reset_index(drop=True)

# Load feature groups (default + custom)
default_fg = None
if "default_feature_groups" in st.session_state:
    default_fg = st.session_state["default_feature_groups"]
else:
    default_fg_path = Path("data/config/default_feature_groups.json")
    if default_fg_path.exists():
        with open(default_fg_path, "r") as f:
            default_fg = json.load(f)

if not isinstance(default_fg, dict):
    st.error("Default feature groups are not a dictionary. Please check feature group configuration.")
    st.stop()

DEFAULT_FEATURE_GROUPS = default_fg
custom_feature_groups = st.session_state.get("feature_groups", {})
if not isinstance(custom_feature_groups, dict):
    custom_feature_groups = {}

feature_groups = {**DEFAULT_FEATURE_GROUPS, **custom_feature_groups}

if not feature_groups:
    st.error("No feature groups available. Please configure feature groups first.")
    st.stop()

# Initialize graph cache from disk
initialize_session_cache()

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
st.subheader("Analysis Settings")

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
        help="Enter k values separated by commas, e.g., 3, 5, 7, 10"
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
        value=1.0,
        step=0.1, format="%.2f",
        help="Higher values produce more/smaller communities"
    )

col_btn1, col_btn2, _ = st.columns([1, 1, 2])
with col_btn1:
    run_analysis = st.button("Run Analysis", type="primary")
with col_btn2:
    clear_cache_clicked = st.button("Clear Cache")

# Handle cache clearing
if clear_cache_clicked:
    clear_all_cache()
    st.success("Cache cleared.")
    st.rerun()

st.markdown("---")

# Parse k values
try:
    k_values = [int(k.strip()) for k in k_values_input.split(",") if k.strip()]
    k_values = [k for k in k_values if 1 <= k <= 20]
except ValueError:
    k_values = []

if not k_values:
    st.warning("Please enter valid k values (integers between 1 and 20).")

if not selected_groups:
    st.info("Select feature groups above to analyze.")

# ---------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------

METRICS = [
    "nodes",
    "edges",
    "n_communities",
    "non_isolated_communities",
    "n_components",
    "non_isolated_components",
    "isolated_nodes",
    "modularity",
]

if run_analysis and selected_groups and k_values:

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Store results: {(group, k): {metric: value}}
    results_dict = {}
    total_iterations = len(selected_groups) * len(k_values)
    current_iteration = 0
    cache_hits = 0
    cache_misses = 0

    for group_name in selected_groups:
        for k in k_values:
            # Check cache first
            cache_key = (group_name, int(k), knn_type, float(resolution))
            cached_result = get_cached_graph(cache_key)

            if cached_result is not None:
                cache_hits += 1
                status_text.text(f"Processing: {group_name}, k={k} (from cache)")

                # Reconstruct from cache WITHOUT expensive recomputation
                try:
                    graph, features_df = cached_result
                    results = reconstruct_results_from_cache(graph, features_df, group_name)
                except Exception as e:
                    st.warning(f"Cache error for {group_name}, k={k}: {e}. Recomputing...")
                    cache_misses += 1
                    cache_hits -= 1
                    results = process_zip_group(
                        zipc=zipc,
                        group_name=group_name,
                        feature_groups=feature_groups,
                        k=k,
                        knn_type=knn_type,
                        default_groups=DEFAULT_FEATURE_GROUPS,
                        resolution=resolution
                    )
            else:
                cache_misses += 1
                status_text.text(f"Processing: {group_name}, k={k}")

                try:
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

                except Exception as e:
                    st.error(f"Error processing {group_name}, k={k}: {str(e)}")
                    results = None

            # Extract metrics if results exist
            try:

                if results is not None:
                    # Extract metrics helper
                    def get_result_col(res, grp, col_name):
                        full_name = f"{col_name}_{grp}"
                        return res[full_name].iloc[0] if full_name in res.columns else None

                    nodes = get_result_col(results, group_name, "nodes")
                    edges = get_result_col(results, group_name, "edges")
                    num_communities = get_result_col(results, group_name, "num_communities")
                    isolated_nodes = get_result_col(results, group_name, "isolated_nodes")
                    n_components = get_result_col(results, group_name, "n_components")
                    modularity = get_result_col(results, group_name, "modularity")

                    # Derived metrics
                    non_isolated_communities = None
                    if num_communities is not None and isolated_nodes is not None:
                        non_isolated_communities = int(num_communities) - int(isolated_nodes)

                    non_isolated_components = None
                    if n_components is not None and isolated_nodes is not None:
                        non_isolated_components = int(n_components) - int(isolated_nodes)

                    results_dict[(group_name, k)] = {
                        "nodes": nodes,
                        "edges": edges,
                        "n_communities": num_communities,
                        "non_isolated_communities": non_isolated_communities,
                        "n_components": n_components,
                        "non_isolated_components": non_isolated_components,
                        "isolated_nodes": isolated_nodes,
                        "modularity": round(modularity, 4) if modularity is not None else None,
                    }
                else:
                    results_dict[(group_name, k)] = {m: None for m in METRICS}

            except Exception as e:
                st.error(f"Error processing {group_name}, k={k}: {str(e)}")
                results_dict[(group_name, k)] = {m: None for m in METRICS}

            current_iteration += 1
            progress_bar.progress(current_iteration / total_iterations)

    progress_bar.empty()
    status_text.empty()

    # Show cache statistics
    total_runs = len(selected_groups) * len(k_values)
    cache_msg = f"Analysis complete: {len(selected_groups)} groups x {len(k_values)} k values"
    if cache_hits > 0:
        cache_msg += f" ({cache_hits} from cache, {cache_misses} computed)"
    st.success(cache_msg)

    # Build cross-tab DataFrame
    rows = []
    for idx, group_name in enumerate(selected_groups):
        for metric_idx, metric in enumerate(METRICS):
            # Only show group name on first row
            row = {
                "Group": group_name if metric_idx == 0 else "",
                "Metric": metric
            }
            for k in k_values:
                value = results_dict.get((group_name, k), {}).get(metric)
                row[f"k={k}"] = value
            rows.append(row)

        # Add blank row between groups (except after last group)
        if idx < len(selected_groups) - 1:
            blank_row = {"Group": "", "Metric": ""}
            for k in k_values:
                blank_row[f"k={k}"] = ""
            rows.append(blank_row)

    crosstab_df = pd.DataFrame(rows)

    # Store in session state
    st.session_state["k_sensitivity_results"] = crosstab_df
    st.session_state["k_sensitivity_settings"] = {
        "groups": selected_groups,
        "k_values": k_values,
        "knn_type": knn_type,
        "resolution": resolution,
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
            f"Resolution: {settings.get('resolution', 'N/A')}"
        )

    # Calculate height to show all rows without scrolling (approx 35px per row + header)
    table_height = (len(crosstab_df) + 1) * 35 + 10
    st.dataframe(crosstab_df, use_container_width=True, hide_index=True, height=table_height)

    # Download button
    csv = crosstab_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="k_sensitivity_analysis.csv",
        mime="text/csv",
    )
