import streamlit as st
import pandas as pd
import json
import networkx as nx

from app._components.plots import (
    plot_zip_scatter,
    plot_networkx_graph,
    plot_geographic_communities,
)

from app._components.zip_context_utils import (
    present,
    pca_first_component,
    build_knn_graph,
    compute_graph_metrics,
)

def load_feature_groups():
    # Load default feature groups from session_state or JSON file
    if "default_feature_groups" in st.session_state:
        default_groups = st.session_state.default_feature_groups
    else:
        try:
            with open("app/data/default_feature_groups.json") as f:
                default_groups = json.load(f)
        except Exception:
            default_groups = {}
    # Load custom feature groups from session_state or JSON file
    if "custom_feature_groups" in st.session_state:
        custom_groups = st.session_state.custom_feature_groups
    else:
        try:
            with open("app/data/custom_feature_groups.json") as f:
                custom_groups = json.load(f)
        except Exception:
            custom_groups = {}
    # Merge dictionaries
    all_groups = {**default_groups, **custom_groups}
    return all_groups


# --- Streamlit Page Code (executes top-to-bottom) ---
st.title("Batch Run ZIP Context Processing")

all_feature_groups = load_feature_groups()
feature_group_names = list(all_feature_groups.keys())

st.sidebar.header("Batch Processing Parameters")

selected_groups = st.sidebar.multiselect(
    "Select Feature Groups to Run",
    options=feature_group_names,
    default=feature_group_names,
)

k = st.sidebar.number_input("k (number of neighbors)", min_value=1, max_value=30, value=5, step=1)
knn_type = st.sidebar.selectbox("kNN Graph Type", options=["mutual", "directed"])
layout_type = st.sidebar.selectbox("Network Layout", options=["spring", "circular", "kamada_kawai"])

if st.button("Run Batch"):

    if not selected_groups:
        st.warning("Please select at least one feature group to run.")
    else:
        pca_variance_list = []
        graph_summary_list = []
        derived_features_list = []

        for group_name in selected_groups:
            group_info = all_feature_groups.get(group_name, {})
            short_code = group_info.get("short_code", group_name[:3].upper())
            feature_cols = group_info.get("columns", [])

            if not feature_cols:
                st.warning(f"No feature columns found for group '{group_name}'. Skipping.")
                continue

            # Check if all columns exist in the ZIP data
            zip_data = st.session_state.get("zip_df")
            if zip_data is None:
                st.error("ZIP data not found in session_state. Please load ZIP data first.")
                break

            missing_cols = [col for col in feature_cols if col not in zip_data.columns]
            if missing_cols:
                st.warning(f"Missing columns for group '{group_name}': {missing_cols}. Skipping this group.")
                continue

            # Run PCA first component
            pca_result = pca_first_component(zip_data, feature_cols)
            if pca_result is None:
                st.warning(f"PCA failed for group '{group_name}'. Skipping.")
                continue
            pc1_scores, variance_explained, used_cols = pca_result

            # Build kNN graph
            graph = build_knn_graph(pc1_scores.reshape(-1, 1), k_neighbors=k, knn_type=knn_type)
            if graph is None:
                st.warning(f"Graph construction failed for group '{group_name}'. Skipping.")
                continue

            # Compute graph metrics
            partition, betweenness, pagerank = compute_graph_metrics(graph)
            graph_metrics = {
                "Nodes": graph.number_of_nodes(),
                "Edges": graph.number_of_edges(),
                "Communities": len(set(partition.values())) if partition else 0,
                "Isolated": len(list(nx.isolates(graph))),
            }

            # Prefix derived feature column names with short_code
            derived_df = pd.DataFrame(pc1_scores, columns=["PC1"])
            derived_df = derived_df.add_prefix(f"{short_code}_")
            derived_df.index = zip_data.index

            derived_features_list.append(derived_df)

            # Record PCA variance explained
            pca_variance_list.append(
                {
                    "Feature Group": group_name,
                    "Short Code": short_code,
                    "Variance Explained": variance_explained,
                }
            )

            # Record graph summary metrics
            graph_summary = {"Feature Group": group_name, "Short Code": short_code}
            graph_summary.update(graph_metrics)
            graph_summary_list.append(graph_summary)

            # Generate and display plots
            st.subheader(f"Plots for Feature Group: {group_name} ({short_code})")
            scatter_fig = plot_zip_scatter(zip_data, pc1_scores, title=f"PCA Scatter - {group_name}")
            st.pyplot(scatter_fig)

            network_fig = plot_networkx_graph(graph, layout=layout_type, title=f"ZIP Network - {group_name}")
            st.pyplot(network_fig)

            map_fig = plot_geographic_communities(zip_data, pc1_scores, title=f"ZIP Map - {group_name}")
            st.pyplot(map_fig)

        # Display summary tables
        if pca_variance_list:
            st.header("PCA Variance Explained Summary")
            pca_variance_df = pd.DataFrame(pca_variance_list)
            st.dataframe(pca_variance_df)

        if graph_summary_list:
            st.header("Graph Summary Metrics")
            graph_summary_df = pd.DataFrame(graph_summary_list)
            st.dataframe(graph_summary_df)

        if derived_features_list:
            st.header("Derived ZIP Features (Merged)")
            merged_df = pd.concat(derived_features_list, axis=1)
            st.dataframe(merged_df.style.hide_index())

            # Provide download button for merged data
            csv = merged_df.to_csv(index=False)
            st.download_button(
                label="Download Derived ZIP Features as CSV",
                data=csv,
                file_name="derived_zip_features.csv",
                mime="text/csv",
            )
