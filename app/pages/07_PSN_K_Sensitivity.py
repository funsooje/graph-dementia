# app/pages/07_PSN_K_Sensitivity.py

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import community as community_louvain

try:
    from pynndescent import NNDescent
    HAS_PYNNDESCENT = True
except ImportError:
    HAS_PYNNDESCENT = False

# ---------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------
st.set_page_config(page_title="PSN K Sensitivity Analysis", layout="wide")
st.title("PSN K Sensitivity Analysis")

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
SIM_BACKEND_THRESHOLD = 5000

# ---------------------------------------------------------------------
# Helper functions (copied from 06_PSN_Graph.py for independence)
# ---------------------------------------------------------------------
def build_weighted_matrix(X_fused, patient_cols, zip_cols, patient_w: float, zip_w: float):
    """Reweights the fused matrix by block."""
    n_pat = len(patient_cols) if patient_cols is not None else 0
    n_zip = len(zip_cols) if zip_cols is not None else 0

    Xw = X_fused.astype(float).copy()
    if n_pat > 0:
        Xw[:, :n_pat] *= float(patient_w)
    if n_zip > 0:
        Xw[:, n_pat:n_pat + n_zip] *= float(zip_w)
    return Xw


def topk_exact_from_matrix(X: np.ndarray, k: int):
    """Exact cosine similarity top-k neighbors."""
    n, d = X.shape
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, -np.inf)
    idxs = np.empty((n, k), dtype=int)
    sims = np.empty((n, k), dtype=float)
    for i in range(n):
        idx = np.argpartition(sim[i], -k)[-k:]
        idx = idx[np.argsort(sim[i, idx])[::-1]]
        idxs[i] = idx
        sims[i] = sim[i, idx]
    return idxs, sims, sim


def topk_ann_or_exact(X: np.ndarray, k: int, ann_mode: str, sim_threshold: int):
    """Use ANN (PyNNDescent) or exact cosine similarity."""
    n, d = X.shape
    use_ann = False
    if ann_mode == "force_ann" and HAS_PYNNDESCENT:
        use_ann = True
    elif ann_mode == "force_exact":
        use_ann = False
    elif ann_mode == "auto" and n > sim_threshold and HAS_PYNNDESCENT:
        use_ann = True

    if use_ann:
        index = NNDescent(X, metric="cosine", n_neighbors=k+1, random_state=42)
        nbrs_idx, nbrs_dist = index.query(X, k=k+1)
        idxs = np.zeros((n, k), dtype=int)
        sims = np.zeros((n, k), dtype=float)
        for i in range(n):
            row_idx = nbrs_idx[i].tolist()
            row_dst = nbrs_dist[i].tolist()
            cleaned = [(j, d_) for j, d_ in zip(row_idx, row_dst) if j != i]
            cleaned = cleaned[:k] if len(cleaned) >= k else cleaned
            while len(cleaned) < k:
                cleaned.append((i, 1.0))
            idxs[i] = [j for j, _ in cleaned]
            sims[i] = [1.0 - d_ for _, d_ in cleaned]
        return idxs, sims, None

    return topk_exact_from_matrix(X, k)


def build_knn_graph_from_neighbors(topk_idx: np.ndarray, topk_sim: np.ndarray, knn_type: str):
    """Build k-NN graph from neighbor lists."""
    n, k = topk_idx.shape
    if knn_type == "directed":
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for r in range(k):
                j = int(topk_idx[i, r])
                w = float(topk_sim[i, r])
                if np.isfinite(w):
                    G.add_edge(i, j, weight=w)
        return G

    neighbor_sets = [set(topk_idx[i]) for i in range(n)]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in neighbor_sets[i]:
            if i < j and i in neighbor_sets[j]:
                wi = float(topk_sim[i, np.where(topk_idx[i] == j)[0][0]])
                wj = float(topk_sim[j, np.where(topk_idx[j] == i)[0][0]])
                w = (wi + wj) / 2.0
                if np.isfinite(w):
                    G.add_edge(i, int(j), weight=w)
    return G


def compute_psn_metrics(G: nx.Graph):
    """Compute graph metrics for PSN sensitivity analysis."""
    n = G.number_of_nodes()
    G_u = G.to_undirected() if G.is_directed() else G

    # Community detection
    partition = community_louvain.best_partition(G_u, weight="weight", random_state=42)

    # Metrics
    edges = G.number_of_edges()
    num_communities = len(set(partition.values()))
    isolated_nodes = sum(1 for _ in nx.isolates(G_u))
    n_components = nx.number_connected_components(G_u)
    non_isolated_communities = num_communities - isolated_nodes
    non_isolated_components = n_components - isolated_nodes

    # Modularity
    try:
        modularity = community_louvain.modularity(partition, G_u, weight="weight")
    except Exception:
        modularity = None

    # Average degree
    avg_degree = float(np.mean([d for _, d in G.degree(weight=None)])) if n > 0 else 0.0

    return {
        "edges": edges,
        "n_communities": num_communities,
        "non_isolated_communities": non_isolated_communities,
        "n_components": n_components,
        "non_isolated_components": non_isolated_components,
        "isolated_nodes": isolated_nodes,
        "modularity": round(modularity, 4) if modularity is not None else None,
        "avg_degree": round(avg_degree, 2),
    }


# ---------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------
missing = []
X_fused = st.session_state.get("pf_fused_matrix")
tbl_fused = st.session_state.get("pf_fused_table")
pat_cols = st.session_state.get("pf_patient_block_cols")
zip_cols = st.session_state.get("pf_zip_block_cols")

if X_fused is None:
    missing.append("pf_fused_matrix")
if tbl_fused is None:
    missing.append("pf_fused_table")
if pat_cols is None:
    missing.append("pf_patient_block_cols")
if zip_cols is None:
    zip_cols = []  # Neighborhood block is optional

if missing:
    st.error(
        "Missing required inputs from PSN Feature Selection: "
        + ", ".join(missing)
        + ". Go to page 05 and click 'Generate PSN Features'."
    )
    st.stop()

# Show data summary
st.subheader("Data Summary")
n_rows, n_cols = X_fused.shape if isinstance(X_fused, np.ndarray) else (None, None)
summary_df = pd.DataFrame([
    {"Metric": "PSN Matrix Shape", "Value": f"{n_rows} × {n_cols}"},
    {"Metric": "Profile Block Cols", "Value": len(pat_cols)},
    {"Metric": "Neighborhood Block Cols", "Value": len(zip_cols)},
])
st.dataframe(summary_df, use_container_width=False, hide_index=True)

st.divider()

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
st.subheader("Analysis Settings")

col1, col2 = st.columns(2)

with col1:
    k_values_input = st.text_input(
        "k values (comma-separated)",
        value="3, 5, 7, 10",
        help="Enter k values separated by commas, e.g., 3, 5, 7, 10"
    )

with col2:
    weight_values_input = st.text_input(
        "Weight balances (comma-separated)",
        value="0.0, 0.3, 0.5, 0.7, 1.0",
        help="Neighborhood weight (0.0 = 100% Profile, 1.0 = 100% Neighborhood)"
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
    ann_mode_label = st.selectbox(
        "Similarity Mode",
        options=["Auto (threshold)", "Force ANN", "Force Exact"],
        index=0,
        help="Auto uses ANN for large datasets"
    )
    ann_mode = (
        "auto" if ann_mode_label.startswith("Auto") else
        "force_ann" if ann_mode_label.startswith("Force ANN") else
        "force_exact"
    )

run_analysis = st.button("Run Analysis", type="primary")

st.markdown("---")

# Parse k values
try:
    k_values = [int(k.strip()) for k in k_values_input.split(",") if k.strip()]
    k_values = [k for k in k_values if 1 <= k <= 50]
except ValueError:
    k_values = []

# Parse weight values
try:
    weight_values = [float(w.strip()) for w in weight_values_input.split(",") if w.strip()]
    weight_values = [w for w in weight_values if 0.0 <= w <= 1.0]
except ValueError:
    weight_values = []

if not k_values:
    st.warning("Please enter valid k values (integers between 1 and 50).")

if not weight_values:
    st.warning("Please enter valid weight values (floats between 0.0 and 1.0).")

# ---------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------
METRICS = [
    "edges",
    "n_communities",
    "non_isolated_communities",
    "n_components",
    "non_isolated_components",
    "isolated_nodes",
    "modularity",
    "avg_degree",
]

if run_analysis and k_values and weight_values:
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Store results: {(weight, k): {metric: value}}
    results_dict = {}
    total_iterations = len(weight_values) * len(k_values)
    current_iteration = 0

    for weight in weight_values:
        patient_w = 1.0 - weight
        zip_w = weight

        # Build weighted matrix for this weight
        X_weighted = build_weighted_matrix(
            X_fused=X_fused,
            patient_cols=pat_cols,
            zip_cols=zip_cols,
            patient_w=patient_w,
            zip_w=zip_w,
        )

        for k in k_values:
            status_text.text(f"Processing: weight={weight:.2f}, k={k}")

            try:
                # Get neighbors
                idxs, sims, _ = topk_ann_or_exact(
                    X_weighted, k, ann_mode, SIM_BACKEND_THRESHOLD
                )

                # Build graph
                G = build_knn_graph_from_neighbors(idxs, sims, knn_type)

                # Compute metrics
                metrics = compute_psn_metrics(G)
                results_dict[(weight, k)] = metrics

            except Exception as e:
                st.error(f"Error processing weight={weight:.2f}, k={k}: {str(e)}")
                results_dict[(weight, k)] = {m: None for m in METRICS}

            current_iteration += 1
            progress_bar.progress(current_iteration / total_iterations)

    progress_bar.empty()
    status_text.empty()

    # Build cross-tab DataFrame
    rows = []
    for idx, weight in enumerate(weight_values):
        weight_label = f"w={weight:.2f} (P:{1-weight:.0%}, N:{weight:.0%})"
        for metric_idx, metric in enumerate(METRICS):
            row = {
                "Weight": weight_label if metric_idx == 0 else "",
                "Metric": metric
            }
            for k in k_values:
                value = results_dict.get((weight, k), {}).get(metric)
                row[f"k={k}"] = value
            rows.append(row)

        # Add blank row between weight groups (except after last)
        if idx < len(weight_values) - 1:
            blank_row = {"Weight": "", "Metric": ""}
            for k in k_values:
                blank_row[f"k={k}"] = ""
            rows.append(blank_row)

    crosstab_df = pd.DataFrame(rows)

    # Store in session state
    st.session_state["psn_k_sensitivity_results"] = crosstab_df
    st.session_state["psn_k_sensitivity_settings"] = {
        "k_values": k_values,
        "weight_values": weight_values,
        "knn_type": knn_type,
        "ann_mode": ann_mode,
    }

    st.success(
        f"Analysis complete: {len(weight_values)} weights × {len(k_values)} k values"
    )

# ---------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------
if "psn_k_sensitivity_results" in st.session_state:
    crosstab_df = st.session_state["psn_k_sensitivity_results"]
    settings = st.session_state.get("psn_k_sensitivity_settings", {})

    st.subheader("Cross-Tab Results")

    if settings:
        st.caption(
            f"Graph type: {settings.get('knn_type', 'N/A')}, "
            f"Similarity mode: {settings.get('ann_mode', 'N/A')}"
        )

    # Calculate height to show all rows
    table_height = (len(crosstab_df) + 1) * 35 + 10
    st.dataframe(crosstab_df, use_container_width=True, hide_index=True, height=table_height)

    # Download button
    csv = crosstab_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="psn_k_sensitivity_analysis.csv",
        mime="text/csv",
    )
