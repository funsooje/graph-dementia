# app/pages/06_PSN_Graph.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import networkx as nx
import community as community_louvain

from app._components.plots import (
    plot_networkx_graph,          # returns a Matplotlib fig
    plot_profile_scatter_embed,   # returns a Matplotlib fig (PCA 2D)
)

# Import PSN graph building utilities
from app._logic.psn_graph_builder import (
    build_weighted_matrix,
    topk_ann_or_exact,
    topk_mixed_similarity,
    build_knn_graph_from_neighbors,
    compute_graph_metrics,
    HAS_PYNNDESCENT,
)



# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(page_title="PSN Graph", layout="wide")
st.title("PSN Graph")

# ---------------------------------------------------------------------
# Constants (ANN + plotting gates)
# ---------------------------------------------------------------------
SIM_BACKEND_THRESHOLD = 5000   # switch to ANN when n > this (for Auto mode)
PLOT_GRAPH_THRESHOLD  = 500    # skip full network plot when n > this
BTW_AUTO_THRESHOLD    = 5000   # switch to approximate betweenness when n >= this

# ---------------------------------------------------------------------
# Disk caches
# ---------------------------------------------------------------------
PATIENT_FIG_DIR = Path("data/cache/patient_figs")
PATIENT_FIG_DIR.mkdir(parents=True, exist_ok=True)

PATIENT_CACHE_DIR = Path("data/cache/patient_graphs")
PATIENT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def fig_to_png_file(fig, path: Path, dpi: int = 110) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return str(path)

def np_to_npy_file(arr: np.ndarray, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return str(path)

# patient figure cache in session (paths)
if "patient_fig_cache" not in st.session_state:
    st.session_state["patient_fig_cache"] = {"network": {}, "scatter": {}}
fig_cache = st.session_state["patient_fig_cache"]

# ---------------------------------------------------------------------
# Phase 0 — Preconditions (fast guardrails)
# ---------------------------------------------------------------------
missing = []

X_fused = st.session_state.get("pf_fused_matrix")        # numpy array
tbl_fused = st.session_state.get("pf_fused_table")       # DataFrame
idx_fused = st.session_state.get("pf_fused_index")       # list[str]
cnt_fused = st.session_state.get("pf_fused_counts")      # list[int/float]

pat_cols = st.session_state.get("pf_patient_block_cols") # list[str]
zip_cols = st.session_state.get("pf_zip_block_cols")     # list[str]

if X_fused is None:
    missing.append("pf_fused_matrix")
if tbl_fused is None:
    missing.append("pf_fused_table")
if idx_fused is None:
    missing.append("pf_fused_index")
if cnt_fused is None:
    missing.append("pf_fused_counts")
if pat_cols is None:
    missing.append("pf_patient_block_cols")
if zip_cols is None:
    st.info("Notice: Neighborhood block columns not found. Will default to zero columns.")
    zip_cols = []

if missing:
    st.error(
        "Missing required inputs from PSN Feature Selection: "
        + ", ".join(missing)
        + ". Go to page 05 and click 'Generate PSN Features'."
    )
    st.stop()

# Quick preflight summary (no heavy work)
st.subheader("Inputs Summary")
n_rows, n_cols = (X_fused.shape if isinstance(X_fused, np.ndarray) else (None, None))

# Get encoding mode from session state
encoding_meta = st.session_state.get("pf_encoding_metadata", {})
encoding_mode = encoding_meta.get("mode", "unknown")

inputs_df = pd.DataFrame([
    {"Metric": "PSN Matrix Shape", "Value": f"{n_rows} × {n_cols}"},
    {"Metric": "PSN Table Rows", "Value": len(tbl_fused)},
    {"Metric": "Profile Block Cols", "Value": len(pat_cols)},
    {"Metric": "Neighborhood Block Cols", "Value": len(zip_cols)},
    {"Metric": "Index Length", "Value": len(idx_fused)},
    {"Metric": "Encoding Mode", "Value": encoding_mode.capitalize()},
])
st.dataframe(inputs_df, use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------
# Initialize session state defaults
# ---------------------------------------------------------------------
if "patient_block_weight" not in st.session_state:
    st.session_state["patient_block_weight"] = 0.70
if "zip_block_weight" not in st.session_state:
    st.session_state["zip_block_weight"] = 0.30
if "patient_graph_k" not in st.session_state:
    st.session_state["patient_graph_k"] = 3
if "patient_knn_type" not in st.session_state:
    st.session_state["patient_knn_type"] = "mutual"
if "patient_graph_layout" not in st.session_state:
    st.session_state["patient_graph_layout"] = "spring"
if "patient_ann_mode" not in st.session_state:
    st.session_state["patient_ann_mode"] = "auto"
if "patient_sim_threshold" not in st.session_state:
    st.session_state["patient_sim_threshold"] = SIM_BACKEND_THRESHOLD
if "patient_plot_threshold" not in st.session_state:
    st.session_state["patient_plot_threshold"] = PLOT_GRAPH_THRESHOLD
if "patient_btw_mode" not in st.session_state:
    st.session_state["patient_btw_mode"] = "auto"
if "patient_btw_threshold" not in st.session_state:
    st.session_state["patient_btw_threshold"] = BTW_AUTO_THRESHOLD
if "patient_btw_k" not in st.session_state:
    st.session_state["patient_btw_k"] = 400
if "scatter_color_by" not in st.session_state:
    st.session_state["scatter_color_by"] = "profile_community"
if "patient_similarity_metric" not in st.session_state:
    st.session_state["patient_similarity_metric"] = "cosine"

# ---------------------------------------------------------------------
# Main page controls
# ---------------------------------------------------------------------
st.header("Configuration")

# --- Core Graph Settings (always visible) ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    default_zip_w = float(st.session_state["zip_block_weight"])
    weight_balance = st.slider(
        "Weight balance (Profile ← → Neighborhood)",
        min_value=0.0,
        max_value=1.0,
        value=default_zip_w,
        step=0.05,
        help="0.0 = 100% Profile, 1.0 = 100% Neighborhood"
    )
    patient_w = 1.0 - weight_balance
    zip_w = weight_balance
    st.caption(f"Profile: {patient_w:.2f} | Neighborhood: {zip_w:.2f}")

with col2:
    default_k = int(st.session_state["patient_graph_k"])
    k = st.number_input("k (k-NN)", min_value=1, max_value=50, value=default_k, step=1)

with col3:
    knn_type_map = {"Mutual k-NN (undirected)": "mutual", "Directed k-NN": "directed"}
    default_knn = st.session_state["patient_knn_type"]
    knn_label_default = {v: lbl for lbl, v in knn_type_map.items()}.get(default_knn, "Mutual k-NN (undirected)")
    knn_label = st.selectbox("k-NN type", list(knn_type_map.keys()),
                             index=list(knn_type_map.keys()).index(knn_label_default))
    knn_type = knn_type_map[knn_label]

with col4:
    default_layout = st.session_state["patient_graph_layout"]
    layout = st.selectbox(
        "Layout",
        ["spring", "kamada", "circular", "random", "shell"],
        index=["spring", "kamada", "circular", "random", "shell"].index(default_layout),
    )

# --- Similarity Settings (collapsible) ---
with st.expander("Similarity Settings", expanded=False):
    st.caption("Configure how profile similarity is computed")
    sim_col1, sim_col2 = st.columns(2)

    with sim_col1:
        ann_mode_options = ["Auto (threshold)", "Force ANN", "Force Exact"]
        ann_mode_default = st.session_state.get("patient_ann_mode", "auto")
        ann_mode_idx = {"auto": 0, "force_ann": 1, "force_exact": 2}.get(ann_mode_default, 0)
        ann_mode_label = st.selectbox(
            "Similarity mode",
            ann_mode_options,
            index=ann_mode_idx,
            help="Auto: use threshold; Force ANN: always approximate; Force Exact: always exact"
        )

    with sim_col2:
        sim_thresh = st.number_input(
            "ANN threshold (Auto only)",
            min_value=1000, max_value=100000, step=500,
            value=int(st.session_state["patient_sim_threshold"]),
            help="Switch to ANN when n > threshold",
            disabled=(ann_mode_label != "Auto (threshold)")
        )

    # Similarity metric selector (shown only for experimental encoding)
    encoding_meta = st.session_state.get("pf_encoding_metadata", {})
    encoding_mode = encoding_meta.get("mode", "standard")

    if encoding_mode == "experimental":
        st.markdown("**Similarity Metric**")
        st.caption("Experimental encoding detected. Choose similarity metric:")

        metric_options = [
            "Cosine Similarity (default)",
            "Mixed Similarity (recommended for experimental)"
        ]
        metric_default = st.session_state.get("patient_similarity_metric", "cosine")
        metric_idx = 0 if metric_default == "cosine" else 1

        metric_choice = st.radio(
            "Metric",
            metric_options,
            index=metric_idx,
            help=(
                "Cosine: Standard cosine similarity (faster, works on any encoding).\n\n"
                "Mixed: Custom metric combining exact-match for categoricals + "
                "Hamming distance for bitflags + cosine for numeric (slower, more accurate for experimental encoding)."
            ),
            label_visibility="collapsed"
        )

# --- Betweenness Settings (collapsible) ---
with st.expander("Betweenness Centrality Settings", expanded=False):
    st.caption("Configure betweenness centrality computation (can be slow on large graphs)")
    btw_col1, btw_col2, btw_col3 = st.columns(3)

    with btw_col1:
        btw_mode_options = ["Auto (threshold)", "Approximate", "Exact", "Skip"]
        btw_mode_default = st.session_state.get("patient_btw_mode", "auto")
        btw_mode_idx = {"auto": 0, "approx": 1, "exact": 2, "skip": 3}.get(btw_mode_default, 0)
        btw_mode_label = st.selectbox(
            "Betweenness mode",
            btw_mode_options,
            index=btw_mode_idx,
            help="Auto: use threshold to decide; Exact is slow on large graphs"
        )

    with btw_col2:
        btw_thresh = st.number_input(
            "Betweenness threshold (Auto only)",
            min_value=1000, max_value=100000, step=500,
            value=int(st.session_state.get("patient_btw_threshold", BTW_AUTO_THRESHOLD)),
            help="Use approximate when n >= threshold",
            disabled=(btw_mode_label != "Auto (threshold)")
        )

    with btw_col3:
        btw_sample_k = st.number_input(
            "Betweenness samples",
            min_value=50, max_value=5000, step=50,
            value=int(st.session_state["patient_btw_k"]),
            help="Number of source nodes for approximate betweenness"
        )

# --- Display Settings (collapsible) ---
with st.expander("Display Settings", expanded=False):
    st.caption("Configure visualization options")
    disp_col1, disp_col2 = st.columns(2)

    with disp_col1:
        plot_thresh = st.number_input(
            "Plot threshold",
            min_value=100, max_value=100000, step=100,
            value=int(st.session_state["patient_plot_threshold"]),
            help="Skip network plot when n > threshold"
        )

    with disp_col2:
        # Heuristic candidates for color options
        cat_low_card = [
            c for c in tbl_fused.columns
            if (tbl_fused[c].dtype == "object" or
                pd.api.types.is_categorical_dtype(tbl_fused[c]) or
                pd.api.types.is_string_dtype(tbl_fused[c]))
            and tbl_fused[c].nunique(dropna=True) <= 30
        ]
        numeric_candidates = [
            c for c in ["profile_count", "environment_index", "ses_index"]
            if c in tbl_fused.columns
        ]
        color_by_options = ["profile_community"] + sorted(set(cat_low_card + numeric_candidates))
        default_color_by = st.session_state["scatter_color_by"]
        color_by = st.selectbox(
            "Scatter color by",
            options=color_by_options,
            index=color_by_options.index(default_color_by) if default_color_by in color_by_options else 0
        )

st.divider()

# --- Action Button ---
generate_clicked = st.button("Generate Graph", type="primary", use_container_width=False)

# Persist selections to session_state
st.session_state["patient_block_weight"] = float(patient_w)
st.session_state["zip_block_weight"] = float(zip_w)
st.session_state["patient_graph_k"] = int(k)
st.session_state["patient_knn_type"] = knn_type
st.session_state["patient_graph_layout"] = layout
st.session_state["scatter_color_by"] = color_by

# ANN mode
st.session_state["patient_ann_mode"] = (
    "auto" if ann_mode_label.startswith("Auto") else
    "force_ann" if ann_mode_label.startswith("Force ANN") else
    "force_exact"
)
st.session_state["patient_sim_threshold"] = int(sim_thresh)
st.session_state["patient_plot_threshold"] = int(plot_thresh)

# Betweenness mode
st.session_state["patient_btw_mode"] = (
    "auto" if btw_mode_label.startswith("Auto") else
    "approx" if btw_mode_label.startswith("Approx") else
    "exact" if btw_mode_label.startswith("Exact") else
    "skip"
)
st.session_state["patient_btw_threshold"] = int(btw_thresh)
st.session_state["patient_btw_k"] = int(btw_sample_k)

# Similarity metric (only set if experimental encoding)
if encoding_mode == "experimental":
    st.session_state["patient_similarity_metric"] = (
        "cosine" if metric_choice.startswith("Cosine") else "mixed"
    )

# Build figure cache keys after persisting
network_key = (
    int(st.session_state["patient_graph_k"]),
    st.session_state["patient_knn_type"],
    float(st.session_state["patient_block_weight"]),
    float(st.session_state["zip_block_weight"]),
    st.session_state["patient_graph_layout"],
)
# include color_by in the key so recoloring invalidates cache
scatter_key = (
    float(st.session_state["patient_block_weight"]),
    float(st.session_state["zip_block_weight"]),
    st.session_state.get("scatter_color_by", "profile_community"),
)

# ---------------------------------------------------------------------
# Phase 2 + 3 — Weighted feature view + similarity + k-NN graph + metrics
# (Functions now imported from app._logic.psn_graph_builder)
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Cache container for graphs and features
# ---------------------------------------------------------------------
if "patient_graph_cache" not in st.session_state:
    st.session_state["patient_graph_cache"] = {}
graph_cache = st.session_state["patient_graph_cache"]

graph_cache_key = (
    int(st.session_state["patient_graph_k"]),
    st.session_state["patient_knn_type"],
    float(st.session_state["patient_block_weight"]),
    float(st.session_state["zip_block_weight"]),
)

# ---------------------------------------------------------------------
# Generate graph (Phase 2 + 3), render figures, and set as active
# ---------------------------------------------------------------------
if generate_clicked:
    # Phase 2: weighted view
    X_weighted = build_weighted_matrix(
        X_fused=X_fused,
        patient_cols=pat_cols,
        zip_cols=zip_cols,
        patient_w=st.session_state["patient_block_weight"],
        zip_w=st.session_state["zip_block_weight"],
    )

    # Phase 3: neighbors (ANN or exact or mixed) + graph
    similarity_metric = st.session_state.get("patient_similarity_metric", "cosine")

    if similarity_metric == "mixed" and encoding_mode == "experimental":
        # Use custom mixed similarity for experimental encoding
        encoding_meta = st.session_state.get("pf_encoding_metadata", {})
        idxs, sims, sim_full = topk_mixed_similarity(
            X_weighted,
            int(st.session_state["patient_graph_k"]),
            patient_cols=pat_cols,
            zip_cols=zip_cols,
            categorical_mappings=encoding_meta.get("categorical_mappings", {}),
            bitflag_mapping=encoding_meta.get("bitflag_mapping", {}),
            bitflag_column="comorbidities_encoded",
            patient_w=st.session_state["patient_block_weight"],
            zip_w=st.session_state["zip_block_weight"],
        )
    else:
        # Standard cosine similarity (ANN or exact)
        idxs, sims, sim_full = topk_ann_or_exact(
            X_weighted,
            int(st.session_state["patient_graph_k"]),
            ann_mode=st.session_state.get("patient_ann_mode", "auto"),
            sim_threshold=int(st.session_state.get("patient_sim_threshold", SIM_BACKEND_THRESHOLD))
        )
    if sim_full is not None:
        sim_path = PATIENT_CACHE_DIR / (
            f"sim_k{st.session_state['patient_graph_k']}_"
            f"{st.session_state['patient_knn_type']}_"
            f"pw{st.session_state['patient_block_weight']:.2f}_"
            f"zw{st.session_state['zip_block_weight']:.2f}.npy"
        )
        sim_file = np_to_npy_file(sim_full, sim_path)
    else:
        sim_file = None  # ANN path: we do not store dense similarity

    G = build_knn_graph_from_neighbors(
        topk_idx=idxs,
        topk_sim=sims,
        knn_type=st.session_state["patient_knn_type"],
    )

    partition, betweenness, pagerank, degree = compute_graph_metrics(
        G,
        btw_mode=st.session_state.get("patient_btw_mode", "auto"),
        btw_k=int(st.session_state.get("patient_btw_k", 400)),
        btw_threshold=int(st.session_state.get("patient_btw_threshold", BTW_AUTO_THRESHOLD)),
    )

    # Compact features table aligned to node order 0..n-1
    n = idxs.shape[0]
    tbl = tbl_fused.copy()
    if "profile_count" not in tbl.columns:
        tbl["profile_count"] = pd.Series(cnt_fused, index=tbl.index)

    tbl["profile_id"] = pd.Series(idx_fused, index=tbl.index)
    tbl["profile_community"] = [partition.get(i, -1) for i in range(n)]
    tbl["profile_betweenness"] = [betweenness.get(i, 0.0) for i in range(n)]
    tbl["profile_pagerank"] = [pagerank.get(i, 0.0) for i in range(n)]
    tbl["profile_degree"] = [degree.get(i, 0) for i in range(n)]

    # Persist into cache (graph, features, optional similarity file path)
    graph_cache[graph_cache_key] = {"graph": G, "features": tbl, "sim_path": sim_file}

    # ---- Plotting gates ----
    # Always compute scatter (depends only on weights).
    sc_path = PATIENT_FIG_DIR / (
        f"scatter_pw{st.session_state['patient_block_weight']:.2f}_"
        f"zw{st.session_state['zip_block_weight']:.2f}_"
        f"color{st.session_state.get('scatter_color_by', 'profile_community')}.png"
    )
    fig_scatter = plot_profile_scatter_embed(
        X_weighted,
        tbl,
        community_col="profile_community",
        size_col="profile_count",
        color_by=st.session_state.get("scatter_color_by", "profile_community"),
        title="Profiles (PCA 2D, weighted features)",
    )
    fig_cache["scatter"][scatter_key] = {"path": fig_to_png_file(fig_scatter, sc_path)}

    # Full network plot only when n <= plot_thresh
    if n <= int(st.session_state.get("patient_plot_threshold", PLOT_GRAPH_THRESHOLD)):
        net_path = PATIENT_FIG_DIR / (
            f"net_k{st.session_state['patient_graph_k']}_"
            f"{st.session_state['patient_knn_type']}_"
            f"pw{st.session_state['patient_block_weight']:.2f}_"
            f"zw{st.session_state['zip_block_weight']:.2f}_"
            f"layout-{st.session_state['patient_graph_layout']}.png"
        )
        fig_net = plot_networkx_graph(
            G,
            out_df=tbl,
            node_size=20,
            edge_width=0.9,
            edge_alpha=0.5,
            edge_color="gray",
            community_col="profile_community",
            size_col="profile_count",
            title=(
                f"Profile graph: k={st.session_state['patient_graph_k']}, "
                f"{st.session_state['patient_knn_type']}, "
                f"weights (pat={st.session_state['patient_block_weight']:.2f}, "
                f"zip={st.session_state['zip_block_weight']:.2f})"
            ),
            layout=st.session_state["patient_graph_layout"],
            scale_factor=4.0,
        )
        fig_cache["network"][network_key] = {"path": fig_to_png_file(fig_net, net_path)}
    else:
        # Remove any stale network fig for these settings
        if network_key in fig_cache["network"]:
            del fig_cache["network"][network_key]

    # Set as active graph
    st.session_state["active_patient_graph"] = {
        "settings": {
            "k": int(st.session_state["patient_graph_k"]),
            "knn_type": st.session_state["patient_knn_type"],
            "layout": st.session_state["patient_graph_layout"],
            "patient_weight": float(st.session_state["patient_block_weight"]),
            "zip_weight": float(st.session_state["zip_block_weight"]),
            "ann_threshold": int(st.session_state.get("patient_sim_threshold", SIM_BACKEND_THRESHOLD)),
            "plot_threshold": int(st.session_state.get("patient_plot_threshold", PLOT_GRAPH_THRESHOLD)),
            "ann_available": bool(HAS_PYNNDESCENT),
        },
        "graph": graph_cache[graph_cache_key]["graph"],
        "features": graph_cache[graph_cache_key]["features"],
        "network_png": fig_cache["network"].get(network_key, {}).get("path"),
        "scatter_png": fig_cache["scatter"].get(scatter_key, {}).get("path"),
        "similarity_npy": graph_cache[graph_cache_key].get("sim_path"),
    }

    st.success("PSN graph generated and set as active.")

    # --- Current Settings ---
    st.subheader("Current Settings")
    settings_df = pd.DataFrame([
        {"Setting": "Profile Block Weight", "Value": f"{st.session_state['patient_block_weight']:.2f}"},
        {"Setting": "Neighborhood Block Weight", "Value": f"{st.session_state['zip_block_weight']:.2f}"},
        {"Setting": "k", "Value": st.session_state["patient_graph_k"]},
        {"Setting": "k-NN Type", "Value": st.session_state["patient_knn_type"]},
        {"Setting": "Layout", "Value": st.session_state["patient_graph_layout"]},
        {"Setting": "Color By", "Value": st.session_state.get("scatter_color_by", "profile_community")},
    ])
    st.dataframe(settings_df, width='content', hide_index=True)
    
    # Summary
    st.divider()
    st.subheader("Graph Summary")

    # Always work on an undirected projection for structure stats
    G_u = G.to_undirected() if G.is_directed() else G

    # Connectedness & isolates
    is_conn = nx.is_connected(G_u)
    n_iso = sum(1 for _ in nx.isolates(G_u))

    # Largest connected (weak/undirected) component size
    components = list(nx.connected_components(G_u))
    largest_comp_size = max((len(c) for c in components), default=0)

    # Communities & modularity (use communities already on tbl)
    if "profile_community" in tbl.columns:
        num_communities = int(tbl["profile_community"].nunique())
        # partition: node_id -> community_id (assumes node order 0..n-1 aligns to tbl rows)
        partition = {i: int(tbl.iloc[i]["profile_community"]) for i in range(len(tbl))}
        # Louvain modularity of existing partition over undirected view
        try:
            modularity = community_louvain.modularity(partition, G_u, weight="weight")
        except Exception:
            modularity = None
    else:
        num_communities, modularity = 0, None
        partition = {}

    # Average degree (total degree for DiGraph, standard for Graph)
    avg_degree = float(np.mean([d for _, d in G.degree(weight=None)])) if G.number_of_nodes() > 0 else 0.0

    # Compose summary
    summary_df = pd.DataFrame([
        {"Metric": "Nodes", "Value": G.number_of_nodes()},
        {"Metric": "Edges", "Value": G.number_of_edges()},
        {"Metric": "k", "Value": int(st.session_state["patient_graph_k"])},
        {"Metric": "k-NN Type", "Value": st.session_state["patient_knn_type"]},
        {"Metric": "Connected", "Value": "Yes" if is_conn else "No"},
        {"Metric": "Isolated Nodes", "Value": n_iso},
        {"Metric": "Largest Component Size", "Value": largest_comp_size},
        {"Metric": "Number of Communities", "Value": num_communities},
        {"Metric": "Modularity", "Value": f"{modularity:.4f}" if modularity is not None else "N/A"},
        {"Metric": "Average Degree", "Value": f"{avg_degree:.2f}"},
        {
            "Metric": "Similarity Metric",
            "Value": (
                "Mixed (experimental)" if similarity_metric == "mixed"
                else "Cosine (standard)"
            )
        },
        {
            "Metric": "Similarity Backend",
            "Value": (
                "ANN (PyNNDescent)"
                if (
                    X_fused.shape[0] > int(
                        st.session_state.get("patient_sim_threshold", SIM_BACKEND_THRESHOLD)
                    )
                    and HAS_PYNNDESCENT
                    and similarity_metric == "cosine"
                )
                else "Exact" if similarity_metric == "mixed" else "Exact cosine"
            )
        },
        # {"Metric": "Network Plot", "Value": "Rendered" if n <= int(st.session_state.get("patient_plot_threshold", PLOT_GRAPH_THRESHOLD)) else f"Skipped (n>{int(st.session_state.get('patient_plot_threshold', PLOT_GRAPH_THRESHOLD))})"},
        {"Metric": "Betweenness Mode", "Value": st.session_state.get("patient_btw_mode", "auto")},
    ])
    st.dataframe(summary_df, width='content', hide_index=True)



# ---------------------------------------------------------------------
# Results Display (only shown after actions)
# ---------------------------------------------------------------------
if generate_clicked or graph_cache_key in graph_cache:
    st.divider()
    st.header("Results")



    
    # # --- Cache Status ---
    # st.subheader("Cache Status")
    # cache_df = pd.DataFrame([
    #     {"Status": "Graph Cached", "Value": "Yes" if graph_cache_key in graph_cache else "No"},
    #     {"Status": "Network Fig Cached", "Value": "Yes" if network_key in fig_cache["network"] else "No"},
    #     {"Status": "Scatter Fig Cached", "Value": "Yes" if scatter_key in fig_cache["scatter"] else "No"},
    #     {"Status": "Similarity Cached", "Value": "Yes" if (graph_cache_key in graph_cache and bool(graph_cache[graph_cache_key].get("sim_path"))) else "No"},
    #     {"Status": "Similarity Path", "Value": graph_cache.get(graph_cache_key, {}).get("sim_path", "N/A")},
    #     {"Status": "ANN Backend Available", "Value": "Yes" if HAS_PYNNDESCENT else "No"},
    # ])
    # st.dataframe(cache_df, use_container_width=True, hide_index=True)

    # --- Images ---
    if scatter_key in fig_cache["scatter"]:
        st.subheader("PSN PCA Scatter")
        st.image(fig_cache["scatter"][scatter_key]["path"], width='content')
        st.caption(f"Colored by: {st.session_state.get('scatter_color_by', 'profile_community')}")
    else:
        st.info("Scatter image not cached yet. Click 'Generate Graph'.")

    if network_key in fig_cache["network"]:
        st.subheader("PSN Network Graph")
        st.image(fig_cache["network"][network_key]["path"], width='content')
    else:
        plot_thresh = int(st.session_state.get('patient_plot_threshold', PLOT_GRAPH_THRESHOLD))
        st.info(f"Network image not cached or plotting skipped (n>{plot_thresh}).")
    
    # --- Download Data ---
    st.divider()
    st.subheader("PSN Data")
    
    if graph_cache_key in graph_cache:
        tbl_download = graph_cache[graph_cache_key]["features"].copy()
        
        # Show preview of the data (first 10 rows)
        st.caption("Preview (first 10 rows)")
        st.dataframe(tbl_download.head(10), use_container_width=True, hide_index=True)
        
        # Data summary
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Total Rows", len(tbl_download))
        with col_info2:
            st.metric("Total Columns", len(tbl_download.columns))
        with col_info3:
            graph_cols = [c for c in ["profile_community", "profile_betweenness", "profile_pagerank", "profile_degree"] if c in tbl_download.columns]
            st.metric("Graph Metrics", len(graph_cols))
        
        # Download button
        csv_data = tbl_download.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download as CSV",
            data=csv_data,
            file_name=f"psn_data_k{st.session_state['patient_graph_k']}_{st.session_state['patient_knn_type']}.csv",
            mime="text/csv",
            use_container_width=False
        )
        
        st.caption(f"Includes all feature columns plus graph metrics: {', '.join(graph_cols)}")
    else:
        st.info("Generate a graph first to download PSN data.")
