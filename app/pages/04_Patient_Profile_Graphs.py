# app/pages/04_Patient_Profile_Graphs.py
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import community as community_louvain

st.set_page_config(page_title="Patient Profile Graphs", layout="wide")
st.title("04 - Patient Profile Graphs")

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
    # Not fatal: allow zero ZIP columns (e.g., aggregate path with none selected)
    st.info("Notice: pf_zip_block_cols not found. ZIP block will default to zero columns.")
    zip_cols = []

if missing:
    st.error(
        "Missing required inputs from Page 03: "
        + ", ".join(missing)
        + ". Go to page 03 and click 'Generate Fused Summary'."
    )
    st.stop()

# Quick preflight summary (no heavy work)
st.subheader("Inputs summary")
n_rows, n_cols = (X_fused.shape if isinstance(X_fused, np.ndarray) else (None, None))
st.write({
    "fused_matrix_shape": (n_rows, n_cols),
    "fused_table_rows": len(tbl_fused),
    "patient_block_cols": len(pat_cols),
    "zip_block_cols": len(zip_cols),
    "index_len": len(idx_fused),
})

# ---------------------------------------------------------------------
# Phase 1 — Sidebar controls (no computation yet)
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    # --- Weights ---
    st.subheader("Block weights")
    default_pw = float(st.session_state.get("patient_block_weight", 0.70))
    default_zw = float(st.session_state.get("zip_block_weight", 0.30))
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        patient_w = st.slider("Patient block", 0.0, 1.0, value=default_pw, step=0.05)
    with col_w2:
        zip_w = st.slider("ZIP block", 0.0, 1.0, value=default_zw, step=0.05)
    st.caption(f"Sum: {patient_w + zip_w:.2f}.")

    # --- Graph options ---
    st.subheader("Graph options")
    default_k = int(st.session_state.get("patient_graph_k", 3))
    k = st.number_input("k (k-NN)", min_value=1, max_value=50, value=default_k, step=1)

    knn_type_map = {"Mutual k-NN (undirected)": "mutual", "Directed k-NN": "directed"}
    default_knn = st.session_state.get("patient_knn_type", "mutual")
    knn_label_default = {v: k for k, v in knn_type_map.items()}.get(default_knn, "Mutual k-NN (undirected)")
    knn_label = st.selectbox("k-NN type", list(knn_type_map.keys()), index=list(knn_type_map.keys()).index(knn_label_default))
    knn_type = knn_type_map[knn_label]

    default_layout = st.session_state.get("patient_graph_layout", "spring")
    layout = st.selectbox(
        "Layout",
        ["spring", "kamada", "circular", "random", "shell"],
        index=["spring", "kamada", "circular", "random", "shell"].index(default_layout),
    )

    # --- Actions ---
    st.subheader("Actions")
    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        recompute_clicked = st.button("Recompute graph")
    with col_b2:
        update_plots_clicked = st.button("Update plots")
    with col_b3:
        set_active_clicked = st.button("Set as patient graph")


# ---------- Phase 2: build weighted feature view ----------
def build_weighted_matrix(X_fused, patient_cols, zip_cols, patient_w: float, zip_w: float):
    """
    Reweights the fused matrix by block:
      - patient block columns * patient_w
      - zip block columns * zip_w
    Assumes X_fused columns are [patient_cols..., zip_cols...] in that order (as in page 03).
    Returns a numpy array X_weighted with same shape.
    """
    n_pat = len(patient_cols) if patient_cols is not None else 0
    n_zip = len(zip_cols) if zip_cols is not None else 0

    Xw = X_fused.astype(float).copy()
    if n_pat > 0:
        Xw[:, :n_pat] *= float(patient_w)
    if n_zip > 0:
        Xw[:, n_pat:n_pat + n_zip] *= float(zip_w)
    return Xw


# ---------- Phase 3: similarity + k-NN graph ----------
def build_knn_graph_from_similarity(sim: np.ndarray, k: int, knn_type: str):
    """
    Build k-NN graph from a similarity matrix (cosine similarity).
    - knn_type == 'mutual' -> undirected mutual k-NN (edge i-j if i in topk(j) and j in topk(i))
    - knn_type == 'directed' -> directed k-NN (edges i -> topk(i))
    """
    n = sim.shape[0]
    # mask diagonal so self isn't chosen
    sim = sim.copy()
    np.fill_diagonal(sim, -np.inf)

    # top-k neighbor indices for each node
    topk_idx = []
    for i in range(n):
        # argpartition gets the k largest, then we order those
        idx = np.argpartition(sim[i], -k)[-k:]
        idx = idx[np.argsort(sim[i, idx])[::-1]]
        topk_idx.append(idx.tolist())

    if knn_type == "directed":
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in topk_idx[i]:
                w = float(sim[i, j])
                if np.isfinite(w):
                    G.add_edge(i, j, weight=w)
        return G

    # mutual (undirected)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        nbrs_i = set(topk_idx[i])
        for j in nbrs_i:
            if i < j and i in set(topk_idx[j]):
                w = float(sim[i, j])
                if np.isfinite(w):
                    G.add_edge(i, j, weight=w)
    return G


def compute_graph_metrics(G: nx.Graph):
    """
    - Communities: Louvain on an undirected view
    - Betweenness: directed if DiGraph, undirected otherwise
    - PageRank: directed if DiGraph, undirected otherwise
    Returns:
      partition: dict[node] -> community_id
      betweenness: dict[node] -> float
      pagerank: dict[node] -> float
      degrees: dict[node] -> degree (in+out for DiGraph)
    """
    # Community on undirected projection
    G_u = G.to_undirected() if G.is_directed() else G
    partition = community_louvain.best_partition(G_u, weight="weight", random_state=42)

    # Centralities
    if G.is_directed():
        btw = nx.betweenness_centrality(G, weight="weight", normalized=True)
        pr = nx.pagerank(G, alpha=0.85, weight="weight")
        deg = dict(G.degree(weight=None))  # unweighted degree count
    else:
        btw = nx.betweenness_centrality(G, weight="weight", normalized=True)
        pr = nx.pagerank(G, alpha=0.85, weight="weight")
        deg = dict(G.degree(weight=None))
    return partition, btw, pr, deg

# ---------- Phase 2 & 3 execution ----------
# Cache container for graphs and features
if "patient_graph_cache" not in st.session_state:
    st.session_state["patient_graph_cache"] = {}

cache_key = (
    int(st.session_state["patient_graph_k"]),
    st.session_state["patient_knn_type"],
    float(st.session_state["patient_block_weight"]),
    float(st.session_state["zip_block_weight"]),
)

if recompute_clicked:
    # Phase 2: weighted feature view
    X_weighted = build_weighted_matrix(
        X_fused=X_fused,
        patient_cols=pat_cols,
        zip_cols=zip_cols,
        patient_w=st.session_state["patient_block_weight"],
        zip_w=st.session_state["zip_block_weight"],
    )

    # Quick sanity preview (cheap)
    st.subheader("Weighted features preview")
    st.write({
        "X_weighted_shape": X_weighted.shape,
        "patient_w": st.session_state["patient_block_weight"],
        "zip_w": st.session_state["zip_block_weight"],
    })

    # Phase 3: cosine similarity + k-NN graph
    sim = cosine_similarity(X_weighted)  # dense; fine at profile scale
    G = build_knn_graph_from_similarity(
        sim=sim,
        k=int(st.session_state["patient_graph_k"]),
        knn_type=st.session_state["patient_knn_type"],
    )

    # Metrics
    partition, betweenness, pagerank, degree = compute_graph_metrics(G)

    # Align to node order 0..n-1
    n = sim.shape[0]
    # Pull a compact feature table for downstream plots/tables
    tbl = tbl_fused.copy()
    # Ensure essential columns exist or fill
    if "profile_count" not in tbl.columns:
        tbl["profile_count"] = pd.Series(cnt_fused, index=tbl.index)

    tbl["profile_id"] = pd.Series(idx_fused, index=tbl.index)
    tbl["profile_community"] = [partition.get(i, -1) for i in range(n)]
    tbl["profile_betweenness"] = [betweenness.get(i, 0.0) for i in range(n)]
    tbl["profile_pagerank"] = [pagerank.get(i, 0.0) for i in range(n)]
    tbl["profile_degree"] = [degree.get(i, 0) for i in range(n)]

    # Persist into cache
    st.session_state["patient_graph_cache"][cache_key] = {
        "graph": G,
        "features": tbl,
    }

    # Lightweight summary
    st.subheader("Graph summary")
    is_conn = nx.is_weakly_connected(G) if G.is_directed() else nx.is_connected(G)
    n_iso = sum(1 for _ in nx.isolates(G.to_undirected() if G.is_directed() else G))
    st.write({
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "k": int(st.session_state["patient_graph_k"]),
        "knn_type": st.session_state["patient_knn_type"],
        "connected": is_conn,
        "isolates": n_iso,
    })

# Show cache status for current settings
st.subheader("Cache status")
cache = st.session_state["patient_graph_cache"]
st.write({
    "cache_key": cache_key,
    "has_cached_graph": cache_key in cache,
})
if cache_key in cache:
    st.write("Cached features head:")
    st.dataframe(cache[cache_key]["features"].head(10), width="content")
    
# Persist selections to session_state (so they’re remembered next visit)
st.session_state["patient_block_weight"] = float(patient_w)
st.session_state["zip_block_weight"] = float(zip_w)
st.session_state["patient_graph_k"] = int(k)
st.session_state["patient_knn_type"] = knn_type
st.session_state["patient_graph_layout"] = layout

# UX hints (no computation yet)
st.subheader("Next steps")
st.markdown(
    "- Click **Recompute graph** to build the profile graph with the current settings.\n"
    "- Click **Update plots** to regenerate figures from a cached graph (faster).\n"
    "- Click **Set as patient graph** to freeze the current run as the active selection for downstream pages."
)

# Show current control state snapshot
st.subheader("Current control state")
st.write({
    "patient_block_weight": st.session_state["patient_block_weight"],
    "zip_block_weight": st.session_state["zip_block_weight"],
    "k": st.session_state["patient_graph_k"],
    "knn_type": st.session_state["patient_knn_type"],
    "layout": st.session_state["patient_graph_layout"],
    "buttons": {
        "recompute_clicked": recompute_clicked,
        "update_plots_clicked": update_plots_clicked,
        "set_active_clicked": set_active_clicked,
    }
})