# app/pages/04_Patient_Profile_Graphs.py
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import community as community_louvain


import matplotlib.pyplot as plt
from pathlib import Path

# --- DEBUG: logging setup ---
import os, time, logging, sys

from app._components.plots import (
    plot_networkx_graph,          # returns a Matplotlib fig
    plot_profile_scatter_embed,   # returns a Matplotlib fig (PCA 2D)
)

from pynndescent import NNDescent
HAS_PYNNDESCENT = True

def _make_logger():
    logger = logging.getLogger("patient_graphs")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    level_name = os.getenv("GD_LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level_name, logging.INFO))
    return logger

LOGGER = _make_logger()

def _elapsed(t0: float) -> str:
    return f"{time.perf_counter() - t0:.2f}s"



# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(page_title="Patient Profile Graphs", layout="wide")
st.title("04 - Patient Profile Graphs")

# ---------------------------------------------------------------------
# Constants (ANN + plotting gates)
# ---------------------------------------------------------------------
SIM_BACKEND_THRESHOLD = 5000   # switch to ANN when n > this
PLOT_GRAPH_THRESHOLD  = 3000   # skip full network plot when n > this

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
    st.caption(f"Sum: {patient_w + zip_w:.2f} (weights are scalars applied to blocks; they need not sum to 1).")

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

    # --- Performance thresholds ---
    st.subheader("Performance thresholds")
    sim_thresh = st.number_input(
        "ANN switch threshold (use ANN when n >)",
        min_value=1000, max_value=100000, step=500,
        value=int(st.session_state.get("patient_sim_threshold", SIM_BACKEND_THRESHOLD))
    )
    plot_thresh = st.number_input(
        "Plotting threshold (skip network when n >)",
        min_value=200, max_value=100000, step=200,
        value=int(st.session_state.get("patient_plot_threshold", PLOT_GRAPH_THRESHOLD))
    )

    # --- Metric options ---
    st.subheader("Metric options")
    btw_mode_label = st.selectbox(
        "Betweenness mode",
        ["Auto (approx if large)", "Approximate", "Exact", "Skip"],
        index=0,
        help="Exact betweenness on large graphs is slow. 'Auto' uses approximate when n is large."
    )
    btw_sample_k = st.number_input(
        "Approx. betweenness samples (k)",
        min_value=50, max_value=5000, step=50,
        value=int(st.session_state.get("patient_btw_k", 400)),
        help="Number of random source nodes for approximate betweenness."
    )

    # --- Scatter coloring ---
    st.subheader("Scatter coloring")

    # Heuristic candidates
    cat_low_card = [
        c for c in tbl_fused.columns
        if (tbl_fused[c].dtype == "object" or pd.api.types.is_categorical_dtype(tbl_fused[c]) or pd.api.types.is_string_dtype(tbl_fused[c]))
        and tbl_fused[c].nunique(dropna=True) <= 30
    ]

    numeric_candidates = [c for c in ["profile_count", "environment_index", "ses_index"] if c in tbl_fused.columns]

    color_by_options = ["profile_community"] + sorted(set(cat_low_card + numeric_candidates))
    default_color_by = st.session_state.get("scatter_color_by", "profile_community")
    color_by = st.selectbox("Color PCA by", options=color_by_options,
                            index=color_by_options.index(default_color_by) if default_color_by in color_by_options else 0)

    # persist
    st.session_state["scatter_color_by"] = color_by

    # --- Actions ---
    st.subheader("Actions")
    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        recompute_clicked = st.button("Recompute graph")
    with col_b2:
        update_plots_clicked = st.button("Update plots")
    with col_b3:
        set_active_clicked = st.button("Set as patient graph")

    # Terminal-only verbose logging toggle
    verbose = st.checkbox("Verbose server logging (terminal)", value=False, help="Toggles DEBUG logs in terminal only.")
    LOGGER.setLevel(logging.DEBUG if verbose else logging.INFO)

# Persist selections to session_state
st.session_state["patient_block_weight"] = float(patient_w)
st.session_state["zip_block_weight"] = float(zip_w)
st.session_state["patient_graph_k"] = int(k)
st.session_state["patient_knn_type"] = knn_type
st.session_state["patient_graph_layout"] = layout
st.session_state["patient_sim_threshold"] = int(sim_thresh)
st.session_state["patient_plot_threshold"] = int(plot_thresh)

st.session_state["patient_btw_mode"] = (
    "auto" if btw_mode_label.startswith("Auto") else
    "approx" if btw_mode_label.startswith("Approx") else
    "exact" if btw_mode_label.startswith("Exact") else
    "skip"
)
st.session_state["patient_btw_k"] = int(btw_sample_k)

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
# Phase 2 — Build weighted feature view
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Phase 3 — Similarity (ANN or exact) + k-NN graph
# ---------------------------------------------------------------------
def topk_exact_from_matrix(X: np.ndarray, k: int):
    """
    Exact cosine similarity top-k neighbors for each row.
    Returns (indices, sims, sim_matrix) with shape (n, k) and (n, n).
    """
    t0 = time.perf_counter()
    n, d = X.shape
    LOGGER.info(f"[SIM] Exact cosine path: n={n}, d={d}, k={k}")
    sim = cosine_similarity(X)
    LOGGER.info(f"[SIM] cosine_similarity done in {_elapsed(t0)}; sim.shape={sim.shape}")

    np.fill_diagonal(sim, -np.inf)
    idxs = np.empty((n, k), dtype=int)
    sims = np.empty((n, k), dtype=float)

    t1 = time.perf_counter()
    for i in range(n):
        idx = np.argpartition(sim[i], -k)[-k:]
        idx = idx[np.argsort(sim[i, idx])[::-1]]
        idxs[i] = idx
        sims[i] = sim[i, idx]
    LOGGER.info(f"[SIM] top-k selection done in {_elapsed(t1)} (total {_elapsed(t0)})")
    return idxs, sims, sim

def topk_ann_or_exact(X: np.ndarray, k: int, sim_threshold: int):
    """
    Use ANN (PyNNDescent) when n > sim_threshold and backend is available.
    Otherwise, use exact cosine.
    Returns (indices, sims, sim_matrix_or_None).
    """
    n, d = X.shape
    LOGGER.info(f"[SIM] Backend chooser: n={n}, d={d}, k={k}, threshold={sim_threshold}, HAS_PYNNDESCENT={HAS_PYNNDESCENT}")
    if n > sim_threshold and HAS_PYNNDESCENT:
        t0 = time.perf_counter()
        LOGGER.info("[SIM] Using ANN (PyNNDescent, metric='cosine')")
        index = NNDescent(X, metric="cosine", n_neighbors=k+1, random_state=42)
        LOGGER.info(f"[SIM] NNDescent build in {_elapsed(t0)}")

        t1 = time.perf_counter()
        nbrs_idx, nbrs_dist = index.query(X, k=k+1)
        LOGGER.info(f"[SIM] NNDescent query in {_elapsed(t1)} (total {_elapsed(t0)})")

        # Convert cosine distance to similarity = 1 - dist; drop self if present
        idxs = np.zeros((n, k), dtype=int)
        sims = np.zeros((n, k), dtype=float)
        for i in range(n):
            row_idx = nbrs_idx[i].tolist()
            row_dst = nbrs_dist[i].tolist()
            cleaned = [(j, d_) for j, d_ in zip(row_idx, row_dst) if j != i]
            cleaned = cleaned[:k] if len(cleaned) >= k else cleaned
            while len(cleaned) < k:
                cleaned.append((i, 1.0))  # worst similarity if padding
            idxs[i] = [j for j, _ in cleaned]
            sims[i] = [1.0 - d_ for _, d_ in cleaned]
        return idxs, sims, None

    LOGGER.info("[SIM] Using EXACT cosine path (n ≤ threshold or ANN unavailable)")
    return topk_exact_from_matrix(X, k)

def build_knn_graph_from_neighbors(topk_idx: np.ndarray, topk_sim: np.ndarray, knn_type: str):
    """
    Build k-NN graph from neighbor lists.
    - 'directed': i -> topk(i) with weight = sim
    - 'mutual': undirected edges only if i in topk(j) and j in topk(i)
    """
    t0 = time.perf_counter()
    n, k = topk_idx.shape
    LOGGER.info(f"[GRAPH] Build from neighbors: n={n}, k={k}, type={knn_type}")

    if knn_type == "directed":
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for r in range(k):
                j = int(topk_idx[i, r])
                w = float(topk_sim[i, r])
                if np.isfinite(w):
                    G.add_edge(i, j, weight=w)
        LOGGER.info(f"[GRAPH] Directed graph edges={G.number_of_edges()} in {_elapsed(t0)}")
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
    LOGGER.info(f"[GRAPH] Mutual graph edges={G.number_of_edges()} in {_elapsed(t0)}")
    return G

def compute_graph_metrics(G: nx.Graph, btw_mode: str = "auto", btw_k: int = 400):
    """
    - Communities: Louvain on undirected view
    - Betweenness: 'skip' | 'approx' (sampling with nx.betweenness_centrality k=...) | 'exact' | 'auto' (approx if large)
    - PageRank: directed if DiGraph, undirected otherwise
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()

    # Community on undirected projection
    t0 = time.perf_counter()
    G_u = G.to_undirected() if G.is_directed() else G
    LOGGER.info(f"[METRICS] Louvain on {'undirected projection' if G.is_directed() else 'undirected graph'}; nodes={G_u.number_of_nodes()}, edges={G_u.number_of_edges()}")
    partition = community_louvain.best_partition(G_u, weight="weight", random_state=42)
    LOGGER.info(f"[METRICS] Louvain done in {_elapsed(t0)}")

    # ---- Betweenness ----
    # decide mode
    mode = btw_mode
    if btw_mode == "auto":
        mode = "approx" if n >= 5000 else "exact"

    if mode == "skip":
        LOGGER.info("[METRICS] Betweenness skipped")
        btw = {i: 0.0 for i in G.nodes()}
    elif mode == "approx":
        LOGGER.info(f"[METRICS] Approx betweenness via sampling (k={int(btw_k)}) starting...")
        t1 = time.perf_counter()
        try:
            # NetworkX supports sampling by passing k (number of sources) to betweenness_centrality.
            btw = nx.betweenness_centrality(G_u, k=int(btw_k), normalized=True, weight="weight", seed=42)
        except TypeError:
            # Fallback for older NetworkX without 'seed' argument
            btw = nx.betweenness_centrality(G_u, k=int(btw_k), normalized=True, weight="weight")
        LOGGER.info(f"[METRICS] Approx betweenness done in {_elapsed(t1)}")
    else:  # exact
        LOGGER.info("[METRICS] Exact betweenness starting (may be slow)...")
        t1 = time.perf_counter()
        btw = nx.betweenness_centrality(G, weight="weight", normalized=True)
        LOGGER.info(f"[METRICS] Exact betweenness done in {_elapsed(t1)}")

    # ---- PageRank ----
    t2 = time.perf_counter()
    pr = nx.pagerank(G, alpha=0.85, weight="weight")
    LOGGER.info(f"[METRICS] PageRank in {_elapsed(t2)}")

    deg = dict(G.degree(weight=None))
    LOGGER.info(f"[METRICS] Degree computed; total nodes={len(deg)}")
    return partition, btw, pr, deg

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
# Recompute graph (Phase 2 + 3) and render figures
# ---------------------------------------------------------------------
if recompute_clicked:
    LOGGER.info("[RUN] Recompute clicked")
    # Phase 2: weighted view
    X_weighted = build_weighted_matrix(
        X_fused=X_fused,
        patient_cols=pat_cols,
        zip_cols=zip_cols,
        patient_w=st.session_state["patient_block_weight"],
        zip_w=st.session_state["zip_block_weight"],
    )

    LOGGER.info(f"[RUN] Weighted matrix built: shape={X_weighted.shape}, pw={st.session_state['patient_block_weight']:.2f}, zw={st.session_state['zip_block_weight']:.2f}")

    st.subheader("Weighted features preview")
    st.write({
        "X_weighted_shape": X_weighted.shape,
        "patient_w": st.session_state["patient_block_weight"],
        "zip_w": st.session_state["zip_block_weight"],
    })

    # Phase 3: neighbors (ANN or exact) + graph
    idxs, sims, sim_full = topk_ann_or_exact(
        X_weighted,
        int(st.session_state["patient_graph_k"]),
        int(st.session_state.get("patient_sim_threshold", SIM_BACKEND_THRESHOLD))
    )
    LOGGER.info(f"[RUN] Neighbor lists ready; sim_full={'present' if sim_full is not None else 'None (ANN path)'}")
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
    LOGGER.info(f"[RUN] Graph built: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}, directed={G.is_directed()}")

    partition, betweenness, pagerank, degree = compute_graph_metrics(
        G,
        btw_mode=st.session_state.get("patient_btw_mode", "auto"),
        btw_k=int(st.session_state.get("patient_btw_k", 400)),
    )
    LOGGER.info("[RUN] Metrics computed (partition, betweenness, pagerank, degree)")

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
        f"zw{st.session_state['zip_block_weight']:.2f}.png"
    )
    LOGGER.info("[PLOT] Rendering scatter (PCA 2D) figure")
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
        LOGGER.info("[PLOT] Rendering network figure")
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

    st.success("Graph recomputed and plots updated.")

    # Summary
    st.subheader("Graph summary")

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

    # Inter-community “overlap” (dominant neighboring community ties) for top communities
    # Define community size by patient count if available, else by profile count
    if "profile_community" in tbl.columns:
        if "profile_count" in tbl.columns:
            comm_sizes = tbl.groupby("profile_community")["profile_count"].sum()
        else:
            comm_sizes = tbl["profile_community"].value_counts()
        # top communities (by size)
        top_comms = comm_sizes.sort_values(ascending=False).head(5).index.tolist()

        # Build inter-community edge counts
        inter_counts = {}
        incident_totals = {}  # total inter-community edges touching each community
        for u, v in G_u.edges():
            cu, cv = partition.get(u, None), partition.get(v, None)
            if cu is None or cv is None or cu == cv:
                continue
            a, b = (cu, cv) if cu <= cv else (cv, cu)
            inter_counts[(a, b)] = inter_counts.get((a, b), 0) + 1
            incident_totals[cu] = incident_totals.get(cu, 0) + 1
            incident_totals[cv] = incident_totals.get(cv, 0) + 1

        # For each top community, find the neighbor community with most ties
        overlap_summary = {}
        for c in top_comms:
            # gather neighbors of c
            neighbors = []
            for (a, b), cnt in inter_counts.items():
                if a == c:
                    neighbors.append((b, cnt))
                elif b == c:
                    neighbors.append((a, cnt))
            if neighbors:
                neighbors.sort(key=lambda x: x[1], reverse=True)
                nb, cnt = neighbors[0]
                denom = max(incident_totals.get(c, 0), 1)
                overlap_summary[int(c)] = {
                    "dominant_neighbor": int(nb),
                    "edges_to_neighbor": int(cnt),
                    "share_of_inter_edges": float(cnt / denom),
                }
            else:
                overlap_summary[int(c)] = {
                    "dominant_neighbor": None,
                    "edges_to_neighbor": 0,
                    "share_of_inter_edges": 0.0,
                }
    else:
        overlap_summary = {}
        comm_sizes = pd.Series(dtype=float)

    # Compose summary dict
    summary_dict = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "k": int(st.session_state["patient_graph_k"]),
        "knn_type": st.session_state["patient_knn_type"],
        "connected": is_conn,
        "isolates": n_iso,
        "largest_component_size": largest_comp_size,
        "num_communities": num_communities,
        "modularity": modularity,
        "average_degree": avg_degree,
        "backend": (
            "ANN (PyNNDescent)"
            if (X_fused.shape[0] > int(st.session_state.get("patient_sim_threshold", SIM_BACKEND_THRESHOLD)) and HAS_PYNNDESCENT)
            else "Exact cosine"
        ),
        "plotting": (
            "Full network"
            if X_fused.shape[0] <= int(st.session_state.get("patient_plot_threshold", PLOT_GRAPH_THRESHOLD))
            else f"Skipped (n>{int(st.session_state.get('patient_plot_threshold', PLOT_GRAPH_THRESHOLD))})"
        ),
        "betweenness_mode": st.session_state.get("patient_btw_mode", "auto"),
        "betweenness_k": int(st.session_state.get("patient_btw_k", 400)),
        # lightweight view of overlap for top communities
        "overlap_top_comms": overlap_summary,
    }

    st.write(summary_dict)
# ---------------------------------------------------------------------
# Update plots from cached graph (no recompute)
# ---------------------------------------------------------------------
if update_plots_clicked:
    LOGGER.info("[RUN] Update plots clicked")
    if graph_cache_key not in graph_cache:
        st.info("No cached graph for these settings. Click 'Recompute graph' first.")
    else:
        G = graph_cache[graph_cache_key]["graph"]
        tbl = graph_cache[graph_cache_key]["features"]
        n = G.number_of_nodes()

        # SCATTER (weights-only). If missing, build weighted view quickly and cache.
        if scatter_key not in fig_cache["scatter"]:
            X_weighted = build_weighted_matrix(
                X_fused=X_fused,
                patient_cols=pat_cols,
                zip_cols=zip_cols,
                patient_w=st.session_state["patient_block_weight"],
                zip_w=st.session_state["zip_block_weight"],
            )
            sc_path = PATIENT_FIG_DIR / (
                f"scatter_pw{st.session_state['patient_block_weight']:.2f}_"
                f"zw{st.session_state['zip_block_weight']:.2f}.png"
            )
            LOGGER.info("[PLOT] Regenerating scatter from cached graph")
            fig_scatter = plot_profile_scatter_embed(
                X_weighted,
                tbl,
                community_col="profile_community",
                size_col="profile_count",
                color_by=st.session_state.get("scatter_color_by", "profile_community"),
                title="Profiles (PCA 2D, weighted features)",
            )
            fig_cache["scatter"][scatter_key] = {"path": fig_to_png_file(fig_scatter, sc_path)}

        # NETWORK (layout-dependent) refresh only if allowed at this size
        if n <= int(st.session_state.get("patient_plot_threshold", PLOT_GRAPH_THRESHOLD)):
            net_path = PATIENT_FIG_DIR / (
                f"net_k{st.session_state['patient_graph_k']}_"
                f"{st.session_state['patient_knn_type']}_"
                f"pw{st.session_state['patient_block_weight']:.2f}_"
                f"zw{st.session_state['zip_block_weight']:.2f}_"
                f"layout-{st.session_state['patient_graph_layout']}.png"
            )
            LOGGER.info("[PLOT] Regenerating network from cached graph")
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
            if network_key in fig_cache["network"]:
                del fig_cache["network"][network_key]
            st.info(f"Graph plotting skipped (n>{int(st.session_state.get('patient_plot_threshold', PLOT_GRAPH_THRESHOLD))}).")

        st.success("Plots updated from cached graph.")

# ---------------------------------------------------------------------
# Set active selection for downstream pages
# ---------------------------------------------------------------------
if set_active_clicked:
    LOGGER.info("[RUN] Set active clicked")
    if graph_cache_key not in graph_cache:
        st.info("No cached graph to set active. Click 'Recompute graph' first.")
    else:
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
        LOGGER.info("[RUN] Active patient graph stored in session_state['active_patient_graph']")
        st.success("Current patient graph set as active.")

# ---------------------------------------------------------------------
# Outputs: cache status and images
# ---------------------------------------------------------------------
st.subheader("Cache status")
st.write({
    "graph_cached": graph_cache_key in graph_cache,
    "network_fig_cached": network_key in fig_cache["network"],
    "scatter_fig_cached": scatter_key in fig_cache["scatter"],
    "similarity_cached": (graph_cache_key in graph_cache and bool(graph_cache[graph_cache_key].get("sim_path"))),
    "similarity_path": (graph_cache.get(graph_cache_key, {}).get("sim_path")),
    "ann_backend_available": HAS_PYNNDESCENT,
})

# Images
if scatter_key in fig_cache["scatter"]:
    st.subheader("Profiles PCA scatter (cached image)")
    st.image(fig_cache["scatter"][scatter_key]["path"])
    st.caption(f"PCA scatter colored by: {st.session_state.get('scatter_color_by', 'profile_community')}")
else:
    st.info("Scatter image not cached yet. Click 'Recompute graph' or 'Update plots'.")

if network_key in fig_cache["network"]:
    st.subheader("Network graph (cached image)")
    st.image(fig_cache["network"][network_key]["path"])
else:
    st.info(f"Network image not cached for this layout/settings or plotting was skipped (n>{int(st.session_state.get('patient_plot_threshold', PLOT_GRAPH_THRESHOLD))}).")

# ---------------------------------------------------------------------
# Current control state (for transparency)
# ---------------------------------------------------------------------
st.subheader("Current control state")
st.write({
    "patient_block_weight": st.session_state["patient_block_weight"],
    "zip_block_weight": st.session_state["zip_block_weight"],
    "k": st.session_state["patient_graph_k"],
    "knn_type": st.session_state["patient_knn_type"],
    "layout": st.session_state["patient_graph_layout"],
    "buttons": {
        "recompute_clicked": 'clicked' if 'recompute_clicked' in locals() and recompute_clicked else 'idle',
        "update_plots_clicked": 'clicked' if 'update_plots_clicked' in locals() and update_plots_clicked else 'idle',
        "set_active_clicked": 'clicked' if 'set_active_clicked' in locals() and set_active_clicked else 'idle',
    }
})