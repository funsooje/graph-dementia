# app/_logic/psn_graph_builder.py
"""
Shared PSN (Patient Similarity Network) graph building utilities.

This module provides functions for building patient similarity networks from
encoded feature matrices, including:
- Block-weighted feature matrices
- Exact and approximate similarity computation (ANN via PyNNDescent)
- k-NN graph construction
- Graph metrics (communities, betweenness, pagerank, degree)
"""

import numpy as np
import networkx as nx
import community as community_louvain
from sklearn.metrics.pairwise import cosine_similarity
import time
import logging
import sys
import os

# Try importing PyNNDescent for approximate nearest neighbors
try:
    from pynndescent import NNDescent
    HAS_PYNNDESCENT = True
except ImportError:
    HAS_PYNNDESCENT = False


# ---------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------
def _make_logger():
    """Create logger for PSN graph building."""
    logger = logging.getLogger("psn_graph_builder")
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
    """Format elapsed time since t0."""
    return f"{time.perf_counter() - t0:.2f}s"


# ---------------------------------------------------------------------
# Block weighting
# ---------------------------------------------------------------------
def build_weighted_matrix(X_fused, patient_cols, zip_cols, patient_w: float, zip_w: float):
    """
    Reweights the fused matrix by block:
      - patient block columns * patient_w
      - zip block columns * zip_w

    Assumes X_fused columns are [patient_cols..., zip_cols...] in that order.

    Args:
        X_fused: numpy array of shape (n, d) with patient and neighborhood features
        patient_cols: list of patient block column names
        zip_cols: list of neighborhood block column names
        patient_w: weight for patient block (0.0 to 1.0)
        zip_w: weight for neighborhood block (0.0 to 1.0)

    Returns:
        numpy array X_weighted with same shape as X_fused
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
# Similarity computation (exact or ANN)
# ---------------------------------------------------------------------
def topk_exact_from_matrix(X: np.ndarray, k: int):
    """
    Exact cosine similarity top-k neighbors for each row.

    Args:
        X: numpy array of shape (n, d)
        k: number of neighbors to find

    Returns:
        Tuple of (indices, sims, sim_matrix):
        - indices: (n, k) array of neighbor indices
        - sims: (n, k) array of similarity scores
        - sim_matrix: (n, n) full similarity matrix
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


def topk_ann_or_exact(X: np.ndarray, k: int, ann_mode: str, sim_threshold: int):
    """
    Use ANN (PyNNDescent) or exact cosine similarity for top-k neighbors.

    Args:
        X: numpy array of shape (n, d)
        k: number of neighbors to find
        ann_mode: "auto" (use threshold), "force_ann", or "force_exact"
        sim_threshold: threshold for auto mode (use ANN when n > threshold)

    Returns:
        Tuple of (indices, sims, sim_matrix_or_None):
        - indices: (n, k) array of neighbor indices
        - sims: (n, k) array of similarity scores
        - sim_matrix_or_None: (n, n) full similarity matrix if exact, None if ANN
    """
    n, d = X.shape
    LOGGER.info(f"[SIM] Backend: n={n}, d={d}, k={k}, mode={ann_mode}, threshold={sim_threshold}")

    # Decide whether to use ANN
    use_ann = False
    if ann_mode == "force_ann" and HAS_PYNNDESCENT:
        use_ann = True
    elif ann_mode == "force_exact":
        use_ann = False
    elif ann_mode == "auto" and n > sim_threshold and HAS_PYNNDESCENT:
        use_ann = True

    if use_ann:
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


def topk_mixed_similarity(
    X: np.ndarray,
    k: int,
    patient_cols: list,
    zip_cols: list,
    categorical_mappings: dict,
    bitflag_mapping: dict,
    bitflag_column: str = "comorbidities_encoded",
    patient_w: float = 0.5,
    zip_w: float = 0.5,
):
    """
    Compute top-k neighbors using mixed similarity metrics for experimental encoding.

    Two-level weighting system:
    1. Internal patient block: Each feature type (categorical, bitflag, numeric) is
       weighted proportionally to the number of columns it represents
    2. Patient vs zip: User-controlled weights (patient_w, zip_w) for final combination

    For patient block:
    - Exact-match distance for integer-encoded categoricals
    - Hamming distance for bitflag-encoded comorbidities
    - Cosine similarity for any remaining numeric features

    For zip block:
    - Standard cosine similarity (already numeric + standardized)

    Args:
        X: numpy array of shape (n, d) with patient and zip blocks
        k: number of neighbors to find
        patient_cols: list of patient block column names
        zip_cols: list of zip block column names
        categorical_mappings: dict {col: {category: int}} from integer encoding
        bitflag_mapping: dict {bit_position: col} from bitflag encoding
        bitflag_column: name of bitflag column (default: "comorbidities_encoded")
        patient_w: weight for patient block in final combination (default: 0.5)
        zip_w: weight for zip block in final combination (default: 0.5)

    Returns:
        Tuple of (indices, sims, sim_matrix)
    """
    t0 = time.perf_counter()
    n, d = X.shape
    n_pat = len(patient_cols) if patient_cols else 0
    n_zip = len(zip_cols) if zip_cols else 0

    LOGGER.info(
        f"[MIXED-SIM] n={n}, d={d}, k={k}, patient_cols={n_pat}, zip_cols={n_zip}"
    )

    # Split into patient and zip blocks
    X_patient = X[:, :n_pat] if n_pat > 0 else np.zeros((n, 0))
    X_zip = X[:, n_pat:n_pat + n_zip] if n_zip > 0 else np.zeros((n, 0))

    # ---- Patient block: mixed similarity ----
    # Identify column indices for different encoding types
    int_encoded_cols = []  # Integer-encoded categoricals (not bitflag)
    bitflag_col_idx = None  # Bitflag column index
    numeric_cols = []  # Any remaining numeric columns

    for i, col_name in enumerate(patient_cols):
        if col_name == bitflag_column:
            bitflag_col_idx = i
        elif col_name.endswith("_encoded"):
            # Integer-encoded categorical
            int_encoded_cols.append(i)
        else:
            # Numeric (e.g., comorbidity binary columns if not bitflagged)
            numeric_cols.append(i)

    # Calculate column-proportional weights for patient block
    total_patient_cols = len(int_encoded_cols) + (1 if bitflag_col_idx is not None else 0) + len(numeric_cols)

    if total_patient_cols > 0:
        cat_weight = len(int_encoded_cols) / total_patient_cols
        bitflag_weight = (1 / total_patient_cols) if bitflag_col_idx is not None else 0.0
        numeric_weight = len(numeric_cols) / total_patient_cols
    else:
        cat_weight = 0.0
        bitflag_weight = 0.0
        numeric_weight = 0.0

    LOGGER.info(
        f"[MIXED-SIM] Patient block breakdown: "
        f"int_encoded={len(int_encoded_cols)} (w={cat_weight:.3f}), "
        f"bitflag={'yes' if bitflag_col_idx is not None else 'no'} "
        f"(w={bitflag_weight:.3f}), "
        f"numeric={len(numeric_cols)} (w={numeric_weight:.3f})"
    )

    # Initialize patient similarity as zeros (will accumulate weighted similarities)
    patient_sim = np.zeros((n, n), dtype=float)
    total_weight = 0.0

    # 1. Exact-match similarity for integer-encoded categoricals
    if int_encoded_cols:
        LOGGER.info(
            f"[MIXED-SIM] Computing exact-match for "
            f"{len(int_encoded_cols)} categorical cols"
        )
        match_count = np.zeros((n, n), dtype=float)
        for col_idx in int_encoded_cols:
            col_vals = X_patient[:, col_idx].astype(int)
            # Pairwise exact match: 1 if equal, 0 otherwise
            matches = (col_vals[:, None] == col_vals[None, :]).astype(float)
            match_count += matches
        # Average match rate across categorical columns
        cat_sim = match_count / len(int_encoded_cols)
        patient_sim += cat_sim * cat_weight
        total_weight += cat_weight

    # 2. Hamming similarity for bitflag-encoded comorbidities
    if bitflag_col_idx is not None:
        LOGGER.info("[MIXED-SIM] Computing Hamming similarity for bitflag column")
        bitflag_vals = X_patient[:, bitflag_col_idx].astype(int)
        num_bits = len(bitflag_mapping)

        # Compute pairwise Hamming distance via XOR + bit count
        xor_matrix = bitflag_vals[:, None] ^ bitflag_vals[None, :]
        hamming_dist = np.array([
            [bin(val).count('1') for val in row]
            for row in xor_matrix
        ], dtype=float)

        # Convert distance to similarity: 1 - (dist / num_bits)
        hamming_sim = 1.0 - (hamming_dist / num_bits) if num_bits > 0 else np.ones((n, n))
        patient_sim += hamming_sim * bitflag_weight
        total_weight += bitflag_weight

    # 3. Cosine similarity for remaining numeric columns
    if numeric_cols:
        LOGGER.info(f"[MIXED-SIM] Computing cosine for {len(numeric_cols)} numeric cols")
        X_numeric = X_patient[:, numeric_cols]
        numeric_sim = cosine_similarity(X_numeric)
        patient_sim += numeric_sim * numeric_weight
        total_weight += numeric_weight

    # Normalize patient similarity
    if total_weight > 0:
        patient_sim /= total_weight

    # ---- Zip block: standard cosine similarity ----
    zip_sim = np.zeros((n, n), dtype=float)
    if n_zip > 0:
        LOGGER.info(f"[MIXED-SIM] Computing cosine for {n_zip} zip cols")
        zip_sim = cosine_similarity(X_zip)

    # ---- Combine patient and zip similarities using user-defined weights ----
    combined_sim = patient_sim * patient_w + zip_sim * zip_w
    LOGGER.info(
        f"[MIXED-SIM] Combined similarity computed in {_elapsed(t0)} "
        f"(patient_w={patient_w:.3f}, zip_w={zip_w:.3f})"
    )

    # ---- Extract top-k neighbors ----
    np.fill_diagonal(combined_sim, -np.inf)
    idxs = np.empty((n, k), dtype=int)
    sims = np.empty((n, k), dtype=float)

    t1 = time.perf_counter()
    for i in range(n):
        idx = np.argpartition(combined_sim[i], -k)[-k:]
        idx = idx[np.argsort(combined_sim[i, idx])[::-1]]
        idxs[i] = idx
        sims[i] = combined_sim[i, idx]
    LOGGER.info(f"[MIXED-SIM] top-k selection done in {_elapsed(t1)} (total {_elapsed(t0)})")

    return idxs, sims, combined_sim


# ---------------------------------------------------------------------
# k-NN graph construction
# ---------------------------------------------------------------------
def build_knn_graph_from_neighbors(topk_idx: np.ndarray, topk_sim: np.ndarray, knn_type: str):
    """
    Build k-NN graph from neighbor lists.

    Args:
        topk_idx: (n, k) array of neighbor indices
        topk_sim: (n, k) array of similarity scores
        knn_type: 'directed' or 'mutual'
            - 'directed': i -> topk(i) with weight = sim
            - 'mutual': undirected edges only if i in topk(j) and j in topk(i)

    Returns:
        NetworkX Graph (undirected) or DiGraph (directed)
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


# ---------------------------------------------------------------------
# Graph metrics
# ---------------------------------------------------------------------
def compute_graph_metrics(G: nx.Graph, btw_mode: str = "auto", btw_k: int = 400, btw_threshold: int = 5000):
    """
    Compute graph metrics:
    - Communities: Louvain on undirected view
    - Betweenness: 'skip' | 'approx' (sampling) | 'exact' | 'auto' (approx if large)
    - PageRank: directed if DiGraph, undirected otherwise
    - Degree: unweighted degree count

    Args:
        G: NetworkX Graph or DiGraph
        btw_mode: betweenness mode ('auto', 'approx', 'exact', 'skip')
        btw_k: number of samples for approximate betweenness
        btw_threshold: threshold for auto mode (use approximate when n >= threshold)

    Returns:
        Tuple of (partition, betweenness, pagerank, degree):
        - partition: dict {node_id: community_id}
        - betweenness: dict {node_id: betweenness_centrality}
        - pagerank: dict {node_id: pagerank_score}
        - degree: dict {node_id: degree}
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
        mode = "approx" if n >= btw_threshold else "exact"

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
