"""Utility helpers for ZIP context analysis: PCA, graph building, and result creation.

This module centralizes common functions used by the ZIP context pages.
"""
# app/_components/zip_context_utils.py
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import community as community_louvain
import matplotlib.pyplot as plt

def present(df: pd.DataFrame, cols: list) -> list:
    """Return list of columns that are present in the DataFrame."""
    return [c for c in cols if c in df.columns]

def get_group_columns(group_data):
    """Extract columns from a feature group (handles dict or list)."""
    if isinstance(group_data, dict) and "columns" in group_data:
        return group_data["columns"]
    return group_data

def pca_first_component(df: pd.DataFrame, cols: list):
    """Compute first PCA component for given columns."""
    cols = present(df, cols)
    if not cols:
        return None, None, []
    X = df[cols].astype(float).values
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=1, random_state=42)
    pc1 = pca.fit_transform(X).ravel()
    return pc1, float(pca.explained_variance_ratio_[0]), cols

def compute_adaptive_pca_indices(
    df: pd.DataFrame,
    selected_features: list,
    default_groups: dict
) -> dict:
    """
    Compute PCA indices adaptively based on presence of env/ses features.
    Only computes indices if corresponding features are present.
    
    Args:
        df: DataFrame containing the features
        selected_features: List of selected feature columns
        default_groups: Dictionary containing default feature groups (must have 'env' and 'ses')
    
    Returns:
        dict with keys:
            - environment_index: array or None
            - ses_index: array or None
            - env_var: float or None (explained variance)
            - ses_var: float or None (explained variance)
            - env_cols: list of columns used
            - ses_cols: list of columns used
    """
    # Get all possible env/ses columns from default groups
    def get_cols_from_default(group_name):
        group = default_groups.get(group_name, [])
        if isinstance(group, dict) and "columns" in group:
            return group["columns"]
        return group

    # Find which env/ses columns are in the selected features
    env_cols_all = present(df, get_cols_from_default("env")) + present(df, get_cols_from_default("env_raw"))
    ses_cols_all = present(df, get_cols_from_default("ses")) + present(df, get_cols_from_default("ses_raw"))
    selected_set = set(selected_features)
    
    env_cols = [c for c in env_cols_all if c in selected_set]
    ses_cols = [c for c in ses_cols_all if c in selected_set]
    
    # Compute indices only if corresponding columns exist
    environment_index, env_var, env_used = pca_first_component(df, env_cols)
    ses_index, ses_var, ses_used = pca_first_component(df, ses_cols)
    
    return {
        "environment_index": environment_index,
        "ses_index": ses_index,
        "env_var": env_var,
        "ses_var": ses_var,
        "env_cols": env_used,
        "ses_cols": ses_used
    }

def build_knn_graph(features: np.ndarray, k_neighbors: int, knn_type: str):
    """
    Build k-NN graph from feature matrix.
    
    Args:
        features: n x d feature matrix
        k_neighbors: number of neighbors per node
        knn_type: "mutual" -> undirected mutual k-NN
                 "directed" -> directed k-NN (i -> top-k neighbors of i)
    """
    sim = cosine_similarity(features)
    np.fill_diagonal(sim, -np.inf)
    n = sim.shape[0]

    # top-k neighbor sets per node
    topk = []
    for i in range(n):
        idx = np.argpartition(sim[i], -k_neighbors)[-k_neighbors:]
        idx = idx[np.argsort(sim[i, idx])[::-1]]
        topk.append(idx.tolist())

    if knn_type == "directed":
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in topk[i]:
                w = float(sim[i, j])
                if np.isfinite(w):
                    G.add_edge(i, j, weight=w)
        return G

    # mutual (intersection) undirected
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        nbrs_i = set(topk[i])
        for j in nbrs_i:
            if i < j and i in set(topk[j]):
                w = float(sim[i, j])
                if np.isfinite(w):
                    G.add_edge(i, j, weight=w)
    return G

def compute_graph_metrics(G):
    """
    Compute graph metrics:
    - Communities (Louvain) on an undirected view
    - Betweenness: directed if DiGraph, undirected otherwise
    - PageRank: directed if DiGraph, undirected otherwise
    """
    if G.is_directed():
        G_u = G.to_undirected(reciprocal=False)
        partition = community_louvain.best_partition(G_u, weight="weight", random_state=42)
        btw = nx.betweenness_centrality(G, weight="weight", normalized=True)
        pr = nx.pagerank(G, alpha=0.85, weight="weight")
    else:
        partition = community_louvain.best_partition(G, weight="weight", random_state=42)
        btw = nx.betweenness_centrality(G, weight="weight", normalized=True)
        pr = nx.pagerank(G, alpha=0.85, weight="weight")
    if G.number_of_edges() == 0:
        empty = {n: -1 for n in G.nodes()}
        zeros = {n: 0.0 for n in G.nodes()}
        return empty, zeros, zeros

    return partition, btw, pr

def save_figure_to_cache(fig, path: Path, dpi: int = 110) -> str:
    """Save a matplotlib figure to disk with proper directory creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return str(path)

def create_cache_key(group_name: str, k: int, knn_type: str, layout: str = None):
    """Create consistent cache keys for graph and figure caches."""
    if layout:
        return (group_name, k, knn_type, layout)
    return (group_name, k, knn_type)

def create_results_dataframe(
    zipc: pd.DataFrame,
    partition: dict,
    betweenness: dict,
    pagerank: dict,
    environment_index: np.ndarray = None,
    ses_index: np.ndarray = None,
    env_var: float = None,
    ses_var: float = None,
    degree: list = None,
    isolated: list = None,
    # scalar graph-summary fields (single value per group)
    nodes: int = None,
    edges: int = None,
    num_communities: int = None,
    isolated_nodes: int = None,
    is_connected: bool = None,
    group_suffix: str = ""
) -> pd.DataFrame:
    """
    Create a standardized results DataFrame with optional group suffix.
    
    Args:
        zipc: Original DataFrame with ZIPCODE column
        partition, betweenness, pagerank: Graph metrics
        environment_index, ses_index: PCA indices (optional)
        env_var, ses_var: Explained variance ratios (optional)
        group_suffix: Suffix to add to column names (e.g., "_env")
    """
    n = len(zipc)
    suffix = f"_{group_suffix}" if group_suffix else ""
    
    results = pd.DataFrame(
        {
            "ZIPCODE": zipc["ZIPCODE"].astype(str).values,
            f"environment_index{suffix}": (
                environment_index if environment_index is not None else [np.nan] * n
            ),
            f"ses_index{suffix}": (
                ses_index if ses_index is not None else [np.nan] * n
            ),
            f"zip_community{suffix}": [partition.get(i, -1) for i in range(n)],
            f"zip_betweenness{suffix}": [
                betweenness.get(i, np.nan) for i in range(n)
            ],
            f"zip_pagerank{suffix}": [pagerank.get(i, np.nan) for i in range(n)],
            f"degree{suffix}": (degree if degree is not None else [np.nan] * n),
            f"isolated{suffix}": (isolated if isolated is not None else [False] * n),
            # Graph summary scalars (same value repeated for every row)
            f"nodes{suffix}": (nodes if nodes is not None else np.nan),
            f"edges{suffix}": (edges if edges is not None else np.nan),
            f"num_communities{suffix}": (num_communities if num_communities is not None else np.nan),
            f"isolated_nodes{suffix}": (isolated_nodes if isolated_nodes is not None else np.nan),
            f"is_connected{suffix}": (is_connected if is_connected is not None else np.nan),
        }
    )
    
    # Add explained variance if available
    if env_var is not None:
        results[f"environment_index_var{suffix}"] = env_var
    if ses_var is not None:
        results[f"ses_index_var{suffix}"] = ses_var
    
    return results

def process_zip_group(
    zipc: pd.DataFrame,
    group_name: str,
    feature_groups: dict,
    k: int,
    knn_type: str,
    default_groups: dict = None
) -> pd.DataFrame:
    """
    Process a single ZIP code feature group and return results DataFrame.
    
    Args:
        zipc: DataFrame with ZIP code data
        group_name: Name of the feature group to process
        feature_groups: Dictionary of all feature groups
        k: Number of neighbors for kNN
        knn_type: Type of kNN graph ("mutual" or "directed")
        default_groups: Default feature groups dict for env/ses detection
    
    Returns:
        DataFrame with processed results
    """
    # Get features for this group
    selected_features = present(zipc, get_group_columns(feature_groups[group_name]))
    if not selected_features:
        return None
        
    # Prepare feature matrix
    feats = zipc[selected_features].astype(float).values
    feats = StandardScaler().fit_transform(feats)
    
    # Build graph and compute metrics
    G = build_knn_graph(feats, k_neighbors=k, knn_type=knn_type)
    partition, betweenness, pagerank = compute_graph_metrics(G)
    # Compute degree and isolated status per node
    # For DiGraph, G.degree() returns the sum of in+out degrees; for Graph it's degree
    degree_dict = dict(G.degree())
    degree_arr = [int(degree_dict.get(i, 0)) for i in range(len(zipc))]
    isolated_arr = [degree_dict.get(i, 0) == 0 for i in range(len(zipc))]
    # Graph summary scalars
    nodes = G.number_of_nodes()
    edges = G.number_of_edges()
    # number of detected communities (exclude unknown -1 if present)
    comm_values = set(partition.values()) if partition else set()
    num_communities = len([c for c in comm_values if c >= 0]) if partition else 0
    isolated_nodes = sum(1 for _ in nx.isolates(G))
    # connectivity: for directed graphs check weak connectivity, else connected
    is_connected = nx.is_weakly_connected(G) if G.is_directed() else nx.is_connected(G)
    
    # Compute PCA indices if default_groups provided
    indices = None
    if default_groups is not None:
        indices = compute_adaptive_pca_indices(zipc, selected_features, default_groups)
    
    # Create results DataFrame
    results = create_results_dataframe(
        zipc=zipc,
        partition=partition,
        betweenness=betweenness,
        pagerank=pagerank,
        environment_index=indices["environment_index"] if indices else None,
        ses_index=indices["ses_index"] if indices else None,
        env_var=indices["env_var"] if indices else None,
        ses_var=indices["ses_var"] if indices else None,
        degree=degree_arr,
        isolated=isolated_arr,
        nodes=nodes,
        edges=edges,
        num_communities=num_communities,
        isolated_nodes=isolated_nodes,
        is_connected=is_connected,
        group_suffix=group_name
    )
    
    return results

