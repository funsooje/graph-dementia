# app/_components/zip_context_utils.py
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import community as community_louvain  

def present(df: pd.DataFrame, cols: list) -> list:
    return [c for c in cols if c in df.columns]

def pca_first_component(df: pd.DataFrame, cols: list):
    cols = present(df, cols)
    if not cols:
        return None, None, []
    X = df[cols].astype(float).values
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=1, random_state=42)
    pc1 = pca.fit_transform(X).ravel()
    return pc1, float(pca.explained_variance_ratio_[0]), cols

def build_knn_graph(features: np.ndarray, k_neighbors: int, knn_type: str):
    """
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

