# app/_logic/graph_cache.py
"""
Shared graph computation cache for neighborhood graph analysis.
Caches graph computations based on (feature_group, k, knn_type, resolution).
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
import networkx as nx
import streamlit as st


# Cache directory
GRAPH_CACHE_DIR = Path("data/cache/neighborhood_graph")


def get_cache_key(feature_group: str, k: int, knn_type: str, resolution: float) -> tuple:
    """
    Generate a cache key for graph computation.

    Args:
        feature_group: Name of the feature group
        k: k-NN parameter
        knn_type: Type of k-NN graph ('mutual' or 'directed')
        resolution: Community detection resolution

    Returns:
        Tuple cache key (feature_group, k, knn_type, resolution)
    """
    return (feature_group, int(k), knn_type, float(resolution))


def _make_json_safe(obj):
    """Recursively convert numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_graph_to_cache(
    cache_key: tuple,
    graph: nx.Graph,
    features_df: pd.DataFrame,
    meta: dict,
    zip_indices: dict = None
):
    """
    Save a computed graph and its features to cache.

    Args:
        cache_key: Tuple (feature_group, k, knn_type, resolution)
        graph: NetworkX graph object
        features_df: DataFrame with computed features (ZIPCODE, indices, metrics, etc.)
        meta: Metadata dictionary (feature_group, k, knn_type, resolution, layout, etc.)
        zip_indices: Optional dictionary with environment/SES indices and variance
    """
    feature_group, k, knn_type, resolution = cache_key

    # Create subdirectory for this cache key
    cache_subdir = GRAPH_CACHE_DIR / f"{feature_group}_k{k}_{knn_type}_res{resolution:.2f}"
    cache_subdir.mkdir(parents=True, exist_ok=True)

    # Save features DataFrame
    features_df.to_csv(cache_subdir / "features.csv", index=False)

    # Save graph edges
    edges = [
        {"source": int(u), "target": int(v), "weight": float(d.get("weight", 1.0))}
        for u, v, d in graph.edges(data=True)
    ]
    pd.DataFrame(edges).to_csv(cache_subdir / "edges.csv", index=False)

    # Save metadata and zip indices
    combined = {
        "meta": meta,
        "zip_indices": zip_indices or {},
        "graph_info": {
            "n_nodes": graph.number_of_nodes(),
            "is_directed": graph.is_directed(),
        },
        "cache_key": {
            "feature_group": feature_group,
            "k": k,
            "knn_type": knn_type,
            "resolution": resolution,
        }
    }

    with open(cache_subdir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(_make_json_safe(combined), f, indent=2)


def load_graph_from_cache(cache_key: tuple):
    """
    Load a cached graph and its features.

    Args:
        cache_key: Tuple (feature_group, k, knn_type, resolution)

    Returns:
        Tuple (graph, features_df, meta, zip_indices) or None if not found
    """
    feature_group, k, knn_type, resolution = cache_key

    cache_subdir = GRAPH_CACHE_DIR / f"{feature_group}_k{k}_{knn_type}_res{resolution:.2f}"

    feat_path = cache_subdir / "features.csv"
    edge_path = cache_subdir / "edges.csv"
    meta_path = cache_subdir / "meta.json"

    if not all(p.exists() for p in (feat_path, edge_path, meta_path)):
        return None

    try:
        # Load features
        features_df = pd.read_csv(feat_path)

        # Load edges and reconstruct graph
        edge_df = pd.read_csv(edge_path)

        with open(meta_path, "r", encoding="utf-8") as f:
            combined = json.load(f)

        meta = combined["meta"]
        graph_info = combined["graph_info"]

        # Reconstruct graph
        if graph_info["is_directed"]:
            graph = nx.DiGraph()
        else:
            graph = nx.Graph()

        graph.add_nodes_from(range(graph_info["n_nodes"]))

        for _, row in edge_df.iterrows():
            graph.add_edge(
                int(row["source"]),
                int(row["target"]),
                weight=float(row["weight"])
            )

        # Restore zip_indices
        zip_indices = {}
        for key, val in combined.get("zip_indices", {}).items():
            zip_indices[key] = np.array(val) if isinstance(val, list) else val

        return graph, features_df, meta, zip_indices

    except Exception as e:
        print(f"Error loading cache for {cache_key}: {e}")
        return None


def initialize_session_cache():
    """
    Initialize session state graph cache if not present.
    Load all cached graphs from disk into session state.
    """
    if "zip_graph_cache" not in st.session_state:
        st.session_state["zip_graph_cache"] = {}

    # Load all cached graphs from disk
    if GRAPH_CACHE_DIR.exists():
        for cache_subdir in GRAPH_CACHE_DIR.iterdir():
            if not cache_subdir.is_dir():
                continue

            meta_path = cache_subdir / "meta.json"
            if not meta_path.exists():
                continue

            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    combined = json.load(f)

                cache_info = combined.get("cache_key", {})
                cache_key = (
                    cache_info.get("feature_group"),
                    cache_info.get("k"),
                    cache_info.get("knn_type"),
                    cache_info.get("resolution", 1.0)
                )

                # Only load metadata into session state, not full graph
                # Full graph will be loaded on demand
                if cache_key not in st.session_state["zip_graph_cache"]:
                    result = load_graph_from_cache(cache_key)
                    if result is not None:
                        graph, features_df, meta, zip_indices = result
                        st.session_state["zip_graph_cache"][cache_key] = {
                            "graph": graph,
                            "features": features_df
                        }
            except Exception:
                continue


def get_cached_graph(cache_key: tuple):
    """
    Get a graph from cache (session state or disk).

    Args:
        cache_key: Tuple (feature_group, k, knn_type, resolution)

    Returns:
        Tuple (graph, features_df) or None if not found
    """
    # Check session state first
    if "zip_graph_cache" in st.session_state:
        if cache_key in st.session_state["zip_graph_cache"]:
            cached = st.session_state["zip_graph_cache"][cache_key]
            return cached["graph"], cached["features"]

    # Try loading from disk
    result = load_graph_from_cache(cache_key)
    if result is not None:
        graph, features_df, meta, zip_indices = result

        # Store in session state for faster access next time
        if "zip_graph_cache" not in st.session_state:
            st.session_state["zip_graph_cache"] = {}

        st.session_state["zip_graph_cache"][cache_key] = {
            "graph": graph,
            "features": features_df
        }

        return graph, features_df

    return None


def clear_all_cache():
    """Clear all cached graphs from session state and disk."""
    import shutil

    # Clear session state
    if "zip_graph_cache" in st.session_state:
        st.session_state["zip_graph_cache"] = {}

    # Clear disk cache
    if GRAPH_CACHE_DIR.exists():
        shutil.rmtree(GRAPH_CACHE_DIR)
        GRAPH_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def clear_cache_entry(cache_key: tuple):
    """
    Clear a specific cache entry from session state and disk.

    Args:
        cache_key: Tuple (feature_group, k, knn_type, resolution)
    """
    import shutil

    # Clear from session state
    if "zip_graph_cache" in st.session_state:
        if cache_key in st.session_state["zip_graph_cache"]:
            del st.session_state["zip_graph_cache"][cache_key]

    # Clear from disk
    feature_group, k, knn_type, resolution = cache_key
    cache_subdir = GRAPH_CACHE_DIR / f"{feature_group}_k{k}_{knn_type}_res{resolution:.2f}"

    if cache_subdir.exists():
        shutil.rmtree(cache_subdir)


def get_cache_stats() -> dict:
    """
    Get statistics about cached graphs.

    Returns:
        Dictionary with cache statistics
    """
    stats = {
        "total_entries": 0,
        "session_entries": 0,
        "disk_entries": 0,
        "cache_keys": []
    }

    # Session state entries
    if "zip_graph_cache" in st.session_state:
        stats["session_entries"] = len(st.session_state["zip_graph_cache"])
        stats["cache_keys"].extend(st.session_state["zip_graph_cache"].keys())

    # Disk entries
    if GRAPH_CACHE_DIR.exists():
        disk_entries = [d for d in GRAPH_CACHE_DIR.iterdir() if d.is_dir()]
        stats["disk_entries"] = len(disk_entries)

    stats["total_entries"] = max(stats["session_entries"], stats["disk_entries"])

    return stats


def reconstruct_results_from_cache(
    graph: nx.Graph,
    features_df: pd.DataFrame,
    group_name: str
) -> pd.DataFrame:
    """
    Reconstruct the results DataFrame from cached graph and features.
    This avoids expensive recomputation of PCA, graph construction, and community detection.

    Args:
        graph: Cached NetworkX graph
        features_df: Cached features DataFrame (columns without group suffix)
        group_name: Name of the group (for adding suffix to columns)

    Returns:
        DataFrame with group-suffixed columns matching process_zip_group output
    """
    # Recompute cheap graph statistics
    nodes = graph.number_of_nodes()
    edges = graph.number_of_edges()

    # Get community values from cached features
    if "zip_community" in features_df.columns:
        comm_values = set(features_df["zip_community"].values)
        num_communities = len([c for c in comm_values if c >= 0])
    else:
        num_communities = 0

    # Count isolated nodes
    isolated_nodes = sum(1 for _ in nx.isolates(graph))

    # Count connected components
    n_components = (
        nx.number_weakly_connected_components(graph) if graph.is_directed()
        else nx.number_connected_components(graph)
    )

    # Check connectivity
    is_connected = nx.is_weakly_connected(graph) if graph.is_directed() else nx.is_connected(graph)

    # Get modularity from cached features if available
    modularity = features_df["modularity"].iloc[0] if "modularity" in features_df.columns else None

    # Create results DataFrame with group suffix
    results = pd.DataFrame()
    # Ensure ZIPCODE is string type for consistency with process_zip_group
    results["ZIPCODE"] = features_df["ZIPCODE"].astype(str)

    # Add all metrics with group suffix
    for col in features_df.columns:
        if col == "ZIPCODE":
            continue
        # Add group suffix to column name
        results[f"{col}_{group_name}"] = features_df[col]

    # Add graph summary scalars (same value for all rows)
    results[f"nodes_{group_name}"] = nodes
    results[f"edges_{group_name}"] = edges
    results[f"num_communities_{group_name}"] = num_communities
    results[f"isolated_nodes_{group_name}"] = isolated_nodes
    results[f"n_components_{group_name}"] = n_components
    results[f"is_connected_{group_name}"] = is_connected

    # Ensure modularity is added with suffix
    if modularity is not None and f"modularity_{group_name}" not in results.columns:
        results[f"modularity_{group_name}"] = modularity

    return results
