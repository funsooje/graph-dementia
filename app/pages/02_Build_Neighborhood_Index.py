# app/pages/02_Build_Neighborhood_Index.py
"""
Build and cache a PyNNDescent approximate k-NN index for neighborhood features.

This is a one-time expensive computation (~5-10 min for 71k census tracts).
Once built and named, the index can be reused in the Neighborhood Graph page
to explore different k values instantly (no recomputation needed).
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

from app._logic.loader import ensure_data_loaded

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Build Neighborhood Index", layout="wide")
st.title("Build Neighborhood Index")
st.caption(
    "Build a reusable approximate k-NN index for neighborhood features. "
    "This computation is expensive once, but lets you explore different k values instantly on the next page."
)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
if not ensure_data_loaded():
    st.stop()

active_context_key = st.session_state.get("active_context_key", "default")
active_context_label = (
    st.session_state.get("context_datasets", {})
    .get(active_context_key, {})
    .get("label", active_context_key)
)
st.caption(f"Active context dataset: **{active_context_label}** (`{active_context_key}`)")

zipc = st.session_state.get("zip_df")
if zipc is None:
    st.error("No ZIP/census data found. Please visit the Neighborhood Features page first.")
    st.stop()
zipc = zipc.reset_index(drop=True)

# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------
import json as _json

default_fg = st.session_state.get("default_feature_groups")
if default_fg is None:
    fg_path = Path("data/feature_groups_default.json")
    if fg_path.exists():
        with open(fg_path) as f:
            default_fg = _json.load(f)
    else:
        st.error("Default feature groups not found. Please visit the Neighborhood Features page first.")
        st.stop()

custom_fg = st.session_state.get("feature_groups", {})
if not isinstance(custom_fg, dict):
    custom_fg = {}
feature_groups = {**default_fg, **custom_fg}


def _get_columns(group_data):
    if isinstance(group_data, dict) and "columns" in group_data:
        return group_data["columns"]
    return group_data


def _present(df, cols):
    return [c for c in cols if c in df.columns]


# ---------------------------------------------------------------------------
# Index cache directory
# ---------------------------------------------------------------------------
INDEX_CACHE_DIR = Path("data/cache/nbr_index")
INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def list_cached_indices():
    """Return list of dicts with metadata for each cached index."""
    indices = []
    for meta_file in sorted(INDEX_CACHE_DIR.glob("*.json")):
        try:
            with open(meta_file) as f:
                meta = _json.load(f)
            npz_file = INDEX_CACHE_DIR / f"{meta_file.stem}.npz"
            meta["_exists"] = npz_file.exists()
            meta["_name"] = meta_file.stem
            indices.append(meta)
        except Exception:
            pass
    return indices


def load_index(name: str):
    """Load a cached index by name. Returns (neighbors, distances, metadata)."""
    meta_file = INDEX_CACHE_DIR / f"{name}.json"
    npz_file = INDEX_CACHE_DIR / f"{name}.npz"
    if not meta_file.exists() or not npz_file.exists():
        return None, None, None
    with open(meta_file) as f:
        meta = _json.load(f)
    data = np.load(npz_file)
    return data["neighbors"], data["distances"], meta


def save_index(name: str, neighbors: np.ndarray, distances: np.ndarray, meta: dict):
    """Save index arrays and metadata to disk."""
    npz_file = INDEX_CACHE_DIR / f"{name}.npz"
    meta_file = INDEX_CACHE_DIR / f"{name}.json"
    np.savez_compressed(npz_file, neighbors=neighbors, distances=distances)
    with open(meta_file, "w") as f:
        _json.dump(meta, f, indent=2)


def delete_index(name: str):
    """Delete a cached index."""
    for ext in [".json", ".npz"]:
        p = INDEX_CACHE_DIR / f"{name}{ext}"
        if p.exists():
            p.unlink()


# ---------------------------------------------------------------------------
# Existing indices table
# ---------------------------------------------------------------------------
st.subheader("Cached Indices")

cached = list_cached_indices()

if not cached:
    st.info("No cached indices yet. Build one below.")
else:
    rows = []
    for m in cached:
        rows.append({
            "Name": m["_name"],
            "Dataset": m.get("dataset_label", m.get("dataset_key", "—")),
            "Feature Group": m.get("group_name", "—"),
            "Max k": m.get("k_max", "—"),
            "Nodes": f"{m.get('n_nodes', 0):,}",
            "Features": len(m.get("columns", [])),
            "Built": m.get("built_at", "—"),
            "File OK": "✓" if m["_exists"] else "✗ missing .npz",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Delete controls
    with st.expander("Delete a cached index"):
        names = [m["_name"] for m in cached]
        to_delete = st.selectbox("Select index to delete", names, key="del_select")
        if st.button("Delete", type="secondary", key="btn_delete"):
            delete_index(to_delete)
            # Clear from session state if it was the active one
            if st.session_state.get("nbr_index_name") == to_delete:
                for k in ["nbr_index_name", "nbr_index_neighbors",
                          "nbr_index_distances", "nbr_index_meta"]:
                    st.session_state.pop(k, None)
            st.success(f"Deleted '{to_delete}'.")
            st.rerun()

st.markdown("---")

# ---------------------------------------------------------------------------
# Active index selector (for use in page 03)
# ---------------------------------------------------------------------------
if cached:
    st.subheader("Active Index")
    st.caption("This index will be used by the Neighborhood Graph page.")

    valid = [m["_name"] for m in cached if m["_exists"]]
    current_active = st.session_state.get("nbr_index_name")
    default_idx = valid.index(current_active) if current_active in valid else 0

    active_name = st.selectbox("Select active index", valid, index=default_idx, key="active_select")

    if st.button("Load this index", type="primary", key="btn_load"):
        with st.spinner(f"Loading '{active_name}'..."):
            neighbors, distances, meta = load_index(active_name)
        if neighbors is None:
            st.error("Failed to load index.")
        else:
            st.session_state["nbr_index_name"] = active_name
            st.session_state["nbr_index_neighbors"] = neighbors
            st.session_state["nbr_index_distances"] = distances
            st.session_state["nbr_index_meta"] = meta
            st.success(
                f"Loaded '{active_name}' — {meta.get('n_nodes', 0):,} nodes, "
                f"max k={meta.get('k_max')}, group={meta.get('group_name')}."
            )

    # Show currently loaded status
    if st.session_state.get("nbr_index_name"):
        st.info(f"Currently loaded: **{st.session_state['nbr_index_name']}**")

st.markdown("---")

# ---------------------------------------------------------------------------
# Build new index
# ---------------------------------------------------------------------------
st.subheader("Build New Index")

col1, col2 = st.columns(2)

with col1:
    group_names = list(feature_groups.keys())
    selected_group = st.selectbox("Feature group", group_names, key="build_group")
    k_max = st.number_input(
        "Max k (neighbors to store)",
        min_value=5, max_value=100, value=30, step=5,
        help="Build index with this many neighbors. You can query any k ≤ this value instantly later."
    )

with col2:
    default_name = f"{selected_group}_k{k_max}"
    cache_name = st.text_input(
        "Cache name",
        value=default_name,
        help="A unique name for this index. Use letters, numbers, underscores."
    )
    cache_name = cache_name.strip().replace(" ", "_")

# Show which columns will be used
cols_to_use = _present(zipc, _get_columns(feature_groups[selected_group]))
st.caption(
    f"**{len(cols_to_use)} features** from group '{selected_group}' "
    f"present in data | **{len(zipc):,} nodes**"
)

if not cols_to_use:
    st.error("No valid feature columns found for this group in the data. Choose a different group.")
    st.stop()

# Warn if name already exists
existing_names = [m["_name"] for m in cached]
if cache_name in existing_names:
    st.warning(f"A cache named **'{cache_name}'** already exists. Building will overwrite it.")

build_clicked = st.button("Build Index", type="primary", key="btn_build")

if build_clicked:
    if not cache_name or not cache_name.replace("_", "").isalnum():
        st.error("Cache name must contain only letters, numbers, and underscores.")
        st.stop()

    with st.spinner(
        f"Building PyNNDescent index for '{selected_group}' with k_max={k_max} "
        f"over {len(zipc):,} nodes. This may take several minutes..."
    ):
        try:
            from pynndescent import NNDescent
        except ImportError:
            st.error("PyNNDescent is not installed. Run: pip install pynndescent")
            st.stop()

        # Standardize features
        feats = zipc[cols_to_use].astype(float).values
        feats = StandardScaler().fit_transform(feats)

        # Build index with k_max + 1 (includes self in neighbor list)
        index = NNDescent(
            feats,
            metric="cosine",
            n_neighbors=k_max + 1,
            n_trees=8,
            n_iters=5,
            max_candidates=30,
            pruning_degree_multiplier=1.2,
            diversify_prob=0.8,
            n_jobs=-1,
            verbose=False,
        )

        neighbors, distances = index.neighbor_graph

        # Remove self-loops: first neighbor is usually self (distance ~0)
        # We keep only non-self neighbors up to k_max
        neighbors_clean = []
        distances_clean = []
        for i in range(len(neighbors)):
            row_nbrs = neighbors[i]
            row_dsts = distances[i]
            # Filter out self
            mask = row_nbrs != i
            nbrs_f = row_nbrs[mask][:k_max]
            dsts_f = row_dsts[mask][:k_max]
            # Pad if needed (shouldn't happen with k_max+1 neighbors)
            if len(nbrs_f) < k_max:
                pad = k_max - len(nbrs_f)
                nbrs_f = np.concatenate([nbrs_f, np.full(pad, -1, dtype=np.int32)])
                dsts_f = np.concatenate([dsts_f, np.ones(pad)])
            neighbors_clean.append(nbrs_f)
            distances_clean.append(dsts_f)

        neighbors_arr = np.array(neighbors_clean, dtype=np.int32)
        distances_arr = np.array(distances_clean, dtype=np.float32)

        meta = {
            "cache_name": cache_name,
            "group_name": selected_group,
            "k_max": k_max,
            "n_nodes": len(zipc),
            "columns": cols_to_use,
            "built_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "dataset_key": active_context_key,
            "dataset_label": active_context_label,
        }

        save_index(cache_name, neighbors_arr, distances_arr, meta)

    # Auto-load the just-built index into session state
    st.session_state["nbr_index_name"] = cache_name
    st.session_state["nbr_index_neighbors"] = neighbors_arr
    st.session_state["nbr_index_distances"] = distances_arr
    st.session_state["nbr_index_meta"] = meta

    st.success(
        f"Index **'{cache_name}'** built and loaded! "
        f"{len(zipc):,} nodes × k_max={k_max}. "
        f"Navigate to the Neighborhood Graph page to explore."
    )
    st.rerun()
