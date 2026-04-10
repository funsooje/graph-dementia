# app/pages/09_PSN_Feature_Group_Comparison.py

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import json
from pathlib import Path
import community as community_louvain

# Import PSN graph building utilities
from app._logic.psn_graph_builder import (
    build_weighted_matrix,
    topk_ann_or_exact,
    topk_mixed_similarity,
    build_knn_graph_from_neighbors,
)

st.set_page_config(page_title="PSN Feature Group Comparison", layout="wide")
st.title("PSN Feature Group Comparison")

st.caption("Compare multiple PSN feature groups side-by-side")

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
SIM_BACKEND_THRESHOLD = 5000

# ---------------------------------------------------------------------
# Load saved feature groups
# ---------------------------------------------------------------------
PSN_GROUPS_FILE = Path("data/config/psn_feature_groups.json")


def load_psn_groups() -> dict:
    """Load PSN feature groups from file."""
    if PSN_GROUPS_FILE.exists():
        try:
            with open(PSN_GROUPS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------
def _as_categorical_str(s: pd.Series) -> pd.Series:
    """Convert series to string categorical."""
    return s.astype(str).replace("nan", "")


def _make_profile_id(row, cols, prefix="prof"):
    """Create profile ID from row values."""
    vals = [str(row[c]) if pd.notna(row[c]) else "" for c in cols]
    return f"{prefix}_{'_'.join(vals)}"


def _safe_merge(left, right, on, how="left"):
    """Merge with common column type casting."""
    if on in left.columns:
        left = left.copy()
        left[on] = left[on].astype(str)
    if on in right.columns:
        right = right.copy()
        right[on] = right[on].astype(str)
    return left.merge(right, on=on, how=how)


def _one_hot(df, cols):
    """One-hot encode categorical columns."""
    if not cols:
        return pd.DataFrame(index=df.index)
    enc = pd.get_dummies(df[cols], prefix=cols, drop_first=False, dtype=float)
    return enc


def _standardize(df):
    """Standardize numeric dataframe (mean=0, std=1)."""
    if df.empty:
        return df
    return (df - df.mean()) / (df.std() + 1e-9)


def integer_encode_categoricals(df, cols):
    """Integer-encode categorical columns."""
    if not cols:
        return pd.DataFrame(index=df.index), {}

    mappings = {}
    encoded_dfs = []

    for col in cols:
        if col not in df.columns:
            continue
        categories = df[col].astype(str).unique()
        mapping = {cat: idx for idx, cat in enumerate(sorted(categories))}
        mappings[col] = mapping

        encoded_col = df[col].astype(str).map(mapping).fillna(0).astype(int)
        encoded_dfs.append(pd.DataFrame({f"{col}_encoded": encoded_col}, index=df.index))

    result = pd.concat(encoded_dfs, axis=1) if encoded_dfs else pd.DataFrame(index=df.index)
    return result, mappings


def bitflag_encode_multibinary(df, cols):
    """Bitflag-encode multiple binary columns into single integer."""
    if not cols:
        return pd.DataFrame({"comorbidities_encoded": 0}, index=df.index), {}

    bitflag_map = {i: col for i, col in enumerate(cols)}
    encoded = pd.Series(0, index=df.index)

    for i, col in enumerate(cols):
        if col in df.columns:
            bit_val = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(0, 1).astype(int)
            encoded += bit_val * (2 ** i)

    return pd.DataFrame({"comorbidities_encoded": encoded}, index=df.index), bitflag_map


def build_psn_matrix_from_config(pat_df, zip_feats_df, config):
    """
    Build PSN feature matrix from a group config.

    Args:
        pat_df: patients dataframe
        zip_feats_df: zip features dataframe
        config: group config dict with keys:
            - selected_cols
            - neighborhood_features
            - experimental_encoding

    Returns:
        Tuple of (X_fused, patient_cols, zip_cols, encoding_metadata)
    """
    selected_cols = config.get("selected_cols", [])
    zip_cfg = config.get("neighborhood_features", {})
    experimental_encoding = config.get("experimental_encoding", False)

    use_degree = zip_cfg.get("zip_degree", False)
    use_pr = zip_cfg.get("zip_pagerank", False)
    use_btw = zip_cfg.get("zip_betweenness", False)
    use_zip_comm = zip_cfg.get("zip_community", False)

    # Prepare work dataframe
    work_cols = list(dict.fromkeys(selected_cols + (["ZIPCODE"] if "ZIPCODE" in pat_df.columns else [])))
    dfw = pat_df[work_cols].copy()

    for c in selected_cols:
        if c in dfw.columns:
            dfw[c] = _as_categorical_str(dfw[c])
    if "ZIPCODE" in dfw.columns:
        dfw["ZIPCODE"] = _as_categorical_str(dfw["ZIPCODE"]).str.replace(" ", "", regex=False)

    # Build base profiles
    base_grp = dfw.groupby(selected_cols, dropna=False).size().reset_index(name="profile_count")
    base_grp["profile_id"] = base_grp.apply(lambda r: _make_profile_id(r, selected_cols, "prof"), axis=1)

    # Compute ZIP counts for weighted averaging
    zip_counts = None
    if "ZIPCODE" not in selected_cols and "ZIPCODE" in dfw.columns:
        zip_counts = (
            dfw.groupby(selected_cols + ["ZIPCODE"], dropna=False)
               .size().reset_index(name="n")
        )

    # Join with ZIP features
    if zip_counts is None:
        # No ZIP aggregation possible
        fused_tbl = base_grp.copy()
    else:
        zc2 = _safe_merge(zip_counts, zip_feats_df, on="ZIPCODE", how="left")

        # Determine grouping columns
        grouping_cols = selected_cols.copy()
        if use_zip_comm and "zip_community" in zc2.columns:
            grouping_cols = grouping_cols + ["zip_community"]

        # Numeric ZIP features to weighted-average
        zip_num_cols = []
        if use_degree and "zip_degree" in zc2.columns:
            zip_num_cols.append("zip_degree")
        if use_pr and "zip_pagerank" in zc2.columns:
            zip_num_cols.append("zip_pagerank")
        if use_btw and "zip_betweenness" in zc2.columns:
            zip_num_cols.append("zip_betweenness")

        if zip_num_cols or use_zip_comm:
            zc2["n"] = zc2["n"].astype(float)
            grp = zc2.groupby(grouping_cols, dropna=False)

            if zip_num_cols:
                num_wavg = (grp.apply(lambda g: pd.Series(
                    {col: np.average(g[col].fillna(0.0), weights=g["n"]) for col in zip_num_cols}
                )).reset_index())
            else:
                num_wavg = pd.DataFrame({col: zc2[col] for col in grouping_cols}).drop_duplicates()
        else:
            num_wavg = base_grp[selected_cols].copy()

        fused_tbl = base_grp.merge(num_wavg, on=selected_cols, how="left")

    # Encode patient block
    RISK = {"Hearingloss", "BrainInjury", "Hypertension", "Alcohol", "Obesity", "Diabetes"}
    bin_cols = [c for c in selected_cols if c in RISK and c in fused_tbl.columns]
    cat_cols = [c for c in selected_cols if c not in bin_cols]

    encoding_metadata = {
        "mode": "experimental" if experimental_encoding else "standard",
        "categorical_mappings": {},
        "bitflag_mapping": {},
    }

    if experimental_encoding:
        X_cat, cat_mappings = integer_encode_categoricals(fused_tbl, cat_cols)
        encoding_metadata["categorical_mappings"] = cat_mappings

        X_bin, bitflag_map = bitflag_encode_multibinary(fused_tbl, bin_cols)
        encoding_metadata["bitflag_mapping"] = bitflag_map

        patient_block = pd.concat([X_cat, X_bin], axis=1)
    else:
        X_cat = _one_hot(fused_tbl, cat_cols)
        X_bin = pd.DataFrame(index=fused_tbl.index)
        for c in bin_cols:
            X_bin[c] = (
                pd.to_numeric(fused_tbl[c], errors="coerce")
                .fillna(0.0).clip(0, 1).astype(float)
            )
        patient_block = pd.concat([X_cat, X_bin], axis=1)

    # Encode ZIP block
    zip_num_cols2 = []
    if use_degree and "zip_degree" in fused_tbl.columns:
        zip_num_cols2.append("zip_degree")
    if use_pr and "zip_pagerank" in fused_tbl.columns:
        zip_num_cols2.append("zip_pagerank")
    if use_btw and "zip_betweenness" in fused_tbl.columns:
        zip_num_cols2.append("zip_betweenness")

    zip_num_df = (
        fused_tbl[zip_num_cols2].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        if zip_num_cols2 else pd.DataFrame(index=fused_tbl.index)
    )
    zip_num_std = _standardize(zip_num_df) if not zip_num_df.empty else zip_num_df

    # Handle zip_community encoding
    zip_comm_df = pd.DataFrame(index=fused_tbl.index)
    if use_zip_comm and "zip_community" in fused_tbl.columns:
        if experimental_encoding:
            zip_comm_df, comm_mapping = integer_encode_categoricals(fused_tbl, ["zip_community"])
            encoding_metadata["categorical_mappings"].update(comm_mapping)
        else:
            zip_comm_df = _one_hot(fused_tbl, ["zip_community"])

    zip_block = pd.concat([zip_num_std, zip_comm_df], axis=1)

    # Final fused matrix
    X_fused = pd.concat([patient_block, zip_block], axis=1).fillna(0.0)

    return X_fused.values, list(patient_block.columns), list(zip_block.columns), encoding_metadata


def compute_psn_metrics(G):
    """Compute graph metrics for comparison."""
    n = G.number_of_nodes()
    G_u = G.to_undirected() if G.is_directed() else G

    # Community detection
    partition = community_louvain.best_partition(G_u, weight="weight", random_state=42)

    # Metrics
    edges = G.number_of_edges()
    num_communities = len(set(partition.values()))
    isolated_nodes = sum(1 for _ in nx.isolates(G_u))
    n_components = nx.number_connected_components(G_u)

    # Modularity
    try:
        modularity = community_louvain.modularity(partition, G_u, weight="weight")
    except Exception:
        modularity = None

    # Average degree
    avg_degree = float(np.mean([d for _, d in G.degree(weight=None)])) if n > 0 else 0.0

    return {
        "nodes": n,
        "edges": edges,
        "communities": num_communities,
        "components": n_components,
        "isolated_nodes": isolated_nodes,
        "modularity": round(modularity, 4) if modularity is not None else None,
        "avg_degree": round(avg_degree, 2),
    }


# Load from file into session state
if "psn_feature_groups" not in st.session_state:
    st.session_state["psn_feature_groups"] = load_psn_groups()

psn_groups = st.session_state.get("psn_feature_groups", {})

if not psn_groups:
    st.warning("No PSN feature groups defined. Go to page 08 to create feature groups first.")
    st.stop()

# ---------------------------------------------------------------------
# Group Selection
# ---------------------------------------------------------------------
st.subheader("1. Select Feature Groups")

selected_groups = st.multiselect(
    "Choose groups to compare",
    options=list(psn_groups.keys()),
    default=None,
    help="Select 2 or more feature groups to compare"
)

if not selected_groups:
    st.info("Select feature groups above to begin comparison.")
    st.stop()

# ---------------------------------------------------------------------
# Config Comparison Table
# ---------------------------------------------------------------------
st.divider()
st.subheader("2. Configuration Comparison")

# Build comparison table
rows = []
for group_name in selected_groups:
    cfg = psn_groups[group_name]

    # Extract config values
    profile_cols = ", ".join(cfg.get("selected_cols", []))
    zip_cfg = cfg.get("neighborhood_features", {})
    active_zip = [k.replace("zip_", "") for k, v in zip_cfg.items() if v]
    neighborhood = ", ".join(active_zip) if active_zip else "-"

    weight_balance = cfg.get("weight_balance", 0.3)
    weight_str = f"P:{1-weight_balance:.0%} / N:{weight_balance:.0%}"

    # Graph construction settings
    experimental = cfg.get("experimental_encoding", False)
    encoding = "Experimental" if experimental else "Standard"
    similarity = cfg.get("similarity_metric", "cosine").capitalize()
    k = cfg.get("k", 5)
    knn_type = cfg.get("knn_type", "mutual").capitalize()
    ann_mode = cfg.get("ann_mode", "auto").capitalize()

    rows.append({
        "Group": group_name,
        "Profile Columns": profile_cols[:50] + "..." if len(profile_cols) > 50 else profile_cols,
        "Neighborhood": neighborhood,
        "Weight": weight_str,
        "Encoding": encoding,
        "Similarity": similarity,
        "k": k,
        "Type": knn_type,
        "ANN": ann_mode,
    })

config_df = pd.DataFrame(rows)
st.dataframe(config_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------
# Data checks
# ---------------------------------------------------------------------
from app._logic.loader import ensure_data_loaded

# Load data (ensure it's loaded)
if not ensure_data_loaded():
    st.stop()

pat = st.session_state.get("patients_df")
zip_feats = st.session_state.get("zip_features")

if pat is None:
    st.error("Patient data not found. Please load data first.")
    st.stop()

if zip_feats is None:
    st.warning("Neighborhood features not found. Graph construction will be limited.")

# ---------------------------------------------------------------------
# Run Analysis
# ---------------------------------------------------------------------
st.divider()
st.subheader("3. Run Analysis")

if len(selected_groups) < 2:
    st.caption("Select at least 2 groups to run comparison")

run_button = st.button("Run Comparison", type="primary", disabled=len(selected_groups) < 2)

if run_button and len(selected_groups) >= 2:
    progress_bar = st.progress(0)
    status_text = st.empty()

    results = []
    total = len(selected_groups)

    for idx, group_name in enumerate(selected_groups):
        status_text.text(f"Processing group {idx+1}/{total}: {group_name}")

        try:
            cfg = psn_groups[group_name]

            # Build PSN matrix from config
            X_fused, patient_cols, zip_cols, encoding_meta = build_psn_matrix_from_config(
                pat, zip_feats, cfg
            )

            # Extract graph construction settings
            k = cfg.get("k", 5)
            knn_type = cfg.get("knn_type", "mutual")
            ann_mode = cfg.get("ann_mode", "auto")
            similarity_metric = cfg.get("similarity_metric", "cosine")
            weight_balance = cfg.get("weight_balance", 0.3)
            experimental_encoding = cfg.get("experimental_encoding", False)

            # Apply block weights
            patient_w = 1.0 - weight_balance
            zip_w = weight_balance

            X_weighted = build_weighted_matrix(
                X_fused=X_fused,
                patient_cols=patient_cols,
                zip_cols=zip_cols,
                patient_w=patient_w,
                zip_w=zip_w,
            )

            # Compute neighbors using appropriate similarity metric
            if similarity_metric == "mixed" and experimental_encoding:
                idxs, sims, _ = topk_mixed_similarity(
                    X_weighted,
                    k,
                    patient_cols=patient_cols,
                    zip_cols=zip_cols,
                    categorical_mappings=encoding_meta.get("categorical_mappings", {}),
                    bitflag_mapping=encoding_meta.get("bitflag_mapping", {}),
                    bitflag_column="comorbidities_encoded",
                    patient_w=patient_w,
                    zip_w=zip_w,
                )
            else:
                idxs, sims, _ = topk_ann_or_exact(
                    X_weighted, k, ann_mode, SIM_BACKEND_THRESHOLD
                )

            # Build graph
            G = build_knn_graph_from_neighbors(idxs, sims, knn_type)

            # Compute metrics
            metrics = compute_psn_metrics(G)
            metrics["group"] = group_name
            results.append(metrics)

        except Exception as e:
            st.error(f"Error processing '{group_name}': {str(e)}")
            results.append({
                "group": group_name,
                "nodes": None,
                "edges": None,
                "communities": None,
                "components": None,
                "isolated_nodes": None,
                "modularity": None,
                "avg_degree": None,
            })

        progress_bar.progress((idx + 1) / total)

    progress_bar.empty()
    status_text.empty()

    # Store results
    results_df = pd.DataFrame(results)
    st.session_state["psn_fg_comparison_results"] = results_df

    st.success(f"Comparison complete for {len(selected_groups)} groups")

# ---------------------------------------------------------------------
# Results Display
# ---------------------------------------------------------------------
if "psn_fg_comparison_results" in st.session_state:
    st.divider()
    st.subheader("4. Results")

    results_df = st.session_state["psn_fg_comparison_results"]

    # Filter to show only currently selected groups
    if selected_groups:
        display_df = results_df[results_df["group"].isin(selected_groups)].copy()
    else:
        display_df = results_df.copy()

    if not display_df.empty:
        # Reorder columns for better readability
        col_order = ["group", "nodes", "edges", "communities", "components",
                     "isolated_nodes", "modularity", "avg_degree"]
        display_df = display_df[col_order]

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download Results CSV",
            data=csv,
            file_name="psn_feature_group_comparison.csv",
            mime="text/csv",
        )
    else:
        st.info("No results to display for selected groups.")
