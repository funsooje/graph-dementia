# app/pages/09_PSN_Feature_Group_Comparison.py

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import community as community_louvain
import hashlib

try:
    from pynndescent import NNDescent
    HAS_PYNNDESCENT = True
except ImportError:
    HAS_PYNNDESCENT = False

st.set_page_config(page_title="PSN Feature Group Comparison", layout="wide")
st.title("PSN Feature Group Comparison")

st.caption("Compare PSN feature groups across different k values")

# ---------------------------------------------------------------------
# Constants & File persistence
# ---------------------------------------------------------------------
SIM_BACKEND_THRESHOLD = 5000
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


# Load from file into session state
if "psn_feature_groups" not in st.session_state:
    st.session_state["psn_feature_groups"] = load_psn_groups()

# ---------------------------------------------------------------------
# Helper functions for building PSN from config
# ---------------------------------------------------------------------
def _as_categorical_str(s: pd.Series, unknown_label: str = "Unknown") -> pd.Series:
    s2 = s.astype("string")
    s2 = s2.fillna(unknown_label)
    s2 = s2.str.strip().replace("", unknown_label)
    return s2


def _make_profile_id(df_row: pd.Series, cols: list, prefix: str) -> str:
    vals = [str(df_row[c]) for c in cols]
    h = hashlib.md5(("||".join(vals)).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{h}"


def _one_hot(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame(index=df.index)
    return pd.get_dummies(df[cols].astype("category"), drop_first=False, dtype=float)


def _standardize(df_num: pd.DataFrame) -> pd.DataFrame:
    if df_num.empty:
        return df_num
    scaler = StandardScaler()
    arr = scaler.fit_transform(df_num.values.astype(float))
    return pd.DataFrame(arr, index=df_num.index, columns=df_num.columns)


def _safe_merge(left: pd.DataFrame, right: pd.DataFrame, on: str, how: str = "left") -> pd.DataFrame:
    L, R = left.copy(), right.copy()
    if on in L.columns:
        L[on] = L[on].astype("string").str.strip().str.replace(" ", "", regex=False)
    if on in R.columns:
        R[on] = R[on].astype("string").str.strip().str.replace(" ", "", regex=False)
    return L.merge(R, on=on, how=how)


def build_psn_matrix_from_config(pat: pd.DataFrame, zip_feats: pd.DataFrame, config: dict):
    """
    Build PSN feature matrix from a saved configuration.
    Returns (X_fused, n_rows) or (None, 0) on error.
    """
    selected_cols = config.get("selected_cols", [])
    if not selected_cols:
        return None, 0

    zip_cfg = config.get("neighborhood_features", {})
    split_by_zip = config.get("split_by_zip", False)

    # Build working dataframe
    work_cols = list(dict.fromkeys(selected_cols + (["ZIPCODE"] if "ZIPCODE" in pat.columns else [])))
    dfw = pat[work_cols].copy()

    for c in selected_cols:
        if c in dfw.columns:
            dfw[c] = _as_categorical_str(dfw[c])
    if "ZIPCODE" in dfw.columns:
        dfw["ZIPCODE"] = _as_categorical_str(dfw["ZIPCODE"]).str.replace(" ", "", regex=False)

    # Build base profiles
    base_grp = dfw.groupby(selected_cols, dropna=False).size().reset_index(name="profile_count")
    base_grp["profile_id"] = base_grp.apply(lambda r: _make_profile_id(r, selected_cols, "prof"), axis=1)

    profiles_by_zip = None
    zip_counts = None

    if "ZIPCODE" not in selected_cols and "ZIPCODE" in dfw.columns:
        if split_by_zip:
            cols_zip = selected_cols + ["ZIPCODE"]
            pbz = dfw.groupby(cols_zip, dropna=False).size().reset_index(name="profile_count")
            pbz["profile_zip_id"] = pbz.apply(lambda r: _make_profile_id(r, cols_zip, "profzip"), axis=1)
            profiles_by_zip = pbz
        else:
            zip_counts = (
                dfw.groupby(selected_cols + ["ZIPCODE"], dropna=False)
                   .size().reset_index(name="n")
            )

    # Determine path and build fused table
    use_split = ("ZIPCODE" in selected_cols) or split_by_zip

    if use_split:
        prof_tbl = profiles_by_zip if profiles_by_zip is not None else base_grp
        if zip_feats is not None and "ZIPCODE" in prof_tbl.columns:
            fused_tbl = _safe_merge(prof_tbl, zip_feats, on="ZIPCODE", how="left")
        else:
            fused_tbl = prof_tbl
    else:
        base = base_grp
        if zip_counts is not None and zip_feats is not None:
            zc2 = _safe_merge(zip_counts, zip_feats, on="ZIPCODE", how="left")

            # Numeric ZIP features to weighted-average
            use_env = zip_cfg.get("environment_index", False)
            use_ses = zip_cfg.get("ses_index", False)
            use_degree = zip_cfg.get("zip_degree", False)
            use_pr = zip_cfg.get("zip_pagerank", False)
            use_btw = zip_cfg.get("zip_betweenness", False)

            zip_num_cols = []
            if use_env:
                zip_num_cols.append("environment_index")
            if use_ses:
                zip_num_cols.append("ses_index")
            if use_degree and "zip_degree" in zc2.columns:
                zip_num_cols.append("zip_degree")
            if use_pr and "zip_pagerank" in zc2.columns:
                zip_num_cols.append("zip_pagerank")
            if use_btw and "zip_betweenness" in zc2.columns:
                zip_num_cols.append("zip_betweenness")

            if zip_num_cols:
                zc2["n"] = zc2["n"].astype(float)
                grp = zc2.groupby(selected_cols, dropna=False)
                num_wavg = grp.apply(lambda g: pd.Series(
                    {col: np.average(g[col].fillna(0.0), weights=g["n"]) for col in zip_num_cols}
                )).reset_index()
                fused_tbl = base.merge(num_wavg, on=selected_cols, how="left")
            else:
                fused_tbl = base.copy()
        else:
            fused_tbl = base.copy()

    # Build patient block (categoricals/binaries)
    RISK = {"Hearingloss", "BrainInjury", "Hypertension", "Alcohol", "Obesity", "Diabetes"}
    bin_cols = [c for c in selected_cols if c in RISK and c in fused_tbl.columns]
    cat_cols = [c for c in selected_cols if c not in bin_cols]

    X_cat = _one_hot(fused_tbl, cat_cols)
    X_bin = pd.DataFrame(index=fused_tbl.index)
    for c in bin_cols:
        X_bin[c] = pd.to_numeric(fused_tbl[c], errors="coerce").fillna(0.0).clip(0, 1).astype(float)
    patient_block = pd.concat([X_cat, X_bin], axis=1)

    # Build ZIP block
    use_env = zip_cfg.get("environment_index", False)
    use_ses = zip_cfg.get("ses_index", False)
    use_degree = zip_cfg.get("zip_degree", False)
    use_pr = zip_cfg.get("zip_pagerank", False)
    use_btw = zip_cfg.get("zip_betweenness", False)
    onehot_comm = zip_cfg.get("onehot_zip_community", False)

    zip_num_cols2 = []
    if use_env and "environment_index" in fused_tbl.columns:
        zip_num_cols2.append("environment_index")
    if use_ses and "ses_index" in fused_tbl.columns:
        zip_num_cols2.append("ses_index")
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

    zip_onehot_df = pd.DataFrame(index=fused_tbl.index)
    if onehot_comm and use_split and "zip_community" in fused_tbl.columns:
        zip_onehot_df = _one_hot(fused_tbl, ["zip_community"])

    zip_block = pd.concat([zip_num_std, zip_onehot_df], axis=1)

    # Final fused matrix (unweighted)
    X_fused = pd.concat([patient_block, zip_block], axis=1).fillna(0.0)

    # Apply weight balance
    weight_balance = config.get("weight_balance", 0.3)
    patient_w = 1.0 - weight_balance
    zip_w = weight_balance

    X_arr = X_fused.values.astype(float)
    n_pat = patient_block.shape[1]
    n_zip = zip_block.shape[1]

    if n_pat > 0:
        X_arr[:, :n_pat] *= patient_w
    if n_zip > 0:
        X_arr[:, n_pat:n_pat + n_zip] *= zip_w

    return X_arr, X_arr.shape[0]


# ---------------------------------------------------------------------
# Graph building functions
# ---------------------------------------------------------------------
def topk_exact_from_matrix(X: np.ndarray, k: int):
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
    return idxs, sims


def topk_ann_or_exact(X: np.ndarray, k: int, ann_mode: str, sim_threshold: int):
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
        return idxs, sims

    return topk_exact_from_matrix(X, k)


def build_knn_graph_from_neighbors(topk_idx: np.ndarray, topk_sim: np.ndarray, knn_type: str):
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
    n = G.number_of_nodes()
    G_u = G.to_undirected() if G.is_directed() else G

    partition = community_louvain.best_partition(G_u, weight="weight", random_state=42)

    edges = G.number_of_edges()
    num_communities = len(set(partition.values()))
    isolated_nodes = sum(1 for _ in nx.isolates(G_u))
    n_components = nx.number_connected_components(G_u)
    non_isolated_communities = num_communities - isolated_nodes
    non_isolated_components = n_components - isolated_nodes

    try:
        modularity = community_louvain.modularity(partition, G_u, weight="weight")
    except Exception:
        modularity = None

    avg_degree = float(np.mean([d for _, d in G.degree(weight=None)])) if n > 0 else 0.0

    return {
        "nodes": n,
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
# Data checks
# ---------------------------------------------------------------------
from app._logic.loader import ensure_data_loaded

if not ensure_data_loaded():
    st.stop()

pat = st.session_state.get("patients_df")
zip_feats = st.session_state.get("zip_features")
psn_groups = st.session_state.get("psn_feature_groups", {})

if not psn_groups:
    st.warning("No PSN feature groups defined. Go to page 08 to create feature groups first.")
    st.stop()

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
st.subheader("Analysis Settings")

col1, col2 = st.columns(2)

with col1:
    selected_groups = st.multiselect(
        "Select Feature Groups",
        options=list(psn_groups.keys()),
        default=None,
        help="Select PSN feature groups to compare"
    )

with col2:
    k_values_input = st.text_input(
        "k values (comma-separated)",
        value="3, 5, 7, 10",
        help="Enter k values separated by commas"
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

if not k_values:
    st.warning("Please enter valid k values (integers between 1 and 50).")

if not selected_groups:
    st.info("Select feature groups above to compare.")

# ---------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------
METRICS = [
    "nodes",
    "edges",
    "n_communities",
    "non_isolated_communities",
    "n_components",
    "non_isolated_components",
    "isolated_nodes",
    "modularity",
    "avg_degree",
]

if run_analysis and selected_groups and k_values:
    progress_bar = st.progress(0)
    status_text = st.empty()

    results_dict = {}
    total_iterations = len(selected_groups) * len(k_values)
    current_iteration = 0

    for group_name in selected_groups:
        config = psn_groups[group_name]
        status_text.text(f"Building PSN matrix for: {group_name}")

        # Build the PSN matrix for this configuration
        X_fused, n_rows = build_psn_matrix_from_config(pat, zip_feats, config)

        if X_fused is None or n_rows == 0:
            st.error(f"Failed to build PSN matrix for '{group_name}'")
            for k in k_values:
                results_dict[(group_name, k)] = {m: None for m in METRICS}
                current_iteration += 1
                progress_bar.progress(current_iteration / total_iterations)
            continue

        for k in k_values:
            status_text.text(f"Processing: {group_name}, k={k}")

            try:
                # Get neighbors
                idxs, sims = topk_ann_or_exact(X_fused, k, ann_mode, SIM_BACKEND_THRESHOLD)

                # Build graph
                G = build_knn_graph_from_neighbors(idxs, sims, knn_type)

                # Compute metrics
                metrics = compute_psn_metrics(G)
                results_dict[(group_name, k)] = metrics

            except Exception as e:
                st.error(f"Error processing {group_name}, k={k}: {str(e)}")
                results_dict[(group_name, k)] = {m: None for m in METRICS}

            current_iteration += 1
            progress_bar.progress(current_iteration / total_iterations)

    progress_bar.empty()
    status_text.empty()

    # Build cross-tab DataFrame
    rows = []
    for idx, group_name in enumerate(selected_groups):
        for metric_idx, metric in enumerate(METRICS):
            row = {
                "Feature Group": group_name if metric_idx == 0 else "",
                "Metric": metric
            }
            for k in k_values:
                value = results_dict.get((group_name, k), {}).get(metric)
                row[f"k={k}"] = value
            rows.append(row)

        # Add blank row between groups (except after last)
        if idx < len(selected_groups) - 1:
            blank_row = {"Feature Group": "", "Metric": ""}
            for k in k_values:
                blank_row[f"k={k}"] = ""
            rows.append(blank_row)

    crosstab_df = pd.DataFrame(rows)

    # Store in session state
    st.session_state["psn_fg_comparison_results"] = crosstab_df
    st.session_state["psn_fg_comparison_settings"] = {
        "groups": selected_groups,
        "k_values": k_values,
        "knn_type": knn_type,
        "ann_mode": ann_mode,
    }

    st.success(f"Analysis complete: {len(selected_groups)} groups × {len(k_values)} k values")

# ---------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------
if "psn_fg_comparison_results" in st.session_state:
    crosstab_df = st.session_state["psn_fg_comparison_results"]
    settings = st.session_state.get("psn_fg_comparison_settings", {})

    st.subheader("Cross-Tab Results")

    if settings:
        st.caption(
            f"Graph type: {settings.get('knn_type', 'N/A')}, "
            f"Similarity mode: {settings.get('ann_mode', 'N/A')}"
        )

    # Calculate height
    table_height = (len(crosstab_df) + 1) * 35 + 10
    st.dataframe(crosstab_df, use_container_width=True, hide_index=True, height=table_height)

    # Download button
    csv = crosstab_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="psn_feature_group_comparison.csv",
        mime="text/csv",
    )

    # Show feature group configs used
    with st.expander("Feature Group Configurations Used", expanded=False):
        for gname in settings.get("groups", []):
            if gname in psn_groups:
                config = psn_groups[gname]
                st.markdown(f"**{gname}**")
                st.text(f"  Profile cols: {', '.join(config.get('selected_cols', []))}")
                zip_feats_cfg = config.get("neighborhood_features", {})
                active_zip = [k for k, v in zip_feats_cfg.items() if v]
                st.text(f"  Neighborhood: {', '.join(active_zip) if active_zip else 'None'}")
                st.text(f"  Split by ZIP: {'Yes' if config.get('split_by_zip') else 'No'}")
                st.markdown("---")
