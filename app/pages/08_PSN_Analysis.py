# app/pages/08_PSN_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

st.set_page_config(page_title="PSN Analysis", layout="wide")
st.title("PSN Analysis")

# ---------------------------------------------------------------------
# Data checks
# ---------------------------------------------------------------------
from app._logic.loader import ensure_data_loaded

if not ensure_data_loaded():
    st.stop()

if "patient_graph_cache" not in st.session_state or not st.session_state["patient_graph_cache"]:
    st.info("No PSN graph found. Run page 07 (PSN Graph) first.")
    st.stop()

graph_cache = st.session_state["patient_graph_cache"]
graph_cache_key = st.session_state.get(
    "active_psn_graph_key", list(graph_cache.keys())[-1]
)
if graph_cache_key not in graph_cache:
    graph_cache_key = list(graph_cache.keys())[-1]
raw_tbl = graph_cache[graph_cache_key]["features"].copy()

if "profile_community" not in raw_tbl.columns:
    st.error("Community assignments not found. Re-run the graph on page 07.")
    st.stop()

n_total_comms = raw_tbl["profile_community"].nunique()
st.caption(
    f"Using graph: `{graph_cache_key}`  |  "
    f"{len(raw_tbl):,} profiles  |  "
    f"{n_total_comms} communities"
)

# ---------------------------------------------------------------------
# Configuration (top of page)
# ---------------------------------------------------------------------
st.subheader("Configuration")

_weight_col_pre = "profile_count" if "profile_count" in raw_tbl.columns else None
_comm_sizes = (
    raw_tbl.groupby("profile_community")[_weight_col_pre].sum()
    if _weight_col_pre
    else raw_tbl.groupby("profile_community").size()
).rename("n_patients")

total_patients = int(_comm_sizes.sum())
default_min = max(10, int(total_patients * 0.001))

min_patients = st.number_input(
    "Min patients per community",
    min_value=1,
    max_value=int(_comm_sizes.max()),
    value=default_min,
    step=1,
    help=(
        "Communities with fewer patients than this are excluded from the analysis. "
        "Default is 0.1% of total patients."
    ),
)

included_comms = _comm_sizes[_comm_sizes >= min_patients].index
excluded_comms = _comm_sizes[_comm_sizes < min_patients].index

st.caption(
    f"**{len(included_comms)}** communities included, "
    f"**{len(excluded_comms)}** excluded (< {min_patients} patients)."
)

if len(excluded_comms) > 0:
    with st.expander(f"Excluded communities ({len(excluded_comms)})", expanded=False):
        excl_tbl = _comm_sizes.loc[excluded_comms].reset_index()
        excl_tbl.columns = ["community", "n_patients"]
        st.dataframe(
            excl_tbl.sort_values("n_patients", ascending=False),
            use_container_width=False, hide_index=True,
        )

run_analysis = st.button("Run Analysis", type="primary")

if not run_analysis and "psn_analysis_results" not in st.session_state:
    st.info("Set the exclusion criteria above and click **Run Analysis**.")
    st.stop()

# If button clicked, run and cache results. Otherwise use cached results.
if run_analysis:
    st.session_state.pop("psn_analysis_results", None)
else:
    # Auto-invalidate if the active graph has changed since last analysis
    cached = st.session_state.get("psn_analysis_results", {})
    if cached.get("graph_cache_key") != graph_cache_key:
        st.session_state.pop("psn_analysis_results", None)

# ---------------------------------------------------------------------
# Feature column detection
# ---------------------------------------------------------------------
BINARY_COLS       = ["Hearingloss", "BrainInjury", "Hypertension", "Alcohol", "Obesity", "Diabetes"]
CATEGORICAL_COLS  = ["SEX", "Race", "AGE_BIN", "LENSTAYD_BIN", "PAYER"]
CONTINUOUS_COLS   = ["LENSTAYD_LOG", "NUM_VISITS", "READMIT_COUNT", "READMIT_RATE", "REVISIT_30"]
GRAPH_METRIC_COLS = ["profile_betweenness", "profile_pagerank", "profile_degree"]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    w = weights.fillna(1.0).to_numpy(dtype=float)
    v = series.fillna(0.0).to_numpy(dtype=float)
    return float(np.average(v, weights=w))

def _entropy(series: pd.Series) -> float:
    counts = series.value_counts(normalize=True)
    return float(stats.entropy(counts))

def build_signature(
    df: pd.DataFrame,
    binary_cols: list,
    categorical_cols: list,
    continuous_cols: list,
    graph_cols: list,
    weight_col: str | None,
    community_col: str = "profile_community",
) -> pd.DataFrame:
    # Decide once globally whether each categorical is binary-like or multi-class
    cat_is_binary = {c: df[c].nunique(dropna=True) <= 2 for c in categorical_cols}

    rows = []
    for comm, grp in df.groupby(community_col, dropna=False):
        w = grp[weight_col] if weight_col else pd.Series(np.ones(len(grp)), index=grp.index)
        row = {"community": comm, "n_patients": int(w.sum())}

        for c in binary_cols:
            vals = pd.to_numeric(grp[c], errors="coerce")
            row[f"{c}_pct"] = round(_weighted_mean(vals, w) * 100, 1)

        for c in categorical_cols:
            col_vals = grp[c].dropna()
            mode_val = col_vals.mode().iloc[0] if not col_vals.mode().empty else None
            row[f"{c}_mode"] = mode_val
            if cat_is_binary[c]:
                mode_pct = (col_vals == mode_val).mean() * 100 if mode_val is not None else None
                row[f"{c}_pct"] = round(mode_pct, 1) if mode_pct is not None else None
            else:
                row[f"{c}_entropy"] = round(_entropy(col_vals), 3) if not col_vals.empty else None

        for c in continuous_cols + graph_cols:
            vals = pd.to_numeric(grp[c], errors="coerce")
            row[f"{c}_mean"] = round(_weighted_mean(vals, w), 4)
            row[f"{c}_sd"]   = round(float(vals.std(skipna=True)), 4)

        rows.append(row)

    return pd.DataFrame(rows).set_index("community").sort_index()

def feature_ranking_raw(
    df: pd.DataFrame,
    binary_cols: list,
    categorical_cols: list,
    continuous_cols: list,
    graph_cols: list,
) -> pd.DataFrame:
    results = []
    comm = df["profile_community"]

    for c in binary_cols:
        groups = [
            pd.to_numeric(g[c], errors="coerce").dropna().tolist()
            for _, g in df.groupby(comm)
        ]
        groups = [g for g in groups if g]
        if len(groups) < 2:
            continue
        try:
            h, p = stats.kruskal(*groups)
        except Exception:
            continue
        results.append({"feature": c, "type": "binary", "H_statistic": round(h, 3), "p_value": round(p, 4)})

    for c in categorical_cols:
        encoded = pd.Categorical(df[c]).codes
        groups = [encoded[comm == v].tolist() for v in comm.unique()]
        groups = [g for g in groups if g]
        if len(groups) < 2:
            continue
        try:
            h, p = stats.kruskal(*groups)
        except Exception:
            continue
        results.append({"feature": c, "type": "categorical", "H_statistic": round(h, 3), "p_value": round(p, 4)})

    for c in continuous_cols + graph_cols:
        groups = [
            pd.to_numeric(g[c], errors="coerce").dropna().tolist()
            for _, g in df.groupby(comm)
        ]
        groups = [g for g in groups if g]
        if len(groups) < 2:
            continue
        try:
            h, p = stats.kruskal(*groups)
        except Exception:
            continue
        results.append({"feature": c, "type": "continuous", "H_statistic": round(h, 3), "p_value": round(p, 4)})

    return pd.DataFrame(results).sort_values("H_statistic", ascending=False).reset_index(drop=True)

def find_outliers(sig_df: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
    numeric = sig_df.select_dtypes(include="number").drop(
        columns=["n_patients", "n_communities", "total_patients"], errors="ignore"
    )
    grand_mean = numeric.mean()
    grand_std  = numeric.std().replace(0, np.nan)
    z_scores   = (numeric - grand_mean) / grand_std
    records = []
    for comm in z_scores.index:
        row = z_scores.loc[comm]
        for feat, z in row[row.abs() > threshold].sort_values(key=abs, ascending=False).items():
            records.append({
                "community":       comm,
                "feature":         feat,
                "community_value": round(float(sig_df.loc[comm, feat]), 3),
                "grand_mean":      round(float(grand_mean[feat]), 3),
                "z_score":         round(float(z), 2),
            })
    return pd.DataFrame(records)

def render_heatmap(sig_df: pd.DataFrame, title: str) -> BytesIO:
    numeric = sig_df.select_dtypes(include="number").drop(
        columns=["n_patients", "n_communities", "total_patients"], errors="ignore"
    ).fillna(0)
    hm_std = (numeric - numeric.mean()) / numeric.std().replace(0, 1)

    if len(hm_std) >= 2:
        Z_hm  = linkage(hm_std.fillna(0), method="ward")
        order = dendrogram(Z_hm, no_plot=True)["leaves"]
        hm_ordered = hm_std.iloc[order]
    else:
        hm_ordered = hm_std

    fig, ax = plt.subplots(
        figsize=(max(10, len(hm_ordered.columns) * 0.6), max(4, len(hm_ordered) * 0.5))
    )
    im = ax.imshow(hm_ordered.values, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(len(hm_ordered.columns)))
    ax.set_xticklabels(hm_ordered.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(hm_ordered)))
    ax.set_yticklabels(hm_ordered.index.tolist(), fontsize=8)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.6, label="z-score")
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf

# ---------------------------------------------------------------------
# Run analysis (or use cached)
# ---------------------------------------------------------------------
if "psn_analysis_results" not in st.session_state:
    tbl = raw_tbl[raw_tbl["profile_community"].isin(included_comms)].copy()

    if tbl.empty:
        st.error("No communities remain after filtering. Lower the minimum patient threshold.")
        st.stop()

    weight_col       = "profile_count" if "profile_count" in tbl.columns else None
    binary_cols      = [c for c in BINARY_COLS       if c in tbl.columns]
    categorical_cols = [c for c in CATEGORICAL_COLS  if c in tbl.columns]
    continuous_cols  = [c for c in CONTINUOUS_COLS   if c in tbl.columns]
    graph_cols       = [c for c in GRAPH_METRIC_COLS if c in tbl.columns]

    n_communities = tbl["profile_community"].nunique()
    tier = 1 if n_communities < 25 else 2 if n_communities <= 50 else 3

    with st.spinner("Computing signatures..."):
        sig = build_signature(tbl, binary_cols, categorical_cols, continuous_cols, graph_cols, weight_col)

    sig_numeric = sig.select_dtypes(include="number").copy()

    # Meta-clustering
    meta_labels = None
    k_cut = None
    Z_link = None
    if len(sig_numeric) >= 4:
        X_clust = sig_numeric.drop(columns=["n_patients"], errors="ignore").fillna(0)
        X_std   = (X_clust - X_clust.mean()) / X_clust.std().replace(0, 1)
        Z_link  = linkage(X_std, method="ward", metric="euclidean")

        if tier == 2:
            last         = Z_link[-10:, 2]
            acceleration = np.diff(last, 2)
            k_cut        = min(max(int(acceleration.argmax()) + 2, 3), 10)
        elif tier == 3:
            last         = Z_link[-20:, 2]
            acceleration = np.diff(last, 2)
            k_cut        = min(max(int(acceleration.argmax()) + 2, 5), 15)

        if k_cut is not None:
            meta_labels = pd.Series(
                fcluster(Z_link, k_cut, criterion="maxclust"),
                index=sig_numeric.index,
                name="meta_group",
            )
            # fcluster maxclust gives AT MOST k_cut groups; record actual count
            k_actual = int(meta_labels.nunique())

    # Store linkage distances for elbow plot (last N merges)
    elbow_window = 10 if tier == 2 else 20 if tier == 3 else 0
    if Z_link is not None and elbow_window > 0:
        elbow_distances = Z_link[-elbow_window:, 2].tolist()
    else:
        elbow_distances = None

    # Meta-group signatures
    meta_sig = None
    if meta_labels is not None:
        tbl_meta = tbl.copy()
        tbl_meta["meta_group"] = tbl_meta["profile_community"].map(meta_labels)
        meta_sig = build_signature(
            tbl_meta, binary_cols, categorical_cols, continuous_cols, graph_cols,
            weight_col, community_col="meta_group",
        )
        meta_sig.index.name = "meta_group"
        comm_counts = meta_labels.reset_index()
        comm_counts.columns = ["community", "meta_group"]
        meta_sig["n_communities"] = comm_counts.groupby("meta_group").size()

    with st.spinner("Running feature ranking..."):
        ranking_df = feature_ranking_raw(tbl, binary_cols, categorical_cols, continuous_cols, graph_cols)

    outlier_df = find_outliers(sig)

    st.session_state["psn_analysis_results"] = {
        "tier":             tier,
        "n_communities":    n_communities,
        "sig":              sig,
        "meta_sig":         meta_sig,
        "meta_labels":      meta_labels,
        "k_cut":            k_cut,
        "k_actual":         k_actual if k_cut is not None else None,
        "ranking_df":       ranking_df,
        "outlier_df":       outlier_df,
        "binary_cols":      binary_cols,
        "categorical_cols": categorical_cols,
        "continuous_cols":  continuous_cols,
        "graph_cols":       graph_cols,
        "elbow_distances":  elbow_distances,
        "graph_cache_key":  graph_cache_key,
    }

res              = st.session_state["psn_analysis_results"]
tier             = res["tier"]
n_communities    = res["n_communities"]
sig              = res["sig"]
meta_sig         = res["meta_sig"]
meta_labels      = res["meta_labels"]
k_cut            = res["k_cut"]
k_actual         = res.get("k_actual", k_cut)
ranking_df       = res["ranking_df"]
outlier_df       = res["outlier_df"]
elbow_distances  = res.get("elbow_distances")

# ---------------------------------------------------------------------
# Results header
# ---------------------------------------------------------------------
st.divider()
tier_label = (
    "Tier 1 — direct inspection (< 25 communities)"
    if tier == 1 else
    f"Tier 2 — meta-clustering applied ({n_communities} communities → {k_actual} meta-groups)"
    if tier == 2 else
    f"Tier 3 — meta-clustering required ({n_communities} communities → {k_actual} meta-groups)"
)
st.info(tier_label)

# ---------------------------------------------------------------------
# Section 1: Heatmap
# ---------------------------------------------------------------------
st.header("1. Signature Heatmap")

if tier == 1:
    st.caption("Rows = individual communities, ordered by hierarchical linkage.")
    hm_buf = render_heatmap(sig, "Community signatures (z-scored)")
    st.image(hm_buf, use_container_width=True)
    st.download_button(
        "Download heatmap (PNG)", data=hm_buf.getvalue(),
        file_name="community_heatmap.png", mime="image/png",
    )
else:
    st.caption(
        f"Rows = {k_actual} meta-groups. "
        "Each meta-group aggregates multiple communities. "
        "Ordered by hierarchical linkage."
    )
    hm_buf = render_heatmap(
        meta_sig, f"Meta-group signatures (z-scored) — {k_actual} groups"
    )
    st.image(hm_buf, use_container_width=True)
    st.download_button(
        "Download heatmap (PNG)", data=hm_buf.getvalue(),
        file_name="meta_group_heatmap.png", mime="image/png",
    )
    with st.expander(f"Individual community heatmap ({n_communities} rows)", expanded=False):
        hm_comm_buf = render_heatmap(
            sig, f"Community signatures (z-scored) — {n_communities} communities"
        )
        st.image(hm_comm_buf, use_container_width=True)
        st.download_button(
            "Download community heatmap (PNG)", data=hm_comm_buf.getvalue(),
            file_name="community_heatmap.png", mime="image/png", key="dl_comm_hm",
        )

# ---------------------------------------------------------------------
# Section 2: Meta-group table (Tier 2/3 only)
# ---------------------------------------------------------------------
if meta_sig is not None:
    st.divider()
    st.header("2. Meta-Group Signatures")
    st.caption(
        f"{k_actual} meta-groups from Ward hierarchical clustering (elbow cut). "
        "n_communities = number of Louvain communities in the group."
    )
    meta_display = meta_sig.reset_index()
    st.dataframe(meta_display, use_container_width=True, hide_index=True)
    st.download_button(
        "Download meta-group signatures (CSV)",
        data=meta_display.to_csv(index=False).encode(),
        file_name="meta_group_signatures.csv", mime="text/csv",
    )

    with st.expander("Community → meta-group assignments", expanded=False):
        assign_tbl = meta_labels.reset_index()
        assign_tbl.columns = ["community", "meta_group"]
        assign_tbl = assign_tbl.merge(
            sig[["n_patients"]].reset_index(), on="community", how="left"
        )
        assign_tbl = assign_tbl.sort_values(
            ["meta_group", "n_patients"], ascending=[True, False]
        )
        st.dataframe(assign_tbl, use_container_width=True, hide_index=True)
        st.download_button(
            "Download assignments (CSV)",
            data=assign_tbl.to_csv(index=False).encode(),
            file_name="meta_group_assignments.csv", mime="text/csv",
        )

    if elbow_distances is not None:
        with st.expander("Elbow plot — how meta-groups were determined", expanded=False):
            st.caption(
                f"Merge distances for the last {len(elbow_distances)} steps. "
                f"Elbow method selected k={k_actual} meta-groups "
                f"(cap was {k_cut})."
            )
            n_steps = len(elbow_distances)
            x_labels = list(range(n_steps, 0, -1))

            fig_elbow, ax_elbow = plt.subplots(figsize=(7, 3.5))
            ax_elbow.plot(x_labels, elbow_distances, marker="o", markersize=4, linewidth=1.5)
            ax_elbow.axvline(
                x=k_actual, color="red", linestyle="--", linewidth=1.2,
                label=f"k={k_actual}",
            )
            ax_elbow.set_xlabel("Number of clusters")
            ax_elbow.set_ylabel("Merge distance")
            ax_elbow.set_title("Ward linkage — merge distances (elbow)")
            ax_elbow.legend(fontsize=8)
            plt.tight_layout()

            elbow_buf = BytesIO()
            fig_elbow.savefig(elbow_buf, format="png", bbox_inches="tight", dpi=130)
            plt.close(fig_elbow)
            elbow_buf.seek(0)
            st.image(elbow_buf, use_container_width=False)
            st.download_button(
                "Download elbow plot (PNG)", data=elbow_buf.getvalue(),
                file_name="elbow_plot.png", mime="image/png", key="dl_elbow",
            )

# ---------------------------------------------------------------------
# Section 3: Community signatures
# ---------------------------------------------------------------------
st.divider()
section_num = 3 if meta_sig is not None else 2
st.header(f"{section_num}. Community Signatures")
st.caption("One row per community. All summary statistics.")

sig_display = sig.reset_index()
if meta_labels is not None:
    sig_display = sig_display.merge(
        meta_labels.reset_index().rename(columns={"index": "community"}),
        on="community", how="left",
    )
    sig_display = sig_display.sort_values(["meta_group", "community"])

st.dataframe(sig_display, use_container_width=True, hide_index=True)
st.download_button(
    "Download community signatures (CSV)",
    data=sig_display.to_csv(index=False).encode(),
    file_name="community_signatures.csv", mime="text/csv",
)

# ---------------------------------------------------------------------
# Section 4: Feature ranking
# ---------------------------------------------------------------------
st.divider()
section_num += 1
st.header(f"{section_num}. Feature Ranking")
st.caption("Kruskal-Wallis H-statistic across all communities. Higher H = stronger differentiator.")

st.dataframe(ranking_df, use_container_width=True, hide_index=True)
st.download_button(
    "Download feature ranking (CSV)",
    data=ranking_df.to_csv(index=False).encode(),
    file_name="feature_ranking.csv", mime="text/csv",
)

# ---------------------------------------------------------------------
# Section 5: Outlier communities
# ---------------------------------------------------------------------
st.divider()
section_num += 1
st.header(f"{section_num}. Outlier Communities")
st.caption("Communities with |z-score| > 2.0 on any summary feature, relative to the grand mean.")

if outlier_df.empty:
    st.info("No outlier communities detected at |z| > 2.0.")
else:
    st.dataframe(outlier_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download outliers (CSV)",
        data=outlier_df.to_csv(index=False).encode(),
        file_name="outlier_communities.csv", mime="text/csv",
    )
