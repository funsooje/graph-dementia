# app/pages/08_PSN_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from scipy.cluster.hierarchy import linkage, dendrogram

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
# Configuration
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

col_cfg1, col_cfg2 = st.columns(2)
with col_cfg1:
    min_patients = st.number_input(
        "Min patients per community",
        min_value=1,
        max_value=int(_comm_sizes.max()),
        value=default_min,
        step=1,
        help="Communities with fewer patients are excluded. Default is 0.1% of total patients.",
    )

included_comms = _comm_sizes[_comm_sizes >= min_patients].index
excluded_comms = _comm_sizes[_comm_sizes < min_patients].index
n_included = len(included_comms)

with col_cfg2:
    default_top_pct = 10
    top_pct = st.slider(
        "Top N% communities to focus on",
        min_value=5,
        max_value=100,
        value=default_top_pct,
        step=5,
        help=(
            "Focus analysis on the largest communities by patient count. "
            "E.g. 10% selects the top 10% of included communities."
        ),
    )

n_top = max(5, int(np.ceil(n_included * top_pct / 100)))
n_top = min(n_top, n_included)

st.caption(
    f"**{n_included}** communities included (≥ {min_patients} patients), "
    f"**{len(excluded_comms)}** excluded.  "
    f"Top **{n_top}** communities selected ({top_pct}% of {n_included})."
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
    st.info("Set the configuration above and click **Run Analysis**.")
    st.stop()

if run_analysis:
    st.session_state.pop("psn_analysis_results", None)
else:
    cached = st.session_state.get("psn_analysis_results", {})
    if cached.get("graph_cache_key") != graph_cache_key:
        st.session_state.pop("psn_analysis_results", None)

# ---------------------------------------------------------------------
# Feature column definitions
# ---------------------------------------------------------------------
BINARY_COLS = [
    "Hearingloss", "BrainInjury", "Hypertension", "Alcohol", "Obesity", "Diabetes"
]
SEX_COL            = "SEX"
MULTICATEGORY_COLS = ["Race", "AGE_BIN", "LENSTAYD_BIN", "PAYER"]
CONTINUOUS_COLS    = ["LENSTAYD_LOG", "NUM_VISITS", "READMIT_COUNT", "READMIT_RATE", "REVISIT_30"]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    w = weights.fillna(1.0).to_numpy(dtype=float)
    v = series.fillna(0.0).to_numpy(dtype=float)
    return float(np.average(v, weights=w))


def build_signature(
    df: pd.DataFrame,
    binary_cols: list,
    has_sex: bool,
    multicategory_cols: list,
    continuous_cols: list,
    weight_col,
    col_categories: dict,
    community_col: str = "profile_community",
) -> pd.DataFrame:
    rows = []
    for comm, grp in df.groupby(community_col, dropna=False):
        w = grp[weight_col].fillna(1.0) if weight_col else pd.Series(
            np.ones(len(grp)), index=grp.index
        )
        row = {"community": comm, "n_patients": int(w.sum())}

        for c in binary_cols:
            vals = pd.to_numeric(grp[c], errors="coerce")
            row[f"{c}_pct"] = round(_weighted_mean(vals, w) * 100, 1)

        if has_sex:
            is_female = (grp[SEX_COL] == "F").astype(float)
            row["SEX_F_pct"] = round(_weighted_mean(is_female, w) * 100, 1)

        for c in multicategory_cols:
            for cat in col_categories.get(c, []):
                is_cat = (grp[c] == cat).astype(float)
                row[f"{c}_{cat}_pct"] = round(_weighted_mean(is_cat, w) * 100, 1)

        for c in continuous_cols:
            vals = pd.to_numeric(grp[c], errors="coerce")
            row[f"{c}_mean"] = round(_weighted_mean(vals, w), 4)
            row[f"{c}_sd"]   = round(float(vals.std(skipna=True)), 4)

        rows.append(row)

    return pd.DataFrame(rows).set_index("community").sort_index()


def find_outliers(sig_df: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
    numeric = sig_df.select_dtypes(include="number").drop(
        columns=["n_patients"], errors="ignore"
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
        columns=["n_patients"], errors="ignore"
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

    weight_col         = "profile_count" if "profile_count" in tbl.columns else None
    binary_cols        = [c for c in BINARY_COLS        if c in tbl.columns]
    has_sex            = SEX_COL in tbl.columns
    multicategory_cols = [c for c in MULTICATEGORY_COLS if c in tbl.columns]
    continuous_cols    = [c for c in CONTINUOUS_COLS    if c in tbl.columns]

    col_categories = {
        c: sorted(tbl[c].dropna().unique().tolist())
        for c in multicategory_cols
    }

    with st.spinner("Computing signatures…"):
        sig = build_signature(
            tbl, binary_cols, has_sex, multicategory_cols, continuous_cols,
            weight_col, col_categories,
        )

    # Select top-N communities by patient count
    top_communities = (
        sig["n_patients"]
        .sort_values(ascending=False)
        .head(n_top)
        .index.tolist()
    )
    top_sig = sig.loc[top_communities].copy()

    outlier_df = find_outliers(top_sig)

    st.session_state["psn_analysis_results"] = {
        "sig":             sig,
        "top_sig":         top_sig,
        "top_communities": top_communities,
        "n_communities":   int(sig.index.nunique()),
        "n_top":           len(top_communities),
        "top_pct":         top_pct,
        "outlier_df":      outlier_df,
        "graph_cache_key": graph_cache_key,
    }

res             = st.session_state["psn_analysis_results"]
sig             = res["sig"]
top_sig         = res["top_sig"]
top_communities = res["top_communities"]
n_communities   = res["n_communities"]
n_top           = res["n_top"]
outlier_df      = res["outlier_df"]

# ---------------------------------------------------------------------
# Results header
# ---------------------------------------------------------------------
st.divider()
st.info(
    f"{n_communities} communities included  |  "
    f"Showing top **{n_top}** by patient count ({res['top_pct']}%)"
)

# ---------------------------------------------------------------------
# Section 1: Heatmap — top N communities
# ---------------------------------------------------------------------
st.header("1. Signature Heatmap (Top Communities)")
st.caption(
    f"Rows = top {n_top} communities ordered by hierarchical linkage. "
    "Colour = z-score relative to the grand mean of the top communities."
)

hm_buf = render_heatmap(top_sig, f"Top {n_top} community signatures (z-scored)")
st.image(hm_buf, use_container_width=True)
st.download_button(
    "Download heatmap (PNG)", data=hm_buf.getvalue(),
    file_name="top_community_heatmap.png", mime="image/png",
)

with st.expander(f"All {n_communities} community heatmap", expanded=False):
    hm_all_buf = render_heatmap(sig, f"All {n_communities} community signatures (z-scored)")
    st.image(hm_all_buf, use_container_width=True)
    st.download_button(
        "Download all-community heatmap (PNG)", data=hm_all_buf.getvalue(),
        file_name="all_community_heatmap.png", mime="image/png", key="dl_all_hm",
    )

# ---------------------------------------------------------------------
# Section 2: Top-N community signatures table
# ---------------------------------------------------------------------
st.divider()
st.header(f"2. Top {n_top} Community Signatures")
st.caption("Sorted by patient count (largest first).")

top_display = top_sig.reset_index().sort_values("n_patients", ascending=False)
st.dataframe(top_display, use_container_width=True, hide_index=True)
st.download_button(
    "Download top community signatures (CSV)",
    data=top_display.to_csv(index=False).encode(),
    file_name="top_community_signatures.csv", mime="text/csv",
)

# ---------------------------------------------------------------------
# Section 3: All community signatures
# ---------------------------------------------------------------------
st.divider()
st.header(f"3. All Community Signatures ({n_communities})")
st.caption("All included communities. One row per community.")

sig_display = sig.reset_index().sort_values("n_patients", ascending=False)
st.dataframe(sig_display, use_container_width=True, hide_index=True)
st.download_button(
    "Download all community signatures (CSV)",
    data=sig_display.to_csv(index=False).encode(),
    file_name="community_signatures.csv", mime="text/csv",
)

# ---------------------------------------------------------------------
# Section 4: Outlier communities (within top N)
# ---------------------------------------------------------------------
st.divider()
st.header(f"4. Outlier Communities (Top {n_top})")
st.caption("Communities with |z-score| > 2.0 on any feature, relative to the top-N grand mean.")

if outlier_df.empty:
    st.info("No outlier communities detected at |z| > 2.0.")
else:
    st.dataframe(outlier_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download outliers (CSV)",
        data=outlier_df.to_csv(index=False).encode(),
        file_name="outlier_communities.csv", mime="text/csv",
    )
