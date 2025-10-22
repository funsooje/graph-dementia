import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import logging, sys, time, os
from pandas.api.types import is_numeric_dtype

st.set_page_config(page_title="Patient Community Summaries", layout="wide")
st.title("05 - Patient Community Summaries")

# ---------------------------- Logging ----------------------------
if not st.session_state.get("pcs_logger_setup", False):
    LOGGER = logging.getLogger("PCS")
    LOGGER.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s"))
    if not LOGGER.handlers:
        LOGGER.addHandler(h)
    st.session_state["pcs_LOGGER"] = LOGGER
    st.session_state["pcs_logger_setup"] = True
else:
    LOGGER = st.session_state["pcs_LOGGER"]

def _elapsed(t0: float) -> str:
    return f"{(time.perf_counter() - t0):.2f}s"

# ---------------------------- Inputs ----------------------------
active = st.session_state.get("active_patient_graph")
patients_df = st.session_state.get("patients_df")
pf_meta = st.session_state.get("pf_profiles_meta", {})

if not active or "graph" not in active or "features" not in active:
    st.error("No active patient graph. Go to page 04 and click 'Set as patient graph'.")
    st.stop()

G = active["graph"]
tbl = active["features"].copy()
params = active.get("settings", active.get("params", {}))

required_cols = {"profile_id", "profile_community", "profile_count"}
missing = sorted(list(required_cols - set(tbl.columns)))
if missing:
    st.warning(f"Missing expected columns in features: {missing}")

# ---------------------------- Feature sets ----------------------------
RISK_COLS = ["Hearingloss", "BrainInjury", "Hypertension", "Alcohol", "Obesity", "Diabetes"]
DEMO_COLS = ["SEX", "AGE_BIN", "Race"]
ZIP_NUM_COLS = ["environment_index", "ses_index", "zip_pagerank"]

# Plot guards
COMMUNITY_PLOT_HARD_MAX = int(os.getenv("PCS_COMMUNITY_PLOT_HARD_MAX", 200))
RISK_HEATMAP_MAX_FEATURES = int(os.getenv("PCS_RISK_HEATMAP_MAX_FEATURES", 18))
RISK_HEATMAP_MAX_CELLS = int(os.getenv("PCS_RISK_HEATMAP_MAX_CELLS", 12000))

# ---------------------------- Helpers ----------------------------
def _community_order(df: pd.DataFrame, by: str = "patients") -> list[int]:
    grp = df.groupby("profile_community")
    if by == "patients" and "profile_count" in df.columns:
        return grp["profile_count"].sum().sort_values(ascending=False).index.tolist()
    return grp.size().sort_values(ascending=False).index.tolist()

def _bar(series: pd.Series, title: str, ylabel: str = "Count"):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(series.index.astype(str), series.values)
    ax.set_title(title); ax.set_xlabel("Community"); ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig

def _heatmap(mat: np.ndarray, rows: list, cols: list, title: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([str(r) for r in rows])
    ax.set_title(title); ax.set_xlabel("Feature"); ax.set_ylabel("Community")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig

def _wmean(x: pd.Series, w: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").fillna(0.0)
    w = pd.to_numeric(w, errors="coerce").fillna(0.0)
    return float(np.average(x, weights=w)) if w.sum() > 0 else float(x.mean())

def _wprev_bin(x: pd.Series, w: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").fillna(0.0).clip(0, 1)
    w = pd.to_numeric(w, errors="coerce").fillna(0.0)
    return float(np.average(x, weights=w)) if w.sum() > 0 else float(x.mean())

def _assort(df: pd.DataFrame, cat: str) -> float:
    if cat not in df.columns: return np.nan
    tmp = df.copy()
    tmp["_w"] = pd.to_numeric(tmp["profile_count"], errors="coerce").fillna(0.0)
    tmp[cat] = tmp[cat].astype("string").fillna("Unknown")
    piv = tmp.pivot_table(index="profile_community", columns=cat, values="_w", aggfunc="sum", fill_value=0.0)
    N = piv.values.sum()
    if N <= 0: return np.nan
    comm_tot = piv.sum(axis=1).values
    p_comm = comm_tot / N
    with np.errstate(divide="ignore", invalid="ignore"):
        p2 = (piv.values / comm_tot[:, None])**2
        p2[np.isnan(p2)] = 0.0
        hom = float((p2.sum(axis=1) * p_comm).sum())
    p_cat = piv.sum(axis=0).values / N
    baseline = float((p_cat**2).sum())
    if 1.0 - baseline <= 1e-9: return 0.0
    return float((hom - baseline) / (1.0 - baseline))

def _signatures(df: pd.DataFrame, risk_cols: list[str], delta: float = 0.05) -> pd.DataFrame:
    present = [c for c in risk_cols if c in df.columns]
    if not present: return pd.DataFrame(columns=["community", "feature", "delta"])
    w_all = pd.to_numeric(df["profile_count"], errors="coerce").fillna(0.0)
    gprev = {c: _wprev_bin(df[c], w_all) for c in present}
    rows = []
    for cid, g in df.groupby("profile_community"):
        w = pd.to_numeric(g["profile_count"], errors="coerce").fillna(0.0)
        for c in present:
            d = _wprev_bin(g[c], w) - gprev[c]
            if d >= delta:
                rows.append({"community": cid, "feature": c, "delta": round(float(d), 4)})
    return pd.DataFrame(rows).sort_values(["community", "delta"], ascending=[True, False])

# ---------------------------- Sidebar ----------------------------
with st.sidebar:
    st.header("View")
    order_by = st.selectbox("Order communities by", ["patients", "profiles"], index=0)
    max_comm = st.number_input("Max communities in plots", 5, 100, 25, 1)
    sig_delta = st.slider("Signature Δ above global prevalence", 0.0, 0.2, 0.05, 0.01)
    can_join_los = patients_df is not None and isinstance(pf_meta.get("selected_cols", []), list)
    show_los = st.checkbox("Compute LOS by community (join patients_df)", value=can_join_los)

# ---------------------------- Echo ----------------------------
st.subheader("Inputs summary")
st.write({
    "features_shape": tbl.shape,
    "graph_nodes": G.number_of_nodes(),
    "graph_edges": G.number_of_edges(),
    "patients_df_rows": None if patients_df is None else len(patients_df),
})
st.subheader("Active graph settings")
st.write({
    "k": params.get("k"),
    "knn_type": params.get("knn_type"),
    "layout": params.get("layout"),
    "patient_weight": params.get("patient_weight") or params.get("patient_w"),
    "zip_weight": params.get("zip_weight") or params.get("zip_w"),
})

# ---------------------------- A. Community structure ----------------------------
st.subheader("A. Community structure")
comm_order = _community_order(tbl, by=order_by)
effective_max = min(int(max_comm), COMMUNITY_PLOT_HARD_MAX)

comm_counts = tbl.groupby("profile_community").size().rename("profiles")
comm_patients = tbl.groupby("profile_community")["profile_count"].sum().rename("patients")
comm_summary = pd.concat([comm_counts, comm_patients], axis=1).fillna(0)
comm_summary["profiles"] = comm_summary["profiles"].astype(int)
comm_summary["patients"] = comm_summary["patients"].astype(int)
den = comm_summary["patients"].replace(0, np.nan)
comm_summary["profiles_per_patient"] = (comm_summary["profiles"] / den).round(3)
comm_summary = comm_summary.loc[comm_order]

st.dataframe(comm_summary, width="content")
st.pyplot(_bar(comm_summary["profiles"].head(effective_max), "Profiles per community"))
st.pyplot(_bar(comm_summary["patients"].head(effective_max), "Patients per community"))

# ---------------------------- B. Risk prevalence heatmap ----------------------------
st.subheader("B. Risk factor prevalence by community")
present_risks = [c for c in RISK_COLS if c in tbl.columns]
if present_risks:
    rows, labels = [], []
    for cid, g in tbl.groupby("profile_community"):
        w = pd.to_numeric(g["profile_count"], errors="coerce").fillna(0.0)
        rows.append([_wprev_bin(g[c], w) for c in present_risks]); labels.append(cid)
    order = [c for c in comm_order if c in labels]
    idx = [labels.index(c) for c in order]
    mat = np.array(rows)[idx, :]
    if len(present_risks) <= RISK_HEATMAP_MAX_FEATURES and len(order[:effective_max]) * len(present_risks) <= RISK_HEATMAP_MAX_CELLS:
        st.pyplot(_heatmap(mat[:effective_max, :], order[:effective_max], present_risks,
                           "Risk prevalence (patient-weighted)"))
    else:
        st.info("Risk heatmap skipped due to size; reduce features or communities.")
else:
    st.info("Risk binaries not present; skipping heatmap.")

# ---------------------------- C. ZIP context summaries ----------------------------
st.subheader("C. ZIP context summaries (mean ± sd)")
zip_cols_present = [c for c in ZIP_NUM_COLS if c in tbl.columns]
if zip_cols_present:
    blocks = []
    for cid, g in tbl.groupby("profile_community"):
        w = pd.to_numeric(g["profile_count"], errors="coerce").fillna(0.0)
        stats = {}
        for c in zip_cols_present:
            mu = _wmean(g[c], w)
            x = pd.to_numeric(g[c], errors="coerce").fillna(0.0)
            var = float(np.average((x - mu) ** 2, weights=w)) if w.sum() > 0 else float(x.var())
            stats[c] = {"mean": round(mu, 4), "std": round(np.sqrt(var), 4)}
        df = pd.DataFrame(stats).T
        df.insert(0, "community", cid)
        blocks.append(df.reset_index().rename(columns={"index": "feature"}))
    zip_stats = pd.concat(blocks, ignore_index=True).sort_values(["community", "feature"]).reset_index(drop=True)
    st.dataframe(zip_stats, width="content")
else:
    st.info("No ZIP numeric columns present; skipping ZIP summaries.")

# ---------------------------------------------------------------------
# D. Length of stay (LOS) by community
# ---------------------------------------------------------------------
st.subheader("D. Length of stay (LOS) by community")

if show_los and patients_df is not None:
    selected_cols = pf_meta.get("selected_cols", [])
    if not selected_cols:
        st.info("Profile selected_cols metadata not found; cannot join patients_df.")
    else:
        # choose join keys that exist on both sides
        join_cols = [c for c in selected_cols if c in tbl.columns and c in patients_df.columns]
        if not join_cols:
            st.info("No overlapping selected columns between features and patients_df.")
        else:
            # Build minimal left/right tables
            left = tbl[join_cols + ["profile_community", "profile_count"]].copy()
            right_cols = join_cols + [c for c in ["LENSTAYD", "LENSTAYD_BIN"] if c in patients_df.columns]
            right = patients_df[right_cols].copy()

            # ---- Helper: normalize join keys BEFORE any groupby/merge ----
            def _normalize_keys(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
                out = df.copy()
                for c in keys:
                    if c not in out.columns:
                        continue
                    s = out[c]

                    # Try numeric → round to integer for binned/binary fields;
                    # otherwise keep original (will be stringified).
                    s_num = pd.to_numeric(s, errors="coerce")
                    if s_num.notna().any() and not s.dtype == bool:
                        s = s_num.round(0).astype("Int64")

                    # Final canonical string form
                    s = s.astype("string").str.strip().fillna("Unknown").replace("", "Unknown")
                    out[c] = s
                return out

            left_keys  = _normalize_keys(left, join_cols)
            right_keys = _normalize_keys(right, join_cols)

            # Which LOS do we have?
            has_L = "LENSTAYD" in right_keys.columns
            has_B = "LENSTAYD_BIN" in right_keys.columns

            if has_L:
                # Ensure numeric LOS
                right_keys["LENSTAYD"] = pd.to_numeric(right_keys["LENSTAYD"], errors="coerce")

                # Group RIGHT by normalized keys, compute mean LOS per key-combo
                grp = (right_keys
                       .groupby(join_cols, dropna=False)["LENSTAYD"]
                       .mean()
                       .reset_index()
                       .rename(columns={"LENSTAYD": "mean_LOS"}))

                # Merge on normalized keys
                merged = left_keys.merge(grp, on=join_cols, how="left")

                # Weighted mean LOS per community (weights = profile_count)
                los_by_comm = (merged.groupby("profile_community")
                               .apply(lambda g: np.average(pd.to_numeric(g["mean_LOS"], errors="coerce").fillna(0.0),
                                                           weights=pd.to_numeric(g["profile_count"], errors="coerce").fillna(0.0)))
                               .rename("weighted_mean_LOS")
                               .sort_index())

                st.dataframe(los_by_comm.to_frame(), width="content")
                st.pyplot(_bar(los_by_comm, "Weighted mean LOS by community", ylabel="Mean LOS"))

            elif has_B:
                # Normalize the categorical LOS column as well
                los_cat = "LENSTAYD_BIN"
                right_keys[los_cat] = (right_keys[los_cat]
                                       .astype("string").str.strip()
                                       .fillna("Unknown").replace("", "Unknown"))

                # Counts by (join_cols, LOS bin)
                grp = (right_keys
                       .groupby(join_cols + [los_cat], dropna=False)
                       .size()
                       .rename("n")
                       .reset_index())

                # Merge counts onto left side (normalized keys)
                merged = left_keys.merge(grp, on=join_cols, how="left")
                merged["n"] = pd.to_numeric(merged["n"], errors="coerce").fillna(0.0)

                # Proportions per community across LOS bins
                prop_blocks, all_bins = [], set()
                for cid, g in merged.groupby("profile_community"):
                    counts = g.groupby(los_cat)["n"].sum().sort_index()
                    props = counts / counts.sum() if counts.sum() > 0 else counts
                    prop_blocks.append((cid, props))
                    all_bins |= set(props.index)

                all_bins = sorted(all_bins)
                mat, row_idx = [], []
                for cid, props in prop_blocks:
                    row_idx.append(cid)
                    mat.append([props.get(b, 0.0) for b in all_bins])
                df_prop = pd.DataFrame(mat, index=row_idx, columns=all_bins).fillna(0.0)

                # Reorder by earlier community order and cap rows for display
                df_prop = df_prop.loc[[c for c in comm_order if c in df_prop.index]].head(effective_max)

                st.dataframe(df_prop.head(15), width="content")

                # Stacked bar chart
                fig, ax = plt.subplots(figsize=(10, 5))
                x = np.arange(len(df_prop.index))
                bottom = np.zeros(len(df_prop.index))
                for col in df_prop.columns:
                    ax.bar(x, df_prop[col].values, bottom=bottom, label=str(col))
                    bottom += df_prop[col].values
                ax.set_xticks(x)
                ax.set_xticklabels(df_prop.index.astype(str), rotation=45, ha="right")
                ax.set_ylabel("Proportion")
                ax.set_xlabel("Community")
                ax.set_title(f"{los_cat} proportions by community (patients_df join)")
                ax.legend(ncol=2, fontsize=8)
                fig.tight_layout()
                st.pyplot(fig)

            else:
                st.info("Neither LENSTAYD nor LENSTAYD_BIN present in patients_df.")

else:
    st.info("LOS computation disabled or patients_df unavailable.")

# ---------------------------- E. Demographic homophily ----------------------------
st.subheader("E. Demographic homophily (assortativity) within communities")
assort_sex = _assort(tbl, "SEX"); assort_age = _assort(tbl, "AGE_BIN")
st.write({"assortativity_SEX": None if pd.isna(assort_sex) else round(float(assort_sex), 4),
          "assortativity_AGE_BIN": None if pd.isna(assort_age) else round(float(assort_age), 4)})

# ---------------------------- F. Community risk signatures ----------------------------
st.subheader("F. Community risk signatures (above global prevalence)")
sig_df = _signatures(tbl, RISK_COLS, delta=float(sig_delta))
st.dataframe(sig_df if not sig_df.empty else pd.DataFrame({"note":["No communities exceeded Δ at this threshold"]}), width="content")

# ---------------------------- G. Centrality leaderboards ----------------------------
st.subheader("G. Centrality leaderboards")
centrality_cols = [c for c in ["profile_pagerank", "profile_betweenness", "profile_degree"] if c in tbl.columns]
if centrality_cols:
    top_n = st.slider("Top N profiles per metric", 5, 50, 10, 1)
    for c in centrality_cols:
        cols = ["profile_id", c, "profile_community", "profile_count"] + [x for x in DEMO_COLS if x in tbl.columns] + [x for x in RISK_COLS if x in tbl.columns]
        df_top = tbl[cols].sort_values(c, ascending=False).head(int(top_n))
        st.write(f"Top {top_n} by {c}")
        st.dataframe(df_top, width="content")
else:
    st.info("Centrality columns not found; leaderboards skipped.")

# ---------------------------- Exports ----------------------------
st.subheader("Exports")
c1, c2, c3 = st.columns(3)
with c1:
    st.download_button(
        "Download community_summary.csv",
        comm_summary.reset_index().rename(columns={"index": "community"}).to_csv(index=False).encode("utf-8"),
        "community_summary.csv", "text/csv"
    )
with c2:
    # rebuild prevalence for export
    risk_export = []
    if present_risks:
        for cid, g in tbl.groupby("profile_community"):
            w = pd.to_numeric(g["profile_count"], errors="coerce").fillna(0.0)
            for c in present_risks:
                risk_export.append({"community": cid, "feature": c, "prevalence": _wprev_bin(g[c], w)})
    st.download_button(
        "Download risk_prevalence_by_community.csv",
        pd.DataFrame(risk_export).to_csv(index=False).encode("utf-8"),
        "risk_prevalence_by_community.csv", "text/csv"
    )
with c3:
    st.download_button(
        "Download profile_features.csv",
        tbl.to_csv(index=False).encode("utf-8"),
        "profile_features.csv", "text/csv"
    )

st.caption("Summaries reflect the active patient graph from Page 04. LOS uses patients_df joined on the profile's selected columns.")