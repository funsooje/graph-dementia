# app/pages/08_PSN_Feature_Groups.py

import streamlit as st
import pandas as pd
import json
from pathlib import Path

st.set_page_config(page_title="PSN Feature Groups", layout="wide")
st.title("PSN Feature Groups")

# ---------------------------------------------------------------------
# File persistence
# ---------------------------------------------------------------------
PSN_GROUPS_FILE = Path("data/config/psn_feature_groups.json")


def load_psn_groups() -> dict:
    """Load PSN feature groups from file."""
    if PSN_GROUPS_FILE.exists():
        try:
            with open(PSN_GROUPS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_psn_groups(groups: dict):
    """Save PSN feature groups to file."""
    PSN_GROUPS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PSN_GROUPS_FILE, "w") as f:
        json.dump(groups, f, indent=2)


# Load from file into session state on first run
if "psn_feature_groups" not in st.session_state:
    st.session_state["psn_feature_groups"] = load_psn_groups()

# ---------------------------------------------------------------------
# Data checks
# ---------------------------------------------------------------------
from app._logic.loader import ensure_data_loaded

if not ensure_data_loaded():
    st.stop()

pat = st.session_state.get("patients_df")
zip_feats = st.session_state.get("zip_features")

# ---------------------------------------------------------------------
# Available columns
# ---------------------------------------------------------------------
PAT_GROUPS = {
    "demographics": ["SEX", "Race", "AGE_BIN"],
    "utilization": ["LENSTAYD_BIN", "LENSTAYD_LOG", "PAYER", "NUM_VISITS"],
    "risk_binaries": ["Hearingloss", "BrainInjury", "Hypertension", "Alcohol", "Obesity", "Diabetes"],
    "outcomes": ["READMIT_PROPORTION", "EVER_READMITTED"],
}

demo_cols = [c for c in PAT_GROUPS["demographics"] if c in pat.columns]
util_cols = [c for c in PAT_GROUPS["utilization"] if c in pat.columns]
risk_cols = [c for c in PAT_GROUPS["risk_binaries"] if c in pat.columns]
zip_available = zip_feats is not None

# ---------------------------------------------------------------------
# Define New Feature Group (compact)
# ---------------------------------------------------------------------
st.subheader("Define New Feature Group")

group_name = st.text_input("Name", placeholder="e.g., Demographics_Only")

# Profile Features - Demographics
st.caption("Demographics")
demo_checks = {}
if demo_cols:
    dcols = st.columns(len(demo_cols))
    for i, col in enumerate(demo_cols):
        with dcols[i]:
            demo_checks[col] = st.checkbox(col, key=f"new_demo_{col}")

# Profile Features - Utilization
st.caption("Utilization")
util_checks = {}
if util_cols:
    ucols = st.columns(len(util_cols))
    for i, col in enumerate(util_cols):
        with ucols[i]:
            util_checks[col] = st.checkbox(col, key=f"new_util_{col}")

# Profile Features - Risk Binaries
st.caption("Risk Binaries")
risk_checks = {}
if risk_cols:
    rcols = st.columns(len(risk_cols))
    for i, col in enumerate(risk_cols):
        with rcols[i]:
            risk_checks[col] = st.checkbox(col, key=f"new_risk_{col}")

# Profile Features - Outcomes
st.caption("Outcomes")
outcome_cols = [c for c in PAT_GROUPS.get("outcomes", []) if c in pat.columns]
outcome_checks = {}
if outcome_cols:
    ocols = st.columns(len(outcome_cols))
    for i, col in enumerate(outcome_cols):
        with ocols[i]:
            outcome_checks[col] = st.checkbox(col, key=f"new_outcome_{col}")

# Gather selected columns
selected_demo = [c for c, v in demo_checks.items() if v]
selected_util = [c for c, v in util_checks.items() if v]
selected_risk = [c for c, v in risk_checks.items() if v]
selected_outcome = [c for c, v in outcome_checks.items() if v]
selected_cols = selected_demo + selected_util + selected_risk + selected_outcome

# Neighborhood features (compact row)
if zip_available:
    st.caption("Neighborhood Features")
    zcol1, zcol2, zcol3, zcol4 = st.columns(4)
    with zcol1:
        use_degree = st.checkbox("zip_degree", key="new_deg")
    with zcol2:
        use_pr = st.checkbox("zip_pagerank", key="new_pr")
    with zcol3:
        use_btw = st.checkbox("zip_betweenness", key="new_btw")
    with zcol4:
        use_zip_comm = st.checkbox("zip_community", key="new_comm", help="Encoding depends on experimental mode")
else:
    use_degree = use_pr = use_btw = use_zip_comm = False
    st.caption("Neighborhood features unavailable (run page 02 first)")

# Weight balance
wcol1, wcol2 = st.columns([2, 1])
with wcol1:
    weight_balance = st.slider(
        "Weight (Profile ← → Neighborhood)",
        min_value=0.0, max_value=1.0, value=0.3, step=0.05,
        help="0.0 = 100% Profile, 1.0 = 100% Neighborhood",
        key="new_weight"
    )
with wcol2:
    st.caption(f"Profile: {1-weight_balance:.0%} | Neighborhood: {weight_balance:.0%}")

st.divider()

# ---------------------------------------------------------------------
# Graph Construction Settings
# ---------------------------------------------------------------------
st.caption("**Graph Construction Settings**")

# Row 1: Encoding mode and k value
gcol1, gcol2 = st.columns(2)

with gcol1:
    experimental_encoding = st.toggle(
        "Compact Encoding (recommended)",
        value=True,
        key="new_experimental",
        help="Compact (default): Integer-encode categoricals + bitflag-encode comorbidities. Standard (unchecked): One-hot encode categoricals."
    )

with gcol2:
    k_value = st.number_input(
        "k (number of neighbors)",
        min_value=1, max_value=50, value=5, step=1,
        key="new_k",
        help="Number of nearest neighbors for k-NN graph construction"
    )

# Row 2: Similarity metric (conditional), k-NN type, ANN mode
gcol3, gcol4, gcol5 = st.columns(3)

with gcol3:
    if experimental_encoding:
        similarity_metric_label = st.selectbox(
            "Similarity Metric",
            options=["Cosine Similarity", "Mixed Similarity (recommended)"],
            index=1,
            key="new_sim_metric",
            help="Cosine: Standard cosine similarity. Mixed: Exact-match for categoricals + Hamming for bitflags + cosine for numeric."
        )
        similarity_metric = "mixed" if similarity_metric_label.startswith("Mixed") else "cosine"
    else:
        st.caption("Similarity Metric")
        st.text("Cosine (standard)")
        similarity_metric = "cosine"

with gcol4:
    knn_type_label = st.selectbox(
        "Graph Type",
        options=["Mutual k-NN (undirected)", "Directed k-NN"],
        index=0,
        key="new_knn_type",
        help="Mutual: Undirected edges (both nodes must be in each other's k-NN). Directed: One-way edges."
    )
    knn_type = "mutual" if knn_type_label.startswith("Mutual") else "directed"

with gcol5:
    ann_mode_label = st.selectbox(
        "Similarity Mode",
        options=["Auto (threshold)", "Force ANN", "Force Exact"],
        index=0,
        key="new_ann_mode",
        help="Auto: Use ANN for large datasets. Force ANN: Always use approximate. Force Exact: Always use exact."
    )
    ann_mode = (
        "auto" if ann_mode_label.startswith("Auto") else
        "force_ann" if ann_mode_label.startswith("Force ANN") else
        "force_exact"
    )

st.divider()

# Save button
if st.button("Save Feature Group", type="primary"):
    if not group_name.strip():
        st.error("Enter a name.")
    elif not selected_cols:
        st.error("Select at least one profile column.")
    elif group_name in st.session_state["psn_feature_groups"]:
        st.error(f"'{group_name}' already exists.")
    else:
        config = {
            "name": group_name,
            "selected_cols": selected_cols,
            "profile_cols": {
                "demographics": selected_demo,
                "utilization": selected_util,
                "risk_binaries": selected_risk,
            },
            "neighborhood_features": {
                "zip_degree": use_degree,
                "zip_pagerank": use_pr,
                "zip_betweenness": use_btw,
                "zip_community": use_zip_comm,
            },
            "weight_balance": weight_balance,
            # Graph construction settings
            "experimental_encoding": experimental_encoding,
            "similarity_metric": similarity_metric,
            "k": int(k_value),
            "knn_type": knn_type,
            "ann_mode": ann_mode,
        }
        st.session_state["psn_feature_groups"][group_name] = config
        save_psn_groups(st.session_state["psn_feature_groups"])
        st.success(f"Saved '{group_name}'")
        st.rerun()

st.divider()

# ---------------------------------------------------------------------
# Saved Feature Groups
# ---------------------------------------------------------------------
st.subheader("Saved Feature Groups")

psn_groups = st.session_state.get("psn_feature_groups", {})

if not psn_groups:
    st.info("No PSN feature groups saved yet.")
else:
    # Display as a table
    rows = []
    for gname, cfg in psn_groups.items():
        profile = ", ".join(cfg.get("selected_cols", []))
        zip_feats_cfg = cfg.get("neighborhood_features", {})
        active_zip = [k.replace("_index", "").replace("zip_", "") for k, v in zip_feats_cfg.items() if v]
        wb = cfg.get("weight_balance", 0.3)

        # Get graph construction settings (with defaults for backward compatibility)
        encoding = "Exp" if cfg.get("experimental_encoding", False) else "Std"
        sim_metric = cfg.get("similarity_metric", "cosine")
        k_val = cfg.get("k", 5)
        knn = cfg.get("knn_type", "mutual")
        ann = cfg.get("ann_mode", "auto")

        rows.append({
            "Name": gname,
            "Profile Columns": profile[:40] + "..." if len(profile) > 40 else profile,
            "Neighborhood": ", ".join(active_zip) if active_zip else "-",
            "Weight": f"P:{1-wb:.0%} N:{wb:.0%}",
            "Encoding": encoding,
            "Similarity": sim_metric[:3],  # cos or mix
            "k": k_val,
            "Type": knn[:3],  # mut or dir
            "ANN": ann[:4],  # auto, forc (for force_ann/force_exact)
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Delete controls
    delete_col1, delete_col2 = st.columns([2, 1])
    with delete_col1:
        delete_selection = st.selectbox("Select to delete", [""] + list(psn_groups.keys()))
    with delete_col2:
        if st.button("Delete", disabled=not delete_selection):
            if delete_selection and delete_selection in st.session_state["psn_feature_groups"]:
                del st.session_state["psn_feature_groups"][delete_selection]
                save_psn_groups(st.session_state["psn_feature_groups"])
                st.success(f"Deleted '{delete_selection}'")
                st.rerun()
