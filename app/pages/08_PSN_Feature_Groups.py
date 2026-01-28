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
pat = st.session_state.get("patients_df")
zip_feats = st.session_state.get("zip_features")

if pat is None:
    st.error("patients_df not found. Load on Home.")
    st.stop()

# ---------------------------------------------------------------------
# Available columns
# ---------------------------------------------------------------------
PAT_GROUPS = {
    "demographics": ["SEX", "Race", "AGE_BIN"],
    "utilization": ["LENSTAYD_BIN", "PAYER"],
    "risk_binaries": ["Hearingloss", "BrainInjury", "Hypertension", "Alcohol", "Obesity", "Diabetes"],
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

# Gather selected columns
selected_demo = [c for c, v in demo_checks.items() if v]
selected_util = [c for c, v in util_checks.items() if v]
selected_risk = [c for c, v in risk_checks.items() if v]
selected_cols = selected_demo + selected_util + selected_risk

# Neighborhood features (compact row)
if zip_available:
    st.caption("Neighborhood Features")
    zcol1, zcol2, zcol3, zcol4, zcol5, zcol6, zcol7 = st.columns(7)
    with zcol1:
        use_env = st.checkbox("environment_index", key="new_env")
    with zcol2:
        use_ses = st.checkbox("ses_index", key="new_ses")
    with zcol3:
        use_degree = st.checkbox("zip_degree", key="new_deg")
    with zcol4:
        use_pr = st.checkbox("zip_pagerank", key="new_pr")
    with zcol5:
        use_btw = st.checkbox("zip_betweenness", key="new_btw")
    with zcol6:
        onehot_comm = st.checkbox("zip_community", key="new_comm", help="One-hot encode (split path only)")
    with zcol7:
        split_by_zip = st.checkbox("Split by ZIP", key="new_split")
else:
    use_env = use_ses = use_degree = use_pr = use_btw = onehot_comm = split_by_zip = False
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
                "environment_index": use_env,
                "ses_index": use_ses,
                "zip_degree": use_degree,
                "zip_pagerank": use_pr,
                "zip_betweenness": use_btw,
                "onehot_zip_community": onehot_comm,
            },
            "split_by_zip": split_by_zip,
            "weight_balance": weight_balance,
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
        rows.append({
            "Name": gname,
            "Profile Columns": profile[:40] + "..." if len(profile) > 40 else profile,
            "Neighborhood": ", ".join(active_zip) if active_zip else "-",
            "Weight": f"P:{1-wb:.0%} N:{wb:.0%}",
            "Split": "Y" if cfg.get("split_by_zip") else "N",
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
