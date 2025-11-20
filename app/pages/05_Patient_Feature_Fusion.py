# app/pages/03_Patient_Feature_Fusion.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import hashlib

st.set_page_config(page_title="Patient Feature Fusion (Profiles)", layout="wide")
st.title("Patient Feature Fusion (Profiles)")

# ---------------------------------------------------------------------
# Data checks
# ---------------------------------------------------------------------
pat = st.session_state.get("patients_df")
zip_feats_initial = st.session_state.get("zip_features")  # from 02_ZIP_Context
if pat is None:
    st.error("patients_df not found in session_state. Load on Home.")
    st.stop()
if zip_feats_initial is None:
    st.warning("zip_features not found in session_state. Run page 02 (ZIP Context) and click 'Set as ZIP features'.")

# ---------------------------------------------------------------------
# Groupings (binned/cleaned only)
# ---------------------------------------------------------------------
PAT_GROUPS = {
    "ids": ["SEQ_NO", "REC_KEY"],
    "location": ["ZIPCODE"],
    "demographics": ["SEX", "Race", "AGE_BIN"],
    "utilization": ["LENSTAYD_BIN", "PAYER"],
    "risk_binaries": ["Hearingloss", "BrainInjury", "Hypertension", "Alcohol", "Obesity", "Diabetes"],
}

# ---------------------------------------------------------------------
# Helpers
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

def _one_hot(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
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

# ---------------------------------------------------------------------
# Initialize session state defaults
# ---------------------------------------------------------------------
if "pf_selected_demo" not in st.session_state:
    st.session_state["pf_selected_demo"] = []
if "pf_selected_util" not in st.session_state:
    st.session_state["pf_selected_util"] = []
if "pf_selected_risk" not in st.session_state:
    st.session_state["pf_selected_risk"] = []
if "pf_use_env" not in st.session_state:
    st.session_state["pf_use_env"] = True
if "pf_use_ses" not in st.session_state:
    st.session_state["pf_use_ses"] = True
if "pf_use_degree" not in st.session_state:
    st.session_state["pf_use_degree"] = False
if "pf_use_pr" not in st.session_state:
    st.session_state["pf_use_pr"] = False
if "pf_use_btw" not in st.session_state:
    st.session_state["pf_use_btw"] = False
if "pf_onehot_comm" not in st.session_state:
    st.session_state["pf_onehot_comm"] = False
if "pf_split_by_zip" not in st.session_state:
    st.session_state["pf_split_by_zip"] = False

# ---------------------------------------------------------------------
# Main page controls
# ---------------------------------------------------------------------
st.header("Configuration")

# --- 1. Profile Columns Section ---
st.subheader("1. Profile Columns")
st.caption("Select columns for patient profiling (categorical/binned only)")

# Get available columns for each group
demo_cols = [c for c in PAT_GROUPS["demographics"] if c in pat.columns]
util_cols = [c for c in PAT_GROUPS["utilization"] if c in pat.columns]
risk_cols = [c for c in PAT_GROUPS["risk_binaries"] if c in pat.columns]

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Demographics**")
    selected_demo = []
    for col in demo_cols:
        checked = st.checkbox(col, key=f"demo_{col}", value=(col in st.session_state["pf_selected_demo"]))
        if checked:
            selected_demo.append(col)

with col2:
    st.markdown("**Utilization**")
    selected_util = []
    for col in util_cols:
        checked = st.checkbox(col, key=f"util_{col}", value=(col in st.session_state["pf_selected_util"]))
        if checked:
            selected_util.append(col)

with col3:
    st.markdown("**Risk Binaries**")
    selected_risk = []
    for col in risk_cols:
        checked = st.checkbox(col, key=f"risk_{col}", value=(col in st.session_state["pf_selected_risk"]))
        if checked:
            selected_risk.append(col)

# Update session state with current selections
st.session_state["pf_selected_demo"] = selected_demo
st.session_state["pf_selected_util"] = selected_util
st.session_state["pf_selected_risk"] = selected_risk

# Combine all selected columns
selected_cols = selected_demo + selected_util + selected_risk

st.divider()

# --- 2. ZIP Context Features Section ---
st.subheader("2. ZIP Context Features")
st.caption("Select ZIP-level features to include in fusion")

zip_col1, zip_col2, zip_col3 = st.columns(3)

with zip_col1:
    use_env = st.checkbox("environment_index", value=st.session_state["pf_use_env"], key="zip_env")
    use_ses = st.checkbox("ses_index", value=st.session_state["pf_use_ses"], key="zip_ses")

with zip_col2:
    use_degree = st.checkbox("zip_degree", value=st.session_state["pf_use_degree"], key="zip_degree")
    use_pr = st.checkbox("zip_pagerank", value=st.session_state["pf_use_pr"], key="zip_pr")

with zip_col3:
    use_btw = st.checkbox("zip_betweenness", value=st.session_state["pf_use_btw"], key="zip_btw")
    onehot_comm = st.checkbox("one-hot zip_community (split path only)", value=st.session_state["pf_onehot_comm"], key="zip_onehot")

# Update session state
st.session_state["pf_use_env"] = use_env
st.session_state["pf_use_ses"] = use_ses
st.session_state["pf_use_degree"] = use_degree
st.session_state["pf_use_pr"] = use_pr
st.session_state["pf_use_btw"] = use_btw
st.session_state["pf_onehot_comm"] = onehot_comm

st.divider()

# --- 3. ZIP Handling Section ---
st.subheader("3. ZIP Handling")
st.caption("Choose how to handle ZIP features when ZIPCODE is not in profile columns")

split_by_zip = st.toggle("Split profiles by ZIP", value=st.session_state["pf_split_by_zip"], key="zip_split",
                         help="ON: Create separate profiles for each ZIP (profile × ZIP). OFF: Aggregate ZIP features across patients.")
st.session_state["pf_split_by_zip"] = split_by_zip

st.divider()

# --- 4. Action Button ---
generate_clicked = st.button("Generate Fused Data", type="primary", use_container_width=False)

# ---------------------------------------------------------------------
# Generate Fused Data (combines computation + save settings)
# ---------------------------------------------------------------------
if generate_clicked:
    # Save settings first
    st.session_state["pf_controls_saved"] = {
        "selected_cols": selected_cols,
        "zip_features": {
            "environment_index": use_env,
            "ses_index": use_ses,
            "zip_degree": use_degree,
            "zip_pagerank": use_pr,
            "zip_betweenness": use_btw,
            "onehot_zip_community": onehot_comm,
        },
        "split_by_zip": split_by_zip,
    }
    # Traceability for this run
    st.session_state["pf_controls_run"] = {
        "selected_cols": selected_cols,
        "zip_features": {
            "environment_index": use_env,
            "ses_index": use_ses,
            "zip_degree": use_degree,
            "zip_pagerank": use_pr,
            "zip_betweenness": use_btw,
            "onehot_zip_community": onehot_comm,
        },
        "split_by_zip": split_by_zip,
    }

    # ===================== STEP 2: profile construction =====================
    if not selected_cols:
        st.error("No columns selected for profiling.")
        st.stop()

    work_cols = list(dict.fromkeys(selected_cols + (["ZIPCODE"] if "ZIPCODE" in pat.columns else [])))
    dfw = pat[work_cols].copy()

    for c in selected_cols:
        if c in dfw.columns:
            dfw[c] = _as_categorical_str(dfw[c])
    if "ZIPCODE" in dfw.columns:
        dfw["ZIPCODE"] = _as_categorical_str(dfw["ZIPCODE"]).str.replace(" ", "", regex=False)

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

    st.session_state["pf_profiles_base"] = base_grp
    st.session_state["pf_profiles_by_zip"] = profiles_by_zip
    st.session_state["pf_zip_counts"] = zip_counts
    st.session_state["pf_profiles_meta"] = {
        "selected_cols": selected_cols,
        "zip_in_selected": ("ZIPCODE" in selected_cols),
        "split_by_zip": split_by_zip,
        "n_profiles_base": int(len(base_grp)),
        "n_profiles_by_zip": int(len(profiles_by_zip)) if profiles_by_zip is not None else 0,
        "has_zip_counts": bool(zip_counts is not None),
    }

    st.success("✓ Profiles constructed.")

    # ===================== STEP 3: ZIP join + fused encoding =====================
    zip_feats = st.session_state.get("zip_features")
    if zip_feats is None:
        st.warning("ZIP features not found. Go to page 02 and click 'Set as ZIP features'.")
    else:
        use_split = (("ZIPCODE" in selected_cols) or split_by_zip)

        if use_split:
            prof_tbl = st.session_state.get("pf_profiles_by_zip")
            if prof_tbl is None or prof_tbl.empty:
                prof_tbl = base_grp

            key_id_col = "profile_zip_id" if ("profile_zip_id" in prof_tbl.columns) else "profile_id"
            join_key = "ZIPCODE" if ("ZIPCODE" in prof_tbl.columns) else None
            if join_key is None:
                st.error("Split path selected but no ZIPCODE column found in profiles. "
                        "Include ZIPCODE in selected columns or turn off split toggle.")
                st.stop()
            fused_tbl = _safe_merge(prof_tbl, zip_feats, on="ZIPCODE", how="left")

        else:
            base = base_grp
            zc = st.session_state.get("pf_zip_counts")
            if zc is None:
                st.error("Aggregate ZIP path requires pf_zip_counts. Click Generate Fused Data again.")
                st.stop()

            zc2 = _safe_merge(zc, zip_feats, on="ZIPCODE", how="left")

            # Numeric ZIP features to weighted-average by patient counts per ZIP within profile
            zip_num_cols = []
            if use_env: zip_num_cols.append("environment_index")
            if use_ses: zip_num_cols.append("ses_index")
            if use_degree and "zip_degree" in zc2.columns: zip_num_cols.append("zip_degree")
            if use_pr and "zip_pagerank" in zc2.columns: zip_num_cols.append("zip_pagerank")
            if use_btw and "zip_betweenness" in zc2.columns: zip_num_cols.append("zip_betweenness")

            if zip_num_cols:
                zc2["n"] = zc2["n"].astype(float)
                grp = zc2.groupby(selected_cols, dropna=False)
                num_wavg = (grp.apply(lambda g: pd.Series(
                    {col: np.average(g[col].fillna(0.0), weights=g["n"]) for col in zip_num_cols}
                )).reset_index())
            else:
                num_wavg = base[selected_cols].copy()

            fused_tbl = base.merge(num_wavg, on=selected_cols, how="left")
            key_id_col = "profile_id"
            join_key = None

        # Patient block (categoricals/binaries)
        RISK = {"Hearingloss", "BrainInjury", "Hypertension", "Alcohol", "Obesity", "Diabetes"}
        bin_cols = [c for c in selected_cols if c in RISK and c in fused_tbl.columns]
        cat_cols = [c for c in selected_cols if c not in bin_cols]

        X_cat = _one_hot(fused_tbl, cat_cols)
        X_bin = pd.DataFrame(index=fused_tbl.index)
        for c in bin_cols:
            X_bin[c] = pd.to_numeric(fused_tbl[c], errors="coerce").fillna(0.0).clip(0, 1).astype(float)
        patient_block = pd.concat([X_cat, X_bin], axis=1)

        # ZIP block (numeric std + optional one-hot community on split path)
        zip_num_cols2 = []
        if use_env and "environment_index" in fused_tbl.columns: zip_num_cols2.append("environment_index")
        if use_ses and "ses_index" in fused_tbl.columns: zip_num_cols2.append("ses_index")
        if use_degree and "zip_degree" in fused_tbl.columns: zip_num_cols2.append("zip_degree")
        if use_pr and "zip_pagerank" in fused_tbl.columns: zip_num_cols2.append("zip_pagerank")
        if use_btw and "zip_betweenness" in fused_tbl.columns: zip_num_cols2.append("zip_betweenness")

        zip_num_df = (
            fused_tbl[zip_num_cols2].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            if zip_num_cols2 else pd.DataFrame(index=fused_tbl.index)
        )
        zip_num_std = _standardize(zip_num_df) if not zip_num_df.empty else zip_num_df

        zip_onehot_df = pd.DataFrame(index=fused_tbl.index)
        if onehot_comm and use_split and "zip_community" in fused_tbl.columns:
            zip_onehot_df = _one_hot(fused_tbl, ["zip_community"])

        zip_block = pd.concat([zip_num_std, zip_onehot_df], axis=1)

        # Final fused (no block weights on this page)
        X_fused = pd.concat([patient_block, zip_block], axis=1).fillna(0.0)

        # Persist artifacts
        st.session_state["pf_fused_table"]   = fused_tbl
        st.session_state["pf_patient_block_cols"] = list(patient_block.columns)
        st.session_state["pf_zip_block_cols"]     = list(zip_block.columns)
        st.session_state["pf_fused_matrix"]  = X_fused.values
        st.session_state["pf_fused_index"]   = fused_tbl.get(key_id_col, pd.Series(range(len(fused_tbl)))).tolist()
        st.session_state["pf_fused_counts"]  = fused_tbl.get("profile_count", pd.Series([np.nan]*len(fused_tbl))).tolist()
        st.session_state["pf_fused_meta"]    = {
            "rows": int(X_fused.shape[0]),
            "cols_total": int(X_fused.shape[1]),
            "cols_patient": int(patient_block.shape[1]),
            "cols_zip": int(zip_block.shape[1]),
            "key_id_col": key_id_col,
            "use_split": use_split,
            "join_key": join_key,
        }

        st.success(f"✓ Fused data generated and saved: {X_fused.shape[0]} rows × {X_fused.shape[1]} cols.")

# ---------------------------------------------------------------------
# Results Display (only shown after Generate is clicked)
# ---------------------------------------------------------------------
if generate_clicked or "pf_profiles_base" in st.session_state:
    st.divider()
    st.header("Fusion Results")
    
    # --- Current settings ---
    st.subheader("Current Settings")
    col_set1, col_set2 = st.columns(2)
    
    with col_set1:
        if "pf_controls_saved" in st.session_state:
            st.caption("**Saved Settings**")
            saved = st.session_state["pf_controls_saved"]
            saved_df = pd.DataFrame([
                {"Parameter": "Selected Columns", "Value": ", ".join(saved["selected_cols"]) if saved["selected_cols"] else "None"},
                {"Parameter": "ZIP Features", "Value": ", ".join([k for k, v in saved["zip_features"].items() if v])},
                {"Parameter": "Split by ZIP", "Value": "Yes" if saved["split_by_zip"] else "No"},
            ])
            st.dataframe(saved_df, width='stretch', hide_index=True)
    
    with col_set2:
        if "pf_controls_run" in st.session_state:
            st.caption("**Last Run Settings**")
            run = st.session_state["pf_controls_run"]
            run_df = pd.DataFrame([
                {"Parameter": "Selected Columns", "Value": ", ".join(run["selected_cols"]) if run["selected_cols"] else "None"},
                {"Parameter": "ZIP Features", "Value": ", ".join([k for k, v in run["zip_features"].items() if v])},
                {"Parameter": "Split by ZIP", "Value": "Yes" if run["split_by_zip"] else "No"},
            ])
            st.dataframe(run_df, width='stretch', hide_index=True)
    
    if "pf_controls_saved" not in st.session_state and "pf_controls_run" not in st.session_state:
        st.info("Configure settings above and click 'Generate Fused Data' or 'Save Fused Settings'.")
    
    # --- Fused feature matrix summary ---
    st.subheader("Fused Feature Matrix — Summary")
    meta = st.session_state.get("pf_fused_meta", {})
    if meta:
        summary_df = pd.DataFrame([
            {"Metric": "Total Rows", "Value": meta.get("rows", "N/A")},
            {"Metric": "Total Columns", "Value": meta.get("cols_total", "N/A")},
            {"Metric": "Patient Block Columns", "Value": meta.get("cols_patient", "N/A")},
            {"Metric": "ZIP Block Columns", "Value": meta.get("cols_zip", "N/A")},
            # {"Metric": "Key ID Column", "Value": meta.get("key_id_col", "N/A")},
            {"Metric": "Use Split", "Value": "Yes" if meta.get("use_split") else "No"},
            # {"Metric": "Join Key", "Value": meta.get("join_key", "N/A")},
        ])
        st.dataframe(summary_df, width='content', hide_index=True)
        
        # Block details
        col_block1, col_block2 = st.columns(2)
        with col_block1:
            st.caption("**Patient Block Columns**")
            patient_cols = st.session_state.get("pf_patient_block_cols", [])
            if patient_cols:
                st.text(f"Count: {len(patient_cols)}")
                st.caption(", ".join(patient_cols[:10]) + ("..." if len(patient_cols) > 10 else ""))
        
        with col_block2:
            st.caption("**ZIP Block Columns**")
            zip_cols = st.session_state.get("pf_zip_block_cols", [])
            if zip_cols:
                st.text(f"Count: {len(zip_cols)}")
                st.caption(", ".join(zip_cols))
    
    # --- Fused table preview ---
    st.subheader("Fused Table — Preview")
    ft = st.session_state.get("pf_fused_table")
    if ft is not None:
        st.dataframe(ft.head(12), width='content', hide_index=True)
