# app/pages/03_Patient_Feature_Fusion.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import hashlib

st.set_page_config(page_title="Patient Feature Fusion (Profiles)", layout="wide")
st.title("03 - Patient Feature Fusion (Profiles)")

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

def default_selected_columns(df: pd.DataFrame) -> list:
    """Demographics + risk + PAYER if present (order preserved)."""
    picks = []
    picks += [c for c in PAT_GROUPS["demographics"] if c in df.columns]
    picks += [c for c in PAT_GROUPS["risk_binaries"] if c in df.columns]
    if "PAYER" in df.columns:
        picks.append("PAYER")
    seen, ordered = set(), []
    for c in picks:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered

# ---------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Controls")

    # --- Profile columns ---
    st.subheader("Profile columns")
    colg1, colg2, colg3 = st.columns(3)
    with colg1:
        inc_demo = st.checkbox("Demographics", value=True)
    with colg2:
        inc_util = st.checkbox("Utilization", value=False)
    with colg3:
        inc_risk = st.checkbox("Risk binaries", value=True)

    base_cols = []
    if inc_demo:
        base_cols += [c for c in PAT_GROUPS["demographics"] if c in pat.columns]
    if inc_util:
        base_cols += [c for c in PAT_GROUPS["utilization"] if c in pat.columns]
    if inc_risk:
        base_cols += [c for c in PAT_GROUPS["risk_binaries"] if c in pat.columns]

    allowed_cols = (
        [c for c in PAT_GROUPS["location"] if c in pat.columns] +
        [c for c in PAT_GROUPS["demographics"] if c in pat.columns] +
        [c for c in PAT_GROUPS["utilization"] if c in pat.columns] +
        [c for c in PAT_GROUPS["risk_binaries"] if c in pat.columns]
    )
    default_cols = st.session_state.get("pf_default_cols", default_selected_columns(pat))
    merged_defaults = list(dict.fromkeys(default_cols + base_cols))

    selected_cols = st.multiselect(
        "Selected columns for profiling (categorical/binned only)",
        options=sorted(allowed_cols),
        default=sorted(set(merged_defaults)) or base_cols,
        help="Use only categorical/binned columns defined in PAT_GROUPS. Include ZIPCODE to split profiles geographically.",
    )
    st.session_state["pf_default_cols"] = selected_cols

    # --- ZIP context features ---
    st.subheader("ZIP context features")
    use_env = st.checkbox("environment_index", value=True)
    use_ses = st.checkbox("ses_index", value=True)
    use_pr  = st.checkbox("zip_pagerank", value=False)
    use_btw = st.checkbox("zip_betweenness", value=False)
    onehot_comm = st.checkbox("one-hot zip_community (split path only)", value=False)

    # --- ZIP handling if ZIP not selected ---
    st.subheader("ZIP handling for profiles")
    zip_strategy = st.radio(
        "If ZIPCODE is not among profile columns:",
        options=["Split profiles by ZIP (profile × ZIP)", "Aggregate ZIP features across patients"],
        index=0,
    )

    # --- Actions ---
    colb1, colb2 = st.columns(2)
    with colb1:
        generate_clicked = st.button("Generate Fused Summary")   # runs Steps 2 + 3 (no weights/graph)
    with colb2:
        save_settings_clicked = st.button("Save Fused Settings")  # stores config only

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
# Save settings (config only; no computation)
# ---------------------------------------------------------------------
if save_settings_clicked:
    st.session_state["pf_controls_saved"] = {
        "selected_cols": selected_cols,
        "zip_features": {
            "environment_index": use_env,
            "ses_index": use_ses,
            "zip_pagerank": use_pr,
            "zip_betweenness": use_btw,
            "onehot_zip_community": onehot_comm,
        },
        "zip_strategy": zip_strategy,
    }
    st.success("Settings saved to session_state['pf_controls_saved'].")

# ---------------------------------------------------------------------
# Generate Fused Summary (Steps 2 + 3; no graph/weights)
# ---------------------------------------------------------------------
if generate_clicked:
    # Traceability for this run
    st.session_state["pf_controls_run"] = {
        "selected_cols": selected_cols,
        "zip_features": {
            "environment_index": use_env,
            "ses_index": use_ses,
            "zip_pagerank": use_pr,
            "zip_betweenness": use_btw,
            "onehot_zip_community": onehot_comm,
        },
        "zip_strategy": zip_strategy,
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
        if zip_strategy.startswith("Split profiles"):
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
        "zip_strategy": zip_strategy,
        "n_profiles_base": int(len(base_grp)),
        "n_profiles_by_zip": int(len(profiles_by_zip)) if profiles_by_zip is not None else 0,
        "has_zip_counts": bool(zip_counts is not None),
    }

    st.success("Profiles constructed.")

    # ===================== STEP 3: ZIP join + fused encoding =====================
    zip_feats = st.session_state.get("zip_features")
    if zip_feats is None:
        st.warning("ZIP features not found. Go to page 02 and click 'Set as ZIP features'.")
    else:
        use_split = (("ZIPCODE" in selected_cols) or zip_strategy.startswith("Split profiles"))

        if use_split:
            prof_tbl = st.session_state.get("pf_profiles_by_zip")
            if prof_tbl is None or prof_tbl.empty:
                prof_tbl = base_grp

            key_id_col = "profile_zip_id" if ("profile_zip_id" in prof_tbl.columns) else "profile_id"
            join_key = "ZIPCODE" if ("ZIPCODE" in prof_tbl.columns) else None
            if join_key is None:
                st.error("Split path selected but no ZIPCODE column found in profiles. "
                        "Include ZIPCODE in selected columns or choose the aggregate path.")
                st.stop()
            fused_tbl = _safe_merge(prof_tbl, zip_feats, on="ZIPCODE", how="left")

        else:
            base = base_grp
            zc = st.session_state.get("pf_zip_counts")
            if zc is None:
                st.error("Aggregate ZIP path requires pf_zip_counts. Click Generate Fused Summary again.")
                st.stop()

            zc2 = _safe_merge(zc, zip_feats, on="ZIPCODE", how="left")

            # Numeric ZIP features to weighted-average by patient counts per ZIP within profile
            zip_num_cols = []
            if use_env: zip_num_cols.append("environment_index")
            if use_ses: zip_num_cols.append("ses_index")
            if use_pr  and "zip_pagerank"    in zc2.columns: zip_num_cols.append("zip_pagerank")
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
        if use_pr  and "zip_pagerank" in fused_tbl.columns: zip_num_cols2.append("zip_pagerank")
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

        st.success(f"Fused summary built: {X_fused.shape[0]} rows × {X_fused.shape[1]} cols.")

# ---------------------------------------------------------------------
# Previews
# ---------------------------------------------------------------------
st.subheader("Current settings (saved / last run)")
if "pf_controls_saved" in st.session_state:
    st.caption("Saved settings:")
    st.json(st.session_state["pf_controls_saved"])
if "pf_controls_run" in st.session_state:
    st.caption("Last run settings:")
    st.json(st.session_state["pf_controls_run"])
if "pf_controls_saved" not in st.session_state and "pf_controls_run" not in st.session_state:
    st.info("Adjust controls in the sidebar, then click 'Generate Fused Summary' or 'Save Fused Settings'.")

st.subheader("Profiles (base) — preview")
if "pf_profiles_base" in st.session_state:
    st.dataframe(st.session_state["pf_profiles_base"].head(12), width="content")
    meta = st.session_state.get("pf_profiles_meta", {})
    st.caption(f"Total base profiles: {meta.get('n_profiles_base', 'NA')}")

if st.session_state.get("pf_profiles_by_zip") is not None:
    st.subheader("Profiles × ZIP (split path) — preview")
    st.dataframe(st.session_state["pf_profiles_by_zip"].head(12), width="content")
    meta = st.session_state.get("pf_profiles_meta", {})
    st.caption(f"Total profiles×ZIP: {meta.get('n_profiles_by_zip', 'NA')}")
elif st.session_state.get("pf_zip_counts") is not None:
    st.subheader("ZIP distribution per base profile (aggregate path) — preview")
    st.dataframe(st.session_state["pf_zip_counts"].head(12), width="content")
    st.caption("Used to compute weighted ZIP features per profile.")

st.subheader("Fused feature matrix — summary")
meta = st.session_state.get("pf_fused_meta", {})
if meta:
    st.write(meta)
    st.write({
        "patient_block_cols": len(st.session_state.get("pf_patient_block_cols", [])),
        "zip_block_cols": len(st.session_state.get("pf_zip_block_cols", [])),
    })

st.subheader("Fused table (head)")
ft = st.session_state.get("pf_fused_table")
if ft is not None:
    st.dataframe(ft.head(12), width="content")