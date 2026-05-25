# app/pages/05_PSN_Feature_Selection.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import hashlib

st.set_page_config(page_title="PSN Feature Selection", layout="wide")
st.title("PSN Feature Selection")

# ---------------------------------------------------------------------
# Data checks
# ---------------------------------------------------------------------
from app._logic.loader import ensure_data_loaded

if not ensure_data_loaded():
    st.stop()

pat = st.session_state.get("patients_df")
zip_feats_initial = st.session_state.get("zip_features")  # from 02_ZIP_Context
if zip_feats_initial is None:
    st.warning("Neighborhood features not found. Run page 02 (Neighborhood Graph) first.")

# ---------------------------------------------------------------------
# Groupings (binned/cleaned only)
# ---------------------------------------------------------------------
PAT_GROUPS = {
    "ids": ["PATIENTID"],
    "location": ["ZIPCODE"],
    "demographics": ["SEX", "Race", "AGE_BIN"],
    "utilization": ["LENSTAYD_BIN", "LENSTAYD_LOG", "PAYER", "NUM_VISITS", "REVISIT_30"],
    "risk_binaries": ["Hearingloss", "BrainInjury", "Hypertension", "Alcohol", "Obesity", "Diabetes"],
    "outcomes": ["READMIT_COUNT", "READMIT_RATE", "EVER_READMITTED"],
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
# Import shared encoding utilities
# ---------------------------------------------------------------------
from app._logic.encoding import (
    integer_encode_categoricals,
    bitflag_encode_multibinary,
    format_encoded_display,
)

# ---------------------------------------------------------------------
# Initialize session state defaults
# ---------------------------------------------------------------------
if "pf_selected_demo" not in st.session_state:
    st.session_state["pf_selected_demo"] = []
if "pf_selected_util" not in st.session_state:
    st.session_state["pf_selected_util"] = []
if "pf_selected_risk" not in st.session_state:
    st.session_state["pf_selected_risk"] = []
if "pf_selected_outcomes" not in st.session_state:
    st.session_state["pf_selected_outcomes"] = []
if "pf_use_degree" not in st.session_state:
    st.session_state["pf_use_degree"] = False
if "pf_use_pr" not in st.session_state:
    st.session_state["pf_use_pr"] = False
if "pf_use_btw" not in st.session_state:
    st.session_state["pf_use_btw"] = False
if "pf_use_zip_comm" not in st.session_state:
    st.session_state["pf_use_zip_comm"] = False
if "pf_experimental_encoding" not in st.session_state:
    st.session_state["pf_experimental_encoding"] = True

# ---------------------------------------------------------------------
# Main page controls
# ---------------------------------------------------------------------
st.header("Configuration")

# --- 1. Profile Columns Section ---
st.subheader("1. Profile Columns")
st.caption("Select columns for patient profiling (categorical/binned only)")

# Get available columns for each group
demo_cols    = [c for c in PAT_GROUPS["demographics"]  if c in pat.columns]
util_cols    = [c for c in PAT_GROUPS["utilization"]   if c in pat.columns]
risk_cols    = [c for c in PAT_GROUPS["risk_binaries"] if c in pat.columns]
outcome_cols = [c for c in PAT_GROUPS["outcomes"]      if c in pat.columns]

col1, col2, col3, col4 = st.columns(4)

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

with col4:
    st.markdown("**Outcomes**")
    selected_outcomes = []
    for col in outcome_cols:
        checked = st.checkbox(col, key=f"out_{col}", value=(col in st.session_state["pf_selected_outcomes"]))
        if checked:
            selected_outcomes.append(col)

# Update session state with current selections
st.session_state["pf_selected_demo"]     = selected_demo
st.session_state["pf_selected_util"]     = selected_util
st.session_state["pf_selected_risk"]     = selected_risk
st.session_state["pf_selected_outcomes"] = selected_outcomes

# Combine all selected columns
selected_cols = selected_demo + selected_util + selected_risk + selected_outcomes

st.divider()

# --- 2. Neighborhood Features Section ---
st.subheader("2. Neighborhood Features")
st.caption("Select neighborhood-level features to include")

zip_col1, zip_col2 = st.columns(2)

with zip_col1:
    use_degree = st.checkbox("zip_degree", value=st.session_state["pf_use_degree"], key="zip_degree")
    use_pr = st.checkbox("zip_pagerank", value=st.session_state["pf_use_pr"], key="zip_pr")

with zip_col2:
    use_btw = st.checkbox("zip_betweenness", value=st.session_state["pf_use_btw"], key="zip_btw")
    use_zip_comm = st.checkbox("zip_community", value=st.session_state["pf_use_zip_comm"], key="zip_comm")

# Update session state
st.session_state["pf_use_degree"] = use_degree
st.session_state["pf_use_pr"] = use_pr
st.session_state["pf_use_btw"] = use_btw
st.session_state["pf_use_zip_comm"] = use_zip_comm

st.divider()

# --- 3. Encoding Options ---
st.subheader("3. Encoding Method")
st.caption("Default: Compact encoding (integer + bitflag). Expand to use standard one-hot encoding.")

with st.expander("⚙️ Change Encoding Method"):
    experimental_encoding = st.toggle(
        "Use compact encoding (recommended)",
        value=st.session_state["pf_experimental_encoding"],
        key="experimental_toggle",
        help=(
            "Compact (default): Integer-encode categoricals + bitflag-encode comorbidities "
            "(fewer columns, faster similarity). "
            "Standard (unchecked): One-hot encode all categoricals (more columns)."
        )
    )
    st.session_state["pf_experimental_encoding"] = experimental_encoding

    if experimental_encoding:
        st.info("✓ Using compact encoding: Integer + bitflag")
    else:
        st.warning("Using standard one-hot encoding (more features)")

if experimental_encoding:
    st.info(
        "**Experimental mode:** Categoricals (Sex, Race, Age, Payer) will be integer-encoded. "
        "Comorbidities (Hearingloss, BrainInjury, etc.) will be combined into a single bitflag column. "
        "zip_community (if selected) will be integer-encoded in the neighborhood block. "
        "This reduces dimensionality and enables custom similarity metrics on page 06."
    )

st.divider()

# --- 5. Action Button ---
generate_clicked = st.button("Generate PSN Features", type="primary", use_container_width=False)

# ---------------------------------------------------------------------
# Generate PSN Features (combines computation + save settings)
# ---------------------------------------------------------------------
if generate_clicked:
    # Store run settings
    st.session_state["pf_controls_run"] = {
        "selected_cols": selected_cols,
        "zip_features": {
            "zip_degree": use_degree,
            "zip_pagerank": use_pr,
            "zip_betweenness": use_btw,
            "zip_community": use_zip_comm,
        },
        "experimental_encoding": experimental_encoding,
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

    # Always compute ZIP counts for weighted averaging (if ZIPCODE available)
    zip_counts = None
    if "ZIPCODE" not in selected_cols and "ZIPCODE" in dfw.columns:
        zip_counts = (
            dfw.groupby(selected_cols + ["ZIPCODE"], dropna=False)
               .size().reset_index(name="n")
        )

    st.session_state["pf_profiles_base"] = base_grp
    st.session_state["pf_zip_counts"] = zip_counts
    st.session_state["pf_profiles_meta"] = {
        "selected_cols": selected_cols,
        "zip_in_selected": ("ZIPCODE" in selected_cols),
        "n_profiles_base": int(len(base_grp)),
        "has_zip_counts": bool(zip_counts is not None),
    }

    st.success("✓ Profiles constructed.")

    # ===================== STEP 3: ZIP join + fused encoding =====================
    zip_feats = st.session_state.get("zip_features")
    if zip_feats is None:
        st.warning("Neighborhood features not found. Go to page 02 (Neighborhood Graph) first.")
    else:
        # Always use aggregate approach with weighted averaging
        base = base_grp
        zc = st.session_state.get("pf_zip_counts")
        if zc is None:
            st.error("Aggregate ZIP path requires pf_zip_counts. Reload the page and try again.")
            st.stop()

        zc2 = _safe_merge(zc, zip_feats, on="ZIPCODE", how="left")

        # Determine grouping columns: add zip_community if selected
        grouping_cols = selected_cols.copy()
        if use_zip_comm and "zip_community" in zc2.columns:
            grouping_cols = grouping_cols + ["zip_community"]

        # Numeric ZIP features to weighted-average by patient counts per ZIP within profile
        zip_num_cols = []
        if use_degree and "zip_degree" in zc2.columns: zip_num_cols.append("zip_degree")
        if use_pr and "zip_pagerank" in zc2.columns: zip_num_cols.append("zip_pagerank")
        if use_btw and "zip_betweenness" in zc2.columns: zip_num_cols.append("zip_betweenness")

        if zip_num_cols or use_zip_comm:
            zc2["n"] = zc2["n"].astype(float)
            grp = zc2.groupby(grouping_cols, dropna=False)

            # Weighted-average numeric ZIP features
            if zip_num_cols:
                num_wavg = (grp.apply(lambda g: pd.Series(
                    {col: np.average(g[col].fillna(0.0), weights=g["n"]) for col in zip_num_cols}
                )).reset_index())
            else:
                # Only zip_community, no numeric features to average
                num_wavg = pd.DataFrame({col: zc2[col] for col in grouping_cols}).drop_duplicates()
        else:
            num_wavg = base[selected_cols].copy()

        fused_tbl = base.merge(num_wavg, on=selected_cols, how="left")

        # When zip_community splits profiles, recompute profile_count at the
        # finer grouping level (selected_cols + zip_community) so each row
        # reflects the patient count for that specific community slice, not
        # the whole pre-split profile.
        if use_zip_comm and "zip_community" in fused_tbl.columns:
            comm_counts = (
                zc2.groupby(grouping_cols, dropna=False)["n"]
                .sum()
                .reset_index(name="profile_count")
            )
            fused_tbl = fused_tbl.drop(columns=["profile_count"], errors="ignore")
            fused_tbl = fused_tbl.merge(comm_counts, on=grouping_cols, how="left")

        key_id_col = "profile_id"

        # Patient block: separate features by type
        RISK = {
            "Hearingloss", "BrainInjury", "Hypertension",
            "Alcohol", "Obesity", "Diabetes"
        }
        # Continuous numeric features (should be standardized, not encoded)
        NUMERIC_CONTINUOUS = {
            "LENSTAYD_LOG", "NUM_VISITS",
        "READMIT_COUNT", "READMIT_RATE", "REVISIT_30",
        }

        bin_cols = [c for c in selected_cols if c in RISK and c in fused_tbl.columns]
        num_cols = [c for c in selected_cols if c in NUMERIC_CONTINUOUS and c in fused_tbl.columns]
        cat_cols = [c for c in selected_cols if c not in bin_cols and c not in num_cols]

        # Experimental encoding mode
        encoding_metadata = {
            "mode": "experimental" if experimental_encoding else "standard",
            "categorical_mappings": {},
            "bitflag_mapping": {},
        }

        if experimental_encoding:
            # Integer-encode categoricals
            X_cat, cat_mappings = integer_encode_categoricals(fused_tbl, cat_cols)
            encoding_metadata["categorical_mappings"] = cat_mappings

            # Bitflag-encode comorbidities
            X_bin, bitflag_map = bitflag_encode_multibinary(fused_tbl, bin_cols)
            encoding_metadata["bitflag_mapping"] = bitflag_map

            # Standardize continuous numerics (like ZIP features)
            if num_cols:
                X_num = fused_tbl[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
                X_num_std = _standardize(X_num)
            else:
                X_num_std = pd.DataFrame(index=fused_tbl.index)

            patient_block = pd.concat([X_cat, X_bin, X_num_std], axis=1)
        else:
            # Standard one-hot encoding
            X_cat = _one_hot(fused_tbl, cat_cols)
            X_bin = pd.DataFrame(index=fused_tbl.index)
            for c in bin_cols:
                X_bin[c] = (
                    pd.to_numeric(fused_tbl[c], errors="coerce")
                    .fillna(0.0).clip(0, 1).astype(float)
                )

            # Standardize continuous numerics (same as experimental)
            if num_cols:
                X_num = fused_tbl[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
                X_num_std = _standardize(X_num)
            else:
                X_num_std = pd.DataFrame(index=fused_tbl.index)

            patient_block = pd.concat([X_cat, X_bin, X_num_std], axis=1)

        # ZIP block (numeric std + optional one-hot community on split path)
        zip_num_cols2 = []
        if use_degree and "zip_degree" in fused_tbl.columns: zip_num_cols2.append("zip_degree")
        if use_pr and "zip_pagerank" in fused_tbl.columns: zip_num_cols2.append("zip_pagerank")
        if use_btw and "zip_betweenness" in fused_tbl.columns: zip_num_cols2.append("zip_betweenness")

        zip_num_df = (
            fused_tbl[zip_num_cols2].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            if zip_num_cols2 else pd.DataFrame(index=fused_tbl.index)
        )
        zip_num_std = _standardize(zip_num_df) if not zip_num_df.empty else zip_num_df

        # Handle zip_community encoding (experimental vs standard)
        zip_comm_df = pd.DataFrame(index=fused_tbl.index)
        if use_zip_comm and "zip_community" in fused_tbl.columns:
            if experimental_encoding:
                # Integer encode zip_community
                zip_comm_df, comm_mapping = integer_encode_categoricals(
                    fused_tbl, ["zip_community"]
                )
                encoding_metadata["categorical_mappings"].update(comm_mapping)
            else:
                # One-hot encode zip_community
                zip_comm_df = _one_hot(fused_tbl, ["zip_community"])

        zip_block = pd.concat([zip_num_std, zip_comm_df], axis=1)

        # Final fused (no block weights on this page)
        X_fused = pd.concat([patient_block, zip_block], axis=1).fillna(0.0)

        # Persist artifacts
        st.session_state["pf_fused_table"]   = fused_tbl
        st.session_state["pf_patient_block_cols"] = list(patient_block.columns)
        st.session_state["pf_zip_block_cols"]     = list(zip_block.columns)
        # Use to_numpy with na_value so pd.NA from nullable dtypes (Int64, Float64,
        # boolean) becomes 0.0 rather than surviving as a Python object, which would
        # cause float(pd.NA) → TypeError when the graph builder calls .astype(float).
        st.session_state["pf_fused_matrix"]  = X_fused.to_numpy(dtype=float, na_value=0.0)
        st.session_state["pf_fused_index"]   = (
            fused_tbl.get(
                key_id_col, pd.Series(range(len(fused_tbl)))
            ).tolist()
        )
        st.session_state["pf_fused_counts"]  = (
            fused_tbl.get(
                "profile_count", pd.Series([np.nan]*len(fused_tbl))
            ).tolist()
        )
        st.session_state["pf_encoding_metadata"] = encoding_metadata
        st.session_state["pf_fused_meta"]    = {
            "rows": int(X_fused.shape[0]),
            "cols_total": int(X_fused.shape[1]),
            "cols_patient": int(patient_block.shape[1]),
            "cols_zip": int(zip_block.shape[1]),
            "key_id_col": key_id_col,
            "encoding_mode": encoding_metadata["mode"],
        }

        st.success(f"PSN features generated: {X_fused.shape[0]} rows × {X_fused.shape[1]} cols.")

# ---------------------------------------------------------------------
# Results Display (only shown after Generate is clicked)
# ---------------------------------------------------------------------
if generate_clicked or "pf_profiles_base" in st.session_state:
    st.divider()
    st.header("PSN Feature Results")
    
    # --- Run settings ---
    if "pf_controls_run" in st.session_state:
        st.subheader("Run Settings")
        run = st.session_state["pf_controls_run"]
        settings_df = pd.DataFrame([
            {
                "Parameter": "Profile Columns",
                "Value": ", ".join(run["selected_cols"]) if run["selected_cols"] else "None"
            },
            {
                "Parameter": "Neighborhood Features",
                "Value": ", ".join([k for k, v in run["zip_features"].items() if v]) or "None"
            },
            {
                "Parameter": "Encoding Mode",
                "Value": "Experimental" if run.get("experimental_encoding", False) else "Standard"
            },
        ])
        st.dataframe(settings_df, use_container_width=False, hide_index=True)
    else:
        st.info("Configure settings above and click 'Generate PSN Features'.")
    
    # --- PSN feature matrix summary ---
    st.subheader("PSN Feature Matrix — Summary")
    meta = st.session_state.get("pf_fused_meta", {})
    if meta:
        summary_df = pd.DataFrame([
            {"Metric": "Total Rows", "Value": str(meta.get("rows", "N/A"))},
            {"Metric": "Total Columns", "Value": str(meta.get("cols_total", "N/A"))},
            {"Metric": "Patient Block Columns", "Value": str(meta.get("cols_patient", "N/A"))},
            {"Metric": "Neighborhood Block Columns", "Value": str(meta.get("cols_zip", "N/A"))},
            {"Metric": "Encoding Mode", "Value": str(meta.get("encoding_mode", "standard"))},
        ])
        st.dataframe(summary_df, width='content', hide_index=True)
        
        # Block details
        col_block1, col_block2 = st.columns(2)
        with col_block1:
            st.caption("**Patient Block Columns**")
            patient_cols = st.session_state.get("pf_patient_block_cols", [])
            if patient_cols:
                st.text(f"Count: {len(patient_cols)}")
                st.caption(", ".join(patient_cols))
                # st.caption(", ".join(patient_cols[:10]) + ("..." if len(patient_cols) > 10 else ""))
        
        with col_block2:
            st.caption("**Neighborhood Block Columns**")
            zip_cols = st.session_state.get("pf_zip_block_cols", [])
            if zip_cols:
                st.text(f"Count: {len(zip_cols)}")
                st.caption(", ".join(zip_cols))
    
    # --- PSN table preview ---
    st.subheader("PSN Feature Table — Preview")
    ft = st.session_state.get("pf_fused_table")
    if ft is not None:
        st.dataframe(ft.head(12), width='content', hide_index=True)

    # --- Encoded matrix preview (experimental mode) ---
    meta = st.session_state.get("pf_fused_meta", {})
    if meta.get("encoding_mode") == "experimental":
        st.subheader("Encoded Feature Matrix — Preview (Experimental)")
        st.caption(
            "Integer-encoded categoricals + bitflag-encoded comorbidities "
            "+ standardized neighborhood features"
        )

        fused_matrix = st.session_state.get("pf_fused_matrix")
        patient_cols = st.session_state.get("pf_patient_block_cols", [])
        zip_cols = st.session_state.get("pf_zip_block_cols", [])

        if fused_matrix is not None and (patient_cols or zip_cols):
            all_cols = patient_cols + zip_cols
            encoded_df = pd.DataFrame(
                fused_matrix[:12],  # First 12 rows
                columns=all_cols
            )

            # Format encoded columns for readability
            encoding_meta = st.session_state.get("pf_encoding_metadata", {})
            cat_maps = encoding_meta.get("categorical_mappings", {})
            bitflag_map = encoding_meta.get("bitflag_mapping", {})

            # Use shared formatting function
            encoded_df = format_encoded_display(
                encoded_df,
                cat_maps,
                bitflag_map,
                bitflag_column="comorbidities_encoded"
            )

            st.dataframe(encoded_df, width='content', hide_index=True)

            with st.expander("View Encoding Mappings", expanded=False):
                # Categorical mappings
                cat_maps = encoding_meta.get("categorical_mappings", {})
                if cat_maps:
                    st.caption("**Categorical Encodings**")
                    for col_name, mapping in cat_maps.items():
                        st.text(f"{col_name}_encoded:")
                        mapping_str = ", ".join(
                            f"{cat}={val}" for cat, val in mapping.items()
                        )
                        st.caption(f"  {mapping_str}")

                # Bitflag mapping
                bitflag_map = encoding_meta.get("bitflag_mapping", {})
                if bitflag_map:
                    st.caption("**Comorbidities Encoding (Bitflag)**")
                    st.text("comorbidities_encoded bits:")
                    for bit_position, col_name in bitflag_map.items():
                        st.caption(f"  Bit {bit_position}: {col_name}")
