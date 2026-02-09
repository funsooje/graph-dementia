import streamlit as st
import pandas as pd
import numpy as np
import hashlib
from pandas.api.types import is_numeric_dtype

from app._logic.loader import ensure_data_loaded

st.set_page_config(page_title="Context-Aware Patient Graphs", layout="wide")

st.title("A Graph-Based Framework for Context-Aware Patient Similarity Networks")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def present_cols(df, cols):
    return [c for c in cols if c in df.columns]

def numeric_summary(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    cols = present_cols(df, cols)
    if not cols:
        return pd.DataFrame()
    use = df[cols].select_dtypes(include=[np.number])
    if use.empty:
        return pd.DataFrame()
    desc = use.describe(percentiles=[0.25, 0.5, 0.75]).T
    cols_order = [c for c in ["count","mean","std","min","25%","50%","75%","max"] if c in desc.columns]
    return desc[cols_order]

def categorical_top(df: pd.DataFrame, cols: list, topn: int = 10) -> dict:
    out = {}
    for c in cols:
        if c in df.columns and not is_numeric_dtype(df[c]):
            out[c] = df[c].value_counts(dropna=False).head(topn)
    return out

def binary_prevalence(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    use = [c for c in cols if c in df.columns]
    if not use:
        return pd.DataFrame()
    arr = df[use].apply(pd.to_numeric, errors="coerce")
    prev = arr.mean().rename("prevalence")
    return pd.DataFrame(prev).sort_values(by="prevalence", ascending=False)

def df_fingerprint(df: pd.DataFrame, cols: list, sample: int = 1000) -> str:
    """Lightweight content hash: schema + head/tail sample to avoid hashing entire large frames."""
    cols = present_cols(df, cols)
    meta = {
        "shape": df.shape,
        "cols": cols,
        "dtypes": {c: str(df[c].dtype) for c in cols},
    }
    parts = [str(meta).encode("utf-8")]
    if len(df) > 0 and cols:
        head = df[cols].head(sample).to_csv(index=False).encode("utf-8")
        tail = df[cols].tail(sample).to_csv(index=False).encode("utf-8")
        parts += [head, tail]
    return hashlib.sha1(b"|".join(parts)).hexdigest()

# ---------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------
PAT_GROUPS = {
    "ids": ["SEQ_NO", "REC_KEY", "PATIENTID"],
    "location": ["ZIPCODE"],
    "demographics": ["AGE", "SEX", "Race", "AGE_BIN"],
    "utilization": ["STAYTYPE", "LENSTAYD", "LENSTAYD_BIN", "LENSTAYD_LOG", "DIAGCNT", "PAYER"],
    "risk_binaries": ["Hearingloss", "BrainInjury", "Hypertension", "Alcohol", "Obesity", "Diabetes", "REVISIT_30"],
}

ZIP_GROUPS = {
    "zip_key": ["ZIPCODE"],
    "env_raw": [
        "E_OZONE","E_PM","E_DSLPM","E_NPL","E_TRI","E_TSD","E_RMP","E_COAL","E_LEAD","E_PARK",
        "E_HOUAGE","E_WLKIND","E_ROAD","E_RAIL","E_AIRPRT","E_IMPWTR"
    ],
    "socdem_raw": [
        "EP_MINRTY","EP_POV200","EP_NOHSDP","EP_UNEMP","EP_RENTER","EP_HOUBDN","EP_UNINSUR",
        "EP_NOINT","EP_AGE17","EP_AGE65","EP_DISABL","EP_LIMENG","EP_MOBILE","EP_GROUPQ"
    ],
    "env_norm": [
        "EPL_OZONE","EPL_PM","EPL_DSLPM","EPL_NPL","EPL_TRI","EPL_TSD","EPL_RMP","EPL_COAL",
        "EPL_LEAD","EPL_PARK","EPL_HOUAGE","EPL_WLKIND","EPL_ROAD","EPL_RAIL","EPL_AIRPRT",
        "EPL_IMPWTR"
    ],
    "socdem_norm": [
        "EPL_MINRTY","EPL_POV200","EPL_NOHSDP","EPL_UNEMP","EPL_RENTER",
        "EPL_HOUBDN","EPL_UNINSUR","EPL_NOINT","EPL_AGE17","EPL_AGE65","EPL_DISABL",
        "EPL_LIMENG","EPL_MOBILE","EPL_GROUPQ"
    ],
}

def data_cache_key(pat: pd.DataFrame, zipc: pd.DataFrame) -> str:
    pat_cols = PAT_GROUPS["location"] + PAT_GROUPS["demographics"] + PAT_GROUPS["utilization"] + PAT_GROUPS["risk_binaries"]
    zip_cols = ZIP_GROUPS["zip_key"] + ZIP_GROUPS["env_raw"] + ZIP_GROUPS["socdem_raw"] + ZIP_GROUPS["env_norm"] + ZIP_GROUPS["socdem_norm"]
    return hashlib.sha1(
        f"pat:{df_fingerprint(pat, pat_cols)}|zip:{df_fingerprint(zipc, zip_cols)}".encode("utf-8")
    ).hexdigest()

# ---------------------------------------------------------------------
# Load data (shared loader)
# ---------------------------------------------------------------------
reload = st.button("Reload Data")

if not ensure_data_loaded(force_reload=reload):
    st.stop()

if reload:
    st.success("Data reloaded successfully!")

pat = st.session_state["patients_df"]
zipc = st.session_state["zip_df"]
zip_coords = st.session_state["zip_coords"]
wa_boundary = st.session_state["wa_boundary"]

# ---------------------------------------------------------------------
# Data status overview
# ---------------------------------------------------------------------
st.subheader("Data Status")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Patients rows", f"{len(pat):,}")
with col2:
    st.metric("ZIP context rows", f"{len(zipc):,}")
with col3:
    st.metric("ZIP coords rows", f"{len(zip_coords):,}")
with col4:
    st.metric("WA polygons", f"{len(wa_boundary):,}")

# ---------------------------------------------------------------------
# Compute summaries (with caching)
# ---------------------------------------------------------------------
summary_key = data_cache_key(pat, zipc)

if (
    st.session_state.get("data_config_key") == summary_key
    and "data_config_cache" in st.session_state
):
    cache = st.session_state["data_config_cache"]
    num_df_pat = cache["num_df_pat"]
    cat_tops_pat = cache["cat_tops_pat"]
    prev_df = cache["prev_df"]
    env_sum_raw = cache["env_sum_raw"]
    env_sum_norm = cache["env_sum_norm"]
    soc_sum_raw = cache["soc_sum_raw"]
    soc_sum_norm = cache["soc_sum_norm"]
else:
    # Patients
    dem_cols = present_cols(pat, PAT_GROUPS["demographics"])
    util_cols = present_cols(pat, PAT_GROUPS["utilization"])
    risk_cols = present_cols(pat, PAT_GROUPS["risk_binaries"])

    num_df_pat = numeric_summary(pat, dem_cols + util_cols)

    cat_cols = [c for c in (dem_cols + util_cols) if c in pat.columns and not is_numeric_dtype(pat[c])]
    cat_tops_pat = categorical_top(pat, cat_cols, topn=10)

    prev_df = binary_prevalence(pat, risk_cols)

    # ZIP summaries
    env_cols = present_cols(zipc, ZIP_GROUPS["env_raw"])
    soc_cols = present_cols(zipc, ZIP_GROUPS["socdem_raw"])
    env_n_cols = present_cols(zipc, ZIP_GROUPS["env_norm"])
    soc_n_cols = present_cols(zipc, ZIP_GROUPS["socdem_norm"])

    env_sum_raw = numeric_summary(zipc, env_cols)
    env_sum_norm = numeric_summary(zipc, env_n_cols)
    soc_sum_raw = numeric_summary(zipc, soc_cols)
    soc_sum_norm = numeric_summary(zipc, soc_n_cols)

    # Store cache
    st.session_state["data_config_key"] = summary_key
    st.session_state["data_config_cache"] = {
        "num_df_pat": num_df_pat,
        "cat_tops_pat": cat_tops_pat,
        "prev_df": prev_df,
        "env_sum_raw": env_sum_raw,
        "env_sum_norm": env_sum_norm,
        "soc_sum_raw": soc_sum_raw,
        "soc_sum_norm": soc_sum_norm,
    }

# ---------------------------------------------------------------------
# Data Summary Tabs
# ---------------------------------------------------------------------
st.subheader("Data Summary")
tab_pat, tab_zip = st.tabs(["Patients Data", "ZIP Context Data"])

with tab_pat:
    st.write("Patients Data Preview")
    st.dataframe(pat.head(10), width='content')

    st.markdown("Schema and summaries by group")

    st.write("Numeric Summary")
    if not num_df_pat.empty:
        st.dataframe(
            num_df_pat.style.format({
                "count":"{:,.0f}",
                "mean":"{:.2f}","std":"{:.2f}","min":"{:.2f}",
                "25%":"{:.2f}","50%":"{:.2f}","75%":"{:.2f}",
                "max":"{:.2f}"
            }),
            width='content'
        )
    else:
        st.info("No numeric columns found for demographics/utilization.")

    st.write("Categorical Summaries")
    if cat_tops_pat:
        for c, s in cat_tops_pat.items():
            st.write(c)
            styled = (
                s.to_frame("count")
                .style.format({"count": "{:,.0f}"})
                .set_properties(**{"text-align": "left"})
            )
            st.dataframe(styled, width='content')
    else:
        st.info("No categorical columns processed.")

    st.write("Risk factor prevalence (binary columns)")
    if not prev_df.empty:
        st.dataframe(prev_df.style.format({"prevalence":"{:.1%}"}), width='content')
    else:
        st.info("No risk binary columns found.")

with tab_zip:
    st.markdown("ZIP Code Context Data Preview")
    st.dataframe(zipc.head(10), width='content')

    st.markdown("ZIP Code Context Grouped Summaries")

    col1, col2 = st.columns(2)
    with col1:
        st.write("Environmental (Raw)")
        if not env_sum_raw.empty:
            st.dataframe(
                env_sum_raw.style.format({
                    "count":"{:,.0f}",
                    "mean":"{:.3f}","std":"{:.3f}","min":"{:.3f}",
                    "25%":"{:.3f}","50%":"{:.3f}","75%":"{:.3f}",
                    "max":"{:.3f}"
                }),
                width='content'
            )
        else:
            st.info("No environmental raw columns present.")

    with col2:
        st.write("Environmental (US Normalized)")
        if not env_sum_norm.empty:
            st.dataframe(
                env_sum_norm.style.format({
                    "count":"{:,.0f}",
                    "mean":"{:.3f}","std":"{:.3f}","min":"{:.3f}",
                    "25%":"{:.3f}","50%":"{:.3f}","75%":"{:.3f}",
                    "max":"{:.3f}"
                }),
                width='content'
            )
        else:
            st.info("No environmental normalized columns present.")

    col3, col4 = st.columns(2)
    with col3:
        st.write("Sociodemographic (Raw)")
        if not soc_sum_raw.empty:
            st.dataframe(
                soc_sum_raw.style.format({
                    "count":"{:,.0f}",
                    "mean":"{:.3f}","std":"{:.3f}","min":"{:.3f}",
                    "25%":"{:.3f}","50%":"{:.3f}","75%":"{:.3f}",
                    "max":"{:.3f}"
                }),
                width='content'
            )
        else:
            st.info("No sociodemographic raw columns present.")

    with col4:
        st.write("Sociodemographic (US Normalized)")
        if not soc_sum_norm.empty:
            st.dataframe(
                soc_sum_norm.style.format({
                    "count":"{:,.0f}",
                    "mean":"{:.3f}","std":"{:.3f}","min":"{:.3f}",
                    "25%":"{:.3f}","50%":"{:.3f}","75%":"{:.3f}",
                    "max":"{:.3f}"
                }),
                width='content'
            )
        else:
            st.info("No sociodemographic normalized columns present.")

st.markdown("---")
st.markdown("Use the sidebar to navigate pages. Run each step or trigger the full pipeline.")