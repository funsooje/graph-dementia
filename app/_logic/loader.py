import json
from pathlib import Path

import streamlit as st
import pandas as pd


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# ---------------------------------------------------------------------
# Default feature groups (canonical definition)
# ---------------------------------------------------------------------
_ENV = [
    "EPL_OZONE", "EPL_PM", "EPL_DSLPM", "EPL_NPL",
    "EPL_TRI", "EPL_TSD", "EPL_RMP", "EPL_COAL", "EPL_LEAD", "EPL_PARK",
    "EPL_HOUAGE", "EPL_WLKIND", "EPL_ROAD", "EPL_RAIL", "EPL_AIRPRT",
    "EPL_IMPWTR",
]
_SES = [
    "EPL_MINRTY", "EPL_POV200", "EPL_NOHSDP", "EPL_UNEMP",
    "EPL_RENTER", "EPL_HOUBDN", "EPL_UNINSUR", "EPL_NOINT", "EPL_AGE17",
    "EPL_AGE65", "EPL_DISABL", "EPL_LIMENG", "EPL_MOBILE", "EPL_GROUPQ",
]
_ENV_RAW = [
    "E_OZONE", "E_PM", "E_DSLPM", "E_NPL",
    "E_TRI", "E_TSD", "E_RMP", "E_COAL", "E_LEAD", "E_PARK",
    "E_HOUAGE", "E_WLKIND", "E_ROAD", "E_RAIL", "E_AIRPRT",
    "E_IMPWTR",
]
_SES_RAW = [
    "EP_MINRTY", "EP_POV200", "EP_NOHSDP", "EP_UNEMP",
    "EP_RENTER", "EP_HOUBDN", "EP_UNINSUR", "EP_NOINT", "EP_AGE17",
    "EP_AGE65", "EP_DISABL", "EP_LIMENG", "EP_MOBILE", "EP_GROUPQ",
]

DEFAULT_FEATURE_GROUPS = {
    "all": {"name": "All Features", "short_code": "all",
            "columns": _ENV + _SES},
    "all_raw": {"name": "All Features (Raw)", "short_code": "all_raw",
                "columns": _ENV_RAW + _SES_RAW},
    "env": {"name": "Environment", "short_code": "env",
            "columns": _ENV},
    "env_raw": {"name": "Environment (Raw)", "short_code": "env_raw",
                "columns": _ENV_RAW},
    "ses": {"name": "Socioeconomic Status", "short_code": "ses",
            "columns": _SES},
    "ses_raw": {"name": "Socioeconomic Status (Raw)", "short_code": "ses_raw",
                "columns": _SES_RAW},
    "air_water": {"name": "Air and Water Quality", "short_code": "air_water",
                  "columns": ["EPL_PM25", "EPL_OZONE", "EPL_DSLPM", "EPL_WATER",
                              "EPL_AIR", "EPL_PM10", "EPL_PM2_5", "EPL_NO2",
                              "EPL_SO2", "EPL_CO"]},
    "toxic_site": {"name": "Toxic Sites", "short_code": "toxic_site",
                   "columns": ["EPL_NPL", "EPL_RMP", "EPL_TSDF", "EPL_LDPNT",
                               "EPL_LEAD"]},
    "built_env": {"name": "Built Environment", "short_code": "built_env",
                  "columns": ["EPL_PARK", "EPL_GREEN", "EPL_WALK", "EPL_BIKE",
                              "EPL_FOOD"]},
    "transport": {"name": "Transportation", "short_code": "transport",
                  "columns": ["EPL_TRANS", "EPL_VEH", "EPL_COMMUTE",
                              "EPL_TRANSIT"]},
    "pop": {"name": "Population", "short_code": "pop",
            "columns": ["EPL_POP", "EPL_AGE65", "EPL_AGE17", "EPL_MINRTY",
                        "EPL_LIMENG"]},
    "household": {"name": "Household", "short_code": "household",
                  "columns": ["EPL_MLTPL", "EPL_MOBILE", "EPL_CROWD",
                              "EPL_NOVEH", "EPL_GROUPQ"]},
    "housing": {"name": "Housing", "short_code": "housing",
                "columns": ["EPL_HBURDEN", "EPL_RENTBURDEN", "EPL_OWNBURDEN",
                            "EPL_UNITS", "EPL_AGE"]},
}


# ---------------------------------------------------------------------
# Feature group loading
# ---------------------------------------------------------------------
_DEFAULT_FG_PATH = Path("data/config/default_feature_groups.json")
_CUSTOM_FG_PATH = Path("data/config/feature_groups.json")


def _ensure_feature_groups():
    """Load default and custom feature groups into session state."""
    if "default_feature_groups" not in st.session_state:
        if _DEFAULT_FG_PATH.exists():
            with open(_DEFAULT_FG_PATH, "r", encoding="utf-8") as f:
                st.session_state["default_feature_groups"] = json.load(f)
        else:
            # Use hardcoded defaults and persist to disk
            st.session_state["default_feature_groups"] = DEFAULT_FEATURE_GROUPS
            _DEFAULT_FG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_DEFAULT_FG_PATH, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_FEATURE_GROUPS, f, indent=4)

    if "feature_groups" not in st.session_state:
        if _CUSTOM_FG_PATH.exists():
            with open(_CUSTOM_FG_PATH, "r", encoding="utf-8") as f:
                st.session_state["feature_groups"] = json.load(f)
        else:
            st.session_state["feature_groups"] = {}


# ---------------------------------------------------------------------
# Main data loader
# ---------------------------------------------------------------------
def ensure_data_loaded(force_reload: bool = False) -> bool:
    """
    Ensure core datasets and feature groups are loaded into st.session_state.

    Loads: patients_df, zip_df, zip_coords, wa_boundary,
           default_feature_groups, feature_groups.
    Shows a spinner while loading. Returns True if successful.
    """
    from app._logic.config import load_config

    all_keys = [
        "patients_df", "zip_df", "zip_coords", "wa_boundary",
        "default_feature_groups",
    ]
    if not force_reload and all(k in st.session_state for k in all_keys):
        return True

    # Feature groups (lightweight — from JSON config files)
    _ensure_feature_groups()

    # Core data — skip if already loaded (unless force)
    data_keys = ["patients_df", "zip_df", "zip_coords", "wa_boundary"]
    if not force_reload and all(k in st.session_state for k in data_keys):
        return True

    try:
        cfg = load_config()
        with st.spinner("Loading data..."):
            # CSVs
            pat = load_csv(cfg["paths"]["patients"])
            zipc = load_csv(cfg["paths"]["zip_context"])
            zip_coords = load_csv(cfg["paths"]["zip_coords"])

            # Validate/clean coords
            required = {"zip", "lat", "lng"}
            if not required.issubset(set(zip_coords.columns)):
                st.error("zip_coords must contain columns: zip, lat, lng")
                return False
            zip_coords = zip_coords.copy()
            zip_coords["zip"] = zip_coords["zip"].astype(str)
            zip_coords["lat"] = pd.to_numeric(zip_coords["lat"], errors="coerce")
            zip_coords["lng"] = pd.to_numeric(zip_coords["lng"], errors="coerce")
            zip_coords = zip_coords.dropna(subset=["lat", "lng"])

            # Washington boundary
            import geopandas as gpd

            states = gpd.read_file(cfg["paths"]["wa_state_zip"])
            wa_boundary = states[states["STUSPS"] == "WA"]
            if wa_boundary.empty:
                st.error("Washington boundary not found in shapefile.")
                return False

            # AGE_BIN normalization
            if "AGE_BIN" in pat.columns:
                pat = pat.copy()
                pat["AGE_BIN"] = (
                    pat["AGE_BIN"]
                    .astype("string")
                    .str.strip()
                    .replace({"<65": "45-65"})
                )

            # Store in session state
            st.session_state["patients_df"] = pat
            st.session_state["zip_df"] = zipc
            st.session_state["zip_coords"] = zip_coords
            st.session_state["wa_boundary"] = wa_boundary

        return True
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return False
