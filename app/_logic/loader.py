import json
from pathlib import Path
from typing import Optional

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
# Patient dataset helpers
# ---------------------------------------------------------------------

def _load_all_patient_datasets(cfg: dict) -> dict:
    """
    Load all patient datasets from config into a dict.

    Returns:
        dict: {key -> {"df": DataFrame, "label": str}}

    Falls back to legacy paths.patients if patient_datasets is not present.
    """
    pat_cfg = cfg.get("patient_datasets", {})

    # Legacy fallback: support old YAML format with paths.patients
    if not pat_cfg:
        paths = cfg.get("paths", {})
        if "patients" in paths:
            pat_cfg = {
                "default": {
                    "label": "Patient Level",
                    "path": paths["patients"],
                }
            }

    datasets = {}
    errors = []
    for key, ds_cfg in pat_cfg.items():
        try:
            df = load_csv(ds_cfg["path"])
            if "AGE_BIN" in df.columns:
                df = df.copy()
                df["AGE_BIN"] = (
                    df["AGE_BIN"]
                    .astype("string")
                    .str.strip()
                    .replace({"<65": "45-65"})
                )
            datasets[key] = {
                "df": df,
                "label": ds_cfg.get("label", key),
            }
        except Exception as e:
            errors.append(f"'{key}': {e}")

    if errors:
        st.warning("Some patient datasets failed to load:\n" + "\n".join(f"- {e}" for e in errors))

    return datasets


def _render_patient_selector() -> Optional[str]:
    """
    Render a sidebar selectbox for switching the active patient dataset.
    Returns the selected dataset key, or None if no datasets are loaded.
    Updates st.session_state['active_patient_key'] and the patients_df alias.
    """
    datasets = st.session_state.get("patient_datasets", {})
    if not datasets:
        return None

    keys = list(datasets.keys())
    current_key = st.session_state.get("active_patient_key", keys[0])
    if current_key not in keys:
        current_key = keys[0]

    labels = {k: v["label"] for k, v in datasets.items()}
    idx = keys.index(current_key)

    selected_key = st.sidebar.selectbox(
        "Patient Dataset",
        options=keys,
        format_func=lambda k: labels[k],
        index=idx,
        key="_pat_dataset_selector",
    )

    st.session_state["active_patient_key"] = selected_key
    st.session_state["patients_df"] = datasets[selected_key]["df"]

    return selected_key


# ---------------------------------------------------------------------
# Context dataset helpers
# ---------------------------------------------------------------------

def _clean_coords(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Validate and clean a zip_coords DataFrame. Returns cleaned df or None."""
    required = {"zip", "lat", "lng"}
    if not required.issubset(set(df.columns)):
        return None
    df = df.copy()
    df["zip"] = df["zip"].astype(str)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lng"] = pd.to_numeric(df["lng"], errors="coerce")
    return df.dropna(subset=["lat", "lng"])


def _load_all_context_datasets(cfg: dict) -> dict:
    """
    Load all context datasets from config into a dict.

    Returns:
        dict: {key -> {"df": DataFrame, "coords": DataFrame, "label": str}}

    Falls back gracefully to old paths.zip_context / paths.zip_coords if
    context_datasets is not present in the config.
    """
    ctx_cfg = cfg.get("context_datasets", {})

    # Legacy fallback: support old YAML format with paths.zip_context / paths.zip_coords
    if not ctx_cfg:
        paths = cfg.get("paths", {})
        if "zip_context" in paths and "zip_coords" in paths:
            ctx_cfg = {
                "default": {
                    "label": "Default",
                    "zip_context": paths["zip_context"],
                    "zip_coords": paths["zip_coords"],
                }
            }

    datasets = {}
    errors = []
    for key, ds_cfg in ctx_cfg.items():
        try:
            zipc = load_csv(ds_cfg["zip_context"])
            raw_coords = load_csv(ds_cfg["zip_coords"])
            coords = _clean_coords(raw_coords)
            if coords is None:
                errors.append(f"'{key}': zip_coords missing required columns (zip, lat, lng)")
                continue
            datasets[key] = {
                "df": zipc,
                "coords": coords,
                "label": ds_cfg.get("label", key),
            }
        except Exception as e:
            errors.append(f"'{key}': {e}")

    if errors:
        st.warning("Some context datasets failed to load:\n" + "\n".join(f"- {e}" for e in errors))

    return datasets


def _render_context_selector() -> Optional[str]:
    """
    Render a sidebar selectbox for switching the active context dataset.
    Returns the selected dataset key, or None if no datasets are loaded.
    Updates st.session_state['active_context_key'] and the zip_df/zip_coords aliases.
    """
    datasets = st.session_state.get("context_datasets", {})
    if not datasets:
        return None

    keys = list(datasets.keys())
    current_key = st.session_state.get("active_context_key", keys[0])
    if current_key not in keys:
        current_key = keys[0]

    labels = {k: v["label"] for k, v in datasets.items()}
    idx = keys.index(current_key)

    selected_key = st.sidebar.selectbox(
        "Context Dataset",
        options=keys,
        format_func=lambda k: labels[k],
        index=idx,
        key="_ctx_dataset_selector",
    )

    # Update active key and backward-compatible aliases
    st.session_state["active_context_key"] = selected_key
    st.session_state["zip_df"] = datasets[selected_key]["df"]
    st.session_state["zip_coords"] = datasets[selected_key]["coords"]

    return selected_key


# ---------------------------------------------------------------------
# Main data loader
# ---------------------------------------------------------------------
def ensure_data_loaded(force_reload: bool = False) -> bool:
    """
    Ensure core datasets and feature groups are loaded into st.session_state.

    Loads: patients_df (alias for active patient dataset), wa_boundary,
    all patient_datasets, and all context_datasets (zip_df/zip_coords are
    backward-compatible aliases pointing to the active context dataset).
    Renders sidebar selectors so users can switch datasets without restart.

    Returns True if successful.
    """
    from app._logic.config import load_config

    # Feature groups (lightweight — from JSON config files)
    _ensure_feature_groups()

    # Check what still needs to be loaded
    base_keys_needed = ["patients_df", "wa_boundary"]
    patients_needed = "patient_datasets" not in st.session_state
    contexts_needed = "context_datasets" not in st.session_state

    already_loaded = (
        not force_reload
        and not patients_needed
        and not contexts_needed
        and all(k in st.session_state for k in base_keys_needed)
    )
    if already_loaded:
        # Data already loaded — just update sidebar selectors + aliases
        _render_patient_selector()
        _render_context_selector()
        return True

    try:
        cfg = load_config()

        # --- Patient datasets (all of them) ---
        if force_reload or "patient_datasets" not in st.session_state:
            with st.spinner("Loading patient datasets..."):
                pat_datasets = _load_all_patient_datasets(cfg)
                if not pat_datasets:
                    st.error("No patient datasets could be loaded. Check configs/default.yaml.")
                    return False
                st.session_state["patient_datasets"] = pat_datasets

                # Default to first dataset if active key is not set or invalid
                if (
                    "active_patient_key" not in st.session_state
                    or st.session_state["active_patient_key"] not in pat_datasets
                ):
                    first_key = next(iter(pat_datasets))
                    st.session_state["active_patient_key"] = first_key
                    st.session_state["patients_df"] = pat_datasets[first_key]["df"]

        # --- WA boundary ---
        if force_reload or "wa_boundary" not in st.session_state:
            with st.spinner("Loading Washington boundary..."):
                import geopandas as gpd
                states = gpd.read_file(cfg["paths"]["wa_state_zip"])
                wa_boundary = states[states["STUSPS"] == "WA"]
                if wa_boundary.empty:
                    st.error("Washington boundary not found in shapefile.")
                    return False
                st.session_state["wa_boundary"] = wa_boundary

        # --- Context datasets (all of them) ---
        if force_reload or "context_datasets" not in st.session_state:
            with st.spinner("Loading context datasets..."):
                datasets = _load_all_context_datasets(cfg)
                if not datasets:
                    st.error("No context datasets could be loaded. Check configs/default.yaml.")
                    return False
                st.session_state["context_datasets"] = datasets

                # Default to first dataset if active key is not set or invalid
                if (
                    "active_context_key" not in st.session_state
                    or st.session_state["active_context_key"] not in datasets
                ):
                    st.session_state["active_context_key"] = next(iter(datasets))

    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return False

    # Render sidebar selectors and update aliases (always, every page render)
    _render_patient_selector()
    _render_context_selector()
    return True
