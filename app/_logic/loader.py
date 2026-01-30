import streamlit as st
import pandas as pd


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def ensure_data_loaded(force_reload: bool = False) -> bool:
    """
    Ensure core datasets are loaded into st.session_state.

    Loads: patients_df, zip_df, zip_coords, wa_boundary.
    Shows a spinner while loading. Returns True if successful.
    """
    from app._logic.config import load_config

    keys = ["patients_df", "zip_df", "zip_coords", "wa_boundary"]
    if not force_reload and all(k in st.session_state for k in keys):
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
