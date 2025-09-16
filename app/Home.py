import streamlit as st
import pandas as pd
import geopandas as gpd
import hashlib

from app._logic.config import load_config
from app._logic.loader import load_csv

st.set_page_config(page_title="Context-Aware Patient Graphs", layout="wide")

st.title("A Graph-Based Framework for Context-Aware Patient Similarity Networks")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def file_fingerprint(path: str, sample: int = 5000) -> str:
    """Lightweight hash of file path + sample bytes; avoids hashing whole file."""
    try:
        with open(path, "rb") as f:
            head = f.read(sample)
        return hashlib.sha1((path + str(len(head)) + str(head)).encode("utf-8", errors="ignore")).hexdigest()
    except Exception:
        # Fall back to hashing the path string only
        return hashlib.sha1(path.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------
# Config and paths
# ---------------------------------------------------------------------
cfg = load_config()
patients_path = cfg["paths"]["patients"]
zip_path = cfg["paths"]["zip_context"]
coords_path = cfg["paths"]["zip_coords"]
wa_zip_path = cfg["paths"]["wa_state_zip"]  # local zip: data/raw/cb_2020_us_state_20m.zip

# Compose a key that changes if any inputs change
key = hashlib.sha1(
    f"patients:{file_fingerprint(patients_path)}|"
    f"zip:{file_fingerprint(zip_path)}|"
    f"coords:{file_fingerprint(coords_path)}|"
    f"wa_zip:{file_fingerprint(wa_zip_path)}".encode("utf-8")
).hexdigest()

reload = st.button("Reload Data")

# ---------------------------------------------------------------------
# Load data conditionally
# ---------------------------------------------------------------------
if (
    not reload
    and st.session_state.get("home_key") == key
    and "patients_df" in st.session_state
    and "zip_df" in st.session_state
    and "zip_coords" in st.session_state
    and "wa_boundary" in st.session_state
):
    pat = st.session_state["patients_df"]
    zipc = st.session_state["zip_df"]
    zip_coords = st.session_state["zip_coords"]
    wa_boundary = st.session_state["wa_boundary"]
else:
    # CSVs
    pat = load_csv(patients_path)
    zipc = load_csv(zip_path)
    zip_coords = load_csv(coords_path)

    # Validate/clean coords
    required_cols = {"zip", "lat", "lng"}
    if not required_cols.issubset(set(zip_coords.columns)):
        st.error("zip_coords must contain columns: zip, lat, lng")
        st.stop()
    zip_coords = zip_coords.copy()
    zip_coords["zip"] = zip_coords["zip"].astype(str)
    zip_coords["lat"] = pd.to_numeric(zip_coords["lat"], errors="coerce")
    zip_coords["lng"] = pd.to_numeric(zip_coords["lng"], errors="coerce")
    zip_coords = zip_coords.dropna(subset=["lat", "lng"])

    # Washington boundary from local zip
    try:
        states = gpd.read_file(wa_zip_path)
        wa_boundary = states[states["STUSPS"] == "WA"]
        if wa_boundary.empty:
            st.error("Washington boundary not found in provided shapefile.")
            st.stop()
    except Exception as e:
        st.error(f"Failed to load Washington boundary from {wa_zip_path}: {e}")
        st.stop()

    # Save to session
    st.session_state["patients_df"] = pat
    st.session_state["zip_df"] = zipc
    st.session_state["zip_coords"] = zip_coords
    st.session_state["wa_boundary"] = wa_boundary
    st.session_state["home_key"] = key

    if reload:
        st.success("Data reloaded successfully!")

# ---------------------------------------------------------------------
# Display status
# ---------------------------------------------------------------------
st.subheader("Data status")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Patients rows", f"{len(pat):,}")
with col2:
    st.metric("ZIP context rows", f"{len(zipc):,}")
with col3:
    st.metric("ZIP coords rows", f"{len(zip_coords):,}")
with col4:
    st.metric("WA polygons", f"{len(wa_boundary):,}")

st.markdown("Use the sidebar to navigate pages. Run each step or trigger the full pipeline.")