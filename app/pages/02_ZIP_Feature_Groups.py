import streamlit as st
import os
import json
import pandas as pd

st.title("ZIP Feature Groups Configuration")

if "zip_df" not in st.session_state:
    st.warning("Please run the previous page to load the ZIP dataframe first.")
    st.stop()

zip_df = st.session_state["zip_df"]

config_dir = "data/config"
config_path = os.path.join(config_dir, "feature_groups.json")
os.makedirs(config_dir, exist_ok=True)

# Define exact column lists for ENV and SES
ENV = [
    'EPL_OZONE', 'EPL_PM', 'EPL_DSLPM', 'EPL_NPL',
        'EPL_TRI', 'EPL_TSD', 'EPL_RMP', 'EPL_COAL', 'EPL_LEAD', 'EPL_PARK',
        'EPL_HOUAGE', 'EPL_WLKIND', 'EPL_ROAD', 'EPL_RAIL', 'EPL_AIRPRT',
        'EPL_IMPWTR'
]

SES = [
    'EPL_MINRTY', 'EPL_POV200', 'EPL_NOHSDP', 'EPL_UNEMP',
        'EPL_RENTER', 'EPL_HOUBDN', 'EPL_UNINSUR', 'EPL_NOINT', 'EPL_AGE17',
        'EPL_AGE65', 'EPL_DISABL', 'EPL_LIMENG', 'EPL_MOBILE', 'EPL_GROUPQ'
]

ENV_RAW = [
    'E_OZONE', 'E_PM', 'E_DSLPM', 'E_NPL',
        'E_TRI', 'E_TSD', 'E_RMP', 'E_COAL', 'E_LEAD', 'E_PARK',
        'E_HOUAGE', 'E_WLKIND', 'E_ROAD', 'E_RAIL', 'E_AIRPRT',
        'E_IMPWTR'
]

SES_RAW = [
    'EP_MINRTY', 'EP_POV200', 'EP_NOHSDP', 'EP_UNEMP',
        'EP_RENTER', 'EP_HOUBDN', 'EP_UNINSUR', 'EP_NOINT', 'EP_AGE17',
        'EP_AGE65', 'EP_DISABL', 'EP_LIMENG', 'EP_MOBILE', 'EP_GROUPQ'
]

ALL = ENV + SES
ALL_RAW = ENV_RAW + SES_RAW

# Define default feature groups
default_feature_groups = {
    "all": {
        "name": "All Features",
        "short_code": "all",
        "columns": ALL
    },
    "all_raw": {
        "name": "All Features (Raw)",
        "short_code": "all_raw",
        "columns": ALL_RAW
    },
    "env": {
        "name": "Environment",
        "short_code": "env",
        "columns": ENV
    },
    "env_raw": {
        "name": "Environment (Raw)",
        "short_code": "env_raw",
        "columns": ENV_RAW
    },
    "ses": {
        "name": "Socioeconomic Status",
        "short_code": "ses",
        "columns": SES
    },
    "ses_raw": {
        "name": "Socioeconomic Status (Raw)",
        "short_code": "ses_raw",
        "columns": SES_RAW
    },
    "air_water": {
        "name": "Air and Water Quality",
        "short_code": "air_water",
        "columns": [
            "EPL_PM25", "EPL_OZONE", "EPL_DSLPM", "EPL_WATER", "EPL_AIR", "EPL_PM10", "EPL_PM2_5", "EPL_NO2", "EPL_SO2", "EPL_CO"
        ]
    },
    "toxic_site": {
        "name": "Toxic Sites",
        "short_code": "toxic_site",
        "columns": [
            "EPL_NPL", "EPL_RMP", "EPL_TSDF", "EPL_LDPNT", "EPL_LEAD"
        ]
    },
    "built_env": {
        "name": "Built Environment",
        "short_code": "built_env",
        "columns": [
            "EPL_PARK", "EPL_GREEN", "EPL_WALK", "EPL_BIKE", "EPL_FOOD"
        ]
    },
    "transport": {
        "name": "Transportation",
        "short_code": "transport",
        "columns": [
            "EPL_TRANS", "EPL_VEH", "EPL_COMMUTE", "EPL_TRANSIT"
        ]
    },
    "pop": {
        "name": "Population",
        "short_code": "pop",
        "columns": [
            "EPL_POP", "EPL_AGE65", "EPL_AGE17", "EPL_MINRTY", "EPL_LIMENG"
        ]
    },
    "household": {
        "name": "Household",
        "short_code": "household",
        "columns": [
            "EPL_MLTPL", "EPL_MOBILE", "EPL_CROWD", "EPL_NOVEH", "EPL_GROUPQ"
        ]
    },
    "housing": {
        "name": "Housing",
        "short_code": "housing",
        "columns": [
            "EPL_HBURDEN", "EPL_RENTBURDEN", "EPL_OWNBURDEN", "EPL_UNITS", "EPL_AGE"
        ]
    }
}

# Save default_feature_groups to session state and to disk
st.session_state["default_feature_groups"] = default_feature_groups
default_config_path = os.path.join(config_dir, "default_feature_groups.json")
with open(default_config_path, "w") as f:
    json.dump(default_feature_groups, f, indent=4)

# Load existing feature groups
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        feature_groups = json.load(f)
else:
    feature_groups = {}

st.session_state["feature_groups"] = feature_groups

st.header("Default Feature Groups")

default_summary_data = []
for g in default_feature_groups.values():
    default_summary_data.append({
        "Name": g["name"],
        "Short Code": g["short_code"],
        "Columns": ", ".join(g["columns"])
    })

default_summary_df = pd.DataFrame(default_summary_data)
st.markdown(
    default_summary_df.to_html(escape=False, index=False),
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    table {
        width: 100%;
        table-layout: fixed;
        word-wrap: break-word;
    }
    th, td {
        white-space: normal !important;
        word-wrap: break-word !important;
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.header("Create New Feature Group")

with st.form("new_group_form"):
    new_name = st.text_input("Group Name")
    new_short_code = st.text_input("Short Code")
    selectable_columns = [col for col in zip_df.columns if col != "ZIPCODE"]
    new_columns = st.multiselect("Select Columns", options=selectable_columns)
    submitted = st.form_submit_button("Save Group")

if submitted:
    if not new_name.strip():
        st.error("Group Name cannot be empty.")
    elif not new_short_code.strip():
        st.error("Short Code cannot be empty.")
    elif not new_columns:
        st.error("Please select at least one column.")
    else:
        # Use short_code as key to avoid duplicates
        key = new_short_code.strip()
        if key in feature_groups:
            st.error(f"A group with short code '{key}' already exists.")
        else:
            feature_groups[key] = {
                "name": new_name.strip(),
                "short_code": key,
                "columns": new_columns
            }
            with open(config_path, "w") as f:
                json.dump(feature_groups, f, indent=4)
            st.session_state["feature_groups"] = feature_groups
            st.success(f"Feature group '{new_name}' saved.")
            st.rerun()

st.header("Custom Feature Groups")

if feature_groups:
    summary_data = []
    for g in feature_groups.values():
        summary_data.append({
            "Name": g["name"],
            "Short Code": g["short_code"],
            "Columns": ", ".join(g["columns"])
        })
    summary_df = pd.DataFrame(summary_data)
    st.markdown(
        summary_df.to_html(escape=False, index=False),
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <style>
        table {
            width: 100%;
            table-layout: fixed;
            word-wrap: break-word;
        }
        th, td {
            white-space: normal !important;
            word-wrap: break-word !important;
            text-align: left;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("### Manage Feature Groups")

    # Layout delete buttons in rows with wrapping using columns
    keys = list(feature_groups.keys())
    max_cols = 5  # Number of columns per row, adjust as needed

    for i in range(0, len(keys), max_cols):
        cols = st.columns(min(max_cols, len(keys) - i))
        for idx, key in enumerate(keys[i:i+max_cols]):
            group = feature_groups[key]
            if cols[idx].button(f"Delete '{group['name']}'", key=f"delete_{key}"):
                del feature_groups[key]
                with open(config_path, "w") as f:
                    json.dump(feature_groups, f, indent=4)
                st.session_state["feature_groups"] = feature_groups
                st.success(f"Feature group '{group['name']}' deleted.")
                st.rerun()
else:
    st.info("No feature groups to display.")
