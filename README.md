# Context-Aware Patient Similarity Networks

A Streamlit application for building and analyzing **graph-based similarity networks** from clinical and neighborhood data. The framework has two interconnected pipelines:

1. **Neighborhood Graph** — constructs a ZIP-code (or census tract) similarity network from environmental and socioeconomic context data (EPA EJSCREEN indices), identifies geographic communities, and characterizes how neighborhood conditions cluster.

2. **Patient Similarity Network (PSN)** — collapses patients into feature profiles, builds a k-nearest-neighbor similarity graph enriched with neighborhood context, applies Louvain community detection to identify patient subgroups, and characterizes each subgroup through a signature matrix.

The two pipelines share data: neighborhood community assignments can be fed as features into the PSN, allowing patient groupings to reflect both clinical characteristics and where patients live.

---

## Requirements

```
Python >= 3.10
streamlit >= 1.37
pandas >= 2.1
numpy >= 1.26
networkx >= 3.2
matplotlib >= 3.8
scikit-learn >= 1.4
python-louvain >= 0.16
pyyaml >= 6.0
geopandas >= 0.14
shapely >= 2.0
plotly >= 5.20
pydeck >= 0.9
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

Optional — enables approximate nearest-neighbor search for large graphs (recommended for census tract context with 70k+ rows):

```bash
pip install pynndescent
```

---

## Quick Start

```bash
streamlit run app/Home.py
```

Runs at `http://localhost:8501` by default.

---

## Data

### Required Files

Data paths are configured in `configs/default.yaml`. The app expects:

| Dataset | Description |
|---|---|
| Patient data (visit level) | One row per hospital visit — demographics, comorbidities, utilization, outcomes |
| Patient data (patient level) | One row per patient — features aggregated across all visits |
| ZIP context | EPA EJSCREEN indices (environmental + socioeconomic) per ZIP code |
| ZIP coordinates | ZIP code lat/lng lookup (`zip`, `lat`, `lng` columns) |
| WA state boundary | US Census TIGER shapefile for geographic overlay |

Optionally, census tract context can be added as a second context dataset alongside ZIP codes.

### Patient Data Columns

The app expects these columns (absent columns are silently skipped):

| Column | Type | Description |
|---|---|---|
| `PATIENTID` | string | Patient identifier (patient-level data) |
| `ZIPCODE` | string | Patient ZIP code — used to join neighborhood context |
| `SEX` | categorical | `F` / `M` |
| `Race` | categorical | Race category |
| `AGE_BIN` | categorical | Age group (`<65`, `65-69`, `70-74`, `75-79`, `80-84`, `85-89`, `>=90`) |
| `LENSTAYD` | numeric | Length of stay in days |
| `LENSTAYD_BIN` | categorical | Binned length of stay (`Short Stay`, `Medium Stay`, …) |
| `LENSTAYD_LOG` | numeric | Log-transformed length of stay |
| `PAYER` | categorical | Payer type (Medicare, Medicaid, Private, etc.) |
| `NUM_VISITS` | integer | Number of visits per patient |
| `REVISIT_30` | binary | 1 if a revisit occurred within 30 days |
| `READMIT_COUNT` | integer | Total 30-day revisits across all visits |
| `READMIT_RATE` | float | `READMIT_COUNT / max(NUM_VISITS − 1, 1)` |
| `EVER_READMITTED` | binary | 1 if patient was ever readmitted |
| `Hearingloss` | binary | 1 if diagnosed |
| `BrainInjury` | binary | 1 if diagnosed |
| `Hypertension` | binary | 1 if diagnosed |
| `Alcohol` | binary | 1 if diagnosed |
| `Obesity` | binary | 1 if diagnosed |
| `Diabetes` | binary | 1 if diagnosed |

### Neighborhood Context Columns

ZIP or census tract context data uses EPA EJSCREEN normalized percentile indices (`EPL_*`) and raw counts (`EP_*` / `E_*`). The app expects columns from the environmental block (`EPL_OZONE`, `EPL_PM`, `EPL_DSLPM`, etc.) and socioeconomic block (`EPL_MINRTY`, `EPL_POV200`, `EPL_NOHSDP`, etc.). See `app/_logic/loader.py` for the full column lists.

### Configuration

Edit `configs/default.yaml` to register datasets:

```yaml
paths:
  wa_state_zip: data/raw/cb_2020_us_state_20m.zip

patient_datasets:
  patient_level:
    label: "Patient Level"
    path: data/processed/patients_patient_level.csv
  visit_level:
    label: "Visit Level"
    path: data/processed/patients_processed.csv

context_datasets:
  zip_codes:
    label: "ZIP Codes"
    zip_context: data/raw/zip_context.csv
    zip_coords: data/raw/uszips.csv
  census_tracts:
    label: "Census Tracts"
    zip_context: data/raw/census_context.csv
    zip_coords: data/raw/uscensus.csv
```

Multiple datasets can be registered under each section. The active dataset is switchable via a sidebar dropdown without restarting the app.

---

## App Walkthrough

### Home — Data Overview

Loads all configured datasets into session state and displays summary statistics:
- Patient demographics, utilization, and risk factor prevalence
- ZIP context environmental and socioeconomic summaries

The **Reload Data** button refreshes datasets without restarting the app.

---

## Pipeline 1 — Neighborhood Graph

Pages 01–05 build and analyze the neighborhood context network.

### 01 — Neighborhood Features

Configure which feature groups from the ZIP or census context data to use. Predefined groups:

| Group | Variables |
|---|---|
| `all` | All EPA EJSCREEN normalized indices (environmental + socioeconomic) |
| `env` | Environmental indices — ozone, particulate matter, diesel, superfund sites, etc. |
| `ses` | Socioeconomic indices — poverty, unemployment, uninsured rate, etc. |
| `all_raw` / `env_raw` / `ses_raw` | Same groups using raw counts rather than normalized percentiles |

Groups are saved to `data/config/feature_groups.json` and persist across sessions.

### 02 — Build Neighborhood Index

Builds a reusable approximate k-NN index over ZIP codes (or census tracts) using the selected feature group. This is a one-time expensive computation (can take several minutes for 70k+ census tracts). Once cached, different k values can be explored instantly on the next page.

- Similarity metric: cosine similarity over standardized feature vectors
- Backend: PyNNDescent (approximate) or exact cosine, depending on dataset size
- Index cached to `data/cache/nbr_index/`

### 03 — Neighborhood Graph

Constructs and visualizes the neighborhood similarity graph from the cached index.

- **Graph construction:** k-NN edges with configurable k; supports mutual (undirected) and directed variants
- **Community detection:** Louvain algorithm at resolution = 1.0
- **Visualizations:**
  - 2D scatter (PCA or feature-based coloring)
  - Network graph (spring layout)
  - Geographic choropleth — communities overlaid on a Washington state map
- Graphs cached to `data/cache/neighborhood_graph/`

Community assignments from this step can optionally be used as a feature in the PSN (page 06).

### 04 — Feature Group Comparison

Compares how different neighborhood feature group choices (ENV vs. SES vs. ALL) affect community structure. Side-by-side heatmaps and statistical summaries help select the most informative configuration.

### 05 — K Sensitivity Analysis (Neighborhood)

Plots how the neighborhood graph's community count and modularity change across a range of k values. Helps identify a stable, robust k before committing to a specific graph.

---

## Pipeline 2 — Patient Similarity Network (PSN)

Pages 06–10 build, analyze, and visualize the patient similarity network.

### 06 — PSN Feature Selection

Configures which patient features enter the PSN and encodes them into a numeric matrix.

**Feature groups:**

| Group | Columns |
|---|---|
| Demographics | `SEX`, `Race`, `AGE_BIN` |
| Utilization | `LENSTAYD_BIN`, `LENSTAYD_LOG`, `PAYER`, `NUM_VISITS`, `REVISIT_30` |
| Risk binaries | `Hearingloss`, `BrainInjury`, `Hypertension`, `Alcohol`, `Obesity`, `Diabetes` |
| Outcomes | `READMIT_COUNT`, `READMIT_RATE`, `EVER_READMITTED` |

Neighborhood context features (ZIP degree, PageRank, betweenness, and community assignment) can also be added from the neighborhood pipeline.

**Profiles:**

Rather than one node per patient, patients are first collapsed into **profiles** — unique combinations of the selected feature values. `profile_count` records how many patients share each profile. This reduces the graph to a manageable number of nodes while preserving the distribution of patients across the feature space.

**Encoding:**

- **Standard (default):** One-hot for categorical columns; binary risk columns kept as 0/1; continuous columns (LENSTAYD_LOG, NUM_VISITS, etc.) standardized with `StandardScaler`.
- **Experimental:** Integer encoding for categoricals, bitflag encoding for all comorbidities packed into a single integer. Reduces dimensionality significantly; enables a custom mixed similarity metric on page 07.

Neighborhood context is joined to each profile by weighted averaging over the ZIP codes of patients in that profile.

### 07 — PSN Graph

Builds the Patient Similarity Network and runs community detection.

**Steps:**

1. **Block weighting** — patient features and neighborhood features are weighted independently (slider: 0.0 = patient features only, 1.0 = neighborhood features only).
2. **Similarity computation** — cosine similarity between profile feature vectors. Uses PyNNDescent (approximate) for large graphs, exact cosine for smaller ones. A custom mixed similarity metric is available in experimental encoding mode.
3. **k-NN graph** — edges connect each profile to its k most similar profiles. Supports `mutual` (both profiles must be in each other's top-k) and `directed` variants.
4. **Community detection** — Louvain algorithm, resolution fixed at 1.0 (see design note below).
5. **Graph metrics** — betweenness centrality, PageRank, and degree computed per node.

**Outputs:**

- PCA 2D scatter plot (colored by community, payer, sex, age, or race)
- Network graph visualization (spring layout)
- Similarity matrix saved to `data/cache/patient_graphs/`
- Community assignments written to the features table

**Design note — resolution = 1.0:**
The Louvain resolution parameter is fixed at 1.0 across all analyses. Adjusting resolution to target a preferred number of communities would be equivalent to post-hoc selection; there is no clinical justification for any particular n. At 1.0, the number and structure of communities is entirely determined by the data and graph construction parameters.

### 08 — PSN Analysis

Characterizes the communities found in the PSN. The analysis focuses on the **top N communities by patient count** (configurable percentage, default: top 10%).

**Configuration:**

- **Min patients per community** — communities below this threshold are excluded from analysis (default: 0.1% of total patients; excluded communities are documented but not deleted).
- **Top N% communities** — focus on the N% largest communities by patient count.

**Signature matrix:**

For each community, a row of summary statistics is computed — all weighted by `profile_count` so larger profiles contribute proportionally:

| Feature type | Statistic |
|---|---|
| Binary risk columns | Prevalence % |
| SEX | Female % |
| Multi-category columns (Race, AGE_BIN, PAYER, LENSTAYD_BIN) | Per-category % for every category |
| Continuous columns (LENSTAYD_LOG, NUM_VISITS, READMIT_COUNT, READMIT_RATE, REVISIT_30) | Weighted mean + SD |

**Outputs (all downloadable as CSV or PNG):**

- Signature heatmap (z-scored, ordered by Ward hierarchical linkage) — top-N communities shown by default, all communities available in an expander
- Top-N community signatures table (sorted by patient count)
- All community signatures table
- Outlier communities — any community with |z-score| > 2.0 on any feature relative to the top-N grand mean

### 09 — PSN K Sensitivity

Tests how the PSN community structure changes as k varies. Plots community count and modularity across a range of k values to guide the choice of k.

### 10 — PSN Publication Plots

Publication-ready figures derived from the PSN analysis results. All plots are generated on demand (button click). Uses the top-N communities from page 08.

**Tab 1 — Network Graph:**
Spring-layout visualization of the full PSN. Top-N communities are highlighted in distinct colors; remaining communities appear in grey. Node size reflects patient count. A slider caps the number of profiles displayed (profiles are sampled proportionally from each community to stay within the limit).

**Tab 2 — Feature Prevalences:**
Horizontal bar charts showing % prevalence (or mean value) per community, sorted by patient count. Feature groups can be filtered by prefix.

**Tab 3 — Size vs. Outcome:**
Bubble chart — x-axis: community patient count; y-axis: any outcome metric from the signature (e.g. `READMIT_RATE_mean`, `REVISIT_30_mean`); bubble size proportional to patient count. Community IDs are labelled on each bubble.

**Tab 4 — Payer Mix:**
Stacked bar chart showing the proportion of each payer category per community. Horizontal or vertical orientation, sorted by patient count.

---

## Project Structure

```
graph-dementia/
├── app/
│   ├── Home.py                          # Data overview and loader
│   ├── _logic/
│   │   ├── loader.py                    # Dataset loading and session state management
│   │   ├── config.py                    # Config file loading (configs/default.yaml)
│   │   ├── encoding.py                  # Integer and bitflag encoding utilities
│   │   ├── graph_cache.py               # Neighborhood graph caching utilities
│   │   └── psn_graph_builder.py         # PSN similarity, k-NN, Louvain, graph metrics
│   ├── _components/
│   │   ├── plots.py                     # Shared plotting helpers
│   │   └── zip_context_utils.py         # ZIP context join and PCA utilities
│   └── pages/
│       ├── 01_Neighborhood_Features.py  # Feature group configuration
│       ├── 02_Build_Neighborhood_Index.py  # k-NN index construction and caching
│       ├── 03_Neighborhood_Graph.py     # Neighborhood graph and community detection
│       ├── 04_Feature_Group_Comparison.py  # Compare feature group configurations
│       ├── 05_K_Sensitivity_Analysis.py    # Neighborhood k sensitivity
│       ├── 06_PSN_Feature_Selection.py  # Feature encoding and profile matrix
│       ├── 07_PSN_Graph.py              # PSN construction and community detection
│       ├── 08_PSN_Analysis.py           # Signature analysis, heatmap, top-N communities
│       ├── 09_PSN_K_Sensitivity.py      # PSN k sensitivity
│       └── 10_PSN_Publication_Plots.py  # Publication-ready figures
├── configs/
│   └── default.yaml                     # Data paths and dataset registry
├── data/
│   ├── raw/                             # Source data (not committed)
│   ├── processed/                       # Preprocessed patient CSVs (not committed)
│   ├── cache/                           # Computed graphs, indices, and figures (not committed)
│   └── config/
│       ├── default_feature_groups.json  # Canonical ZIP feature group definitions
│       ├── feature_groups.json          # User-edited feature groups (persisted by app)
│       └── psn_feature_groups.json      # PSN-specific feature group configuration
├── docs/
│   ├── community_analysis_framework.md  # Design decisions for community analysis
│   ├── psn_analysis_reference.md        # Reference guide for interpreting PSN outputs
│   └── methodology/                     # Technical methodology document (LaTeX + PDF)
├── archive/                             # Retired pages (not in active pipeline)
├── outputs/                             # Figures, tables, logs (not committed)
└── requirements.txt
```

---

## Session State Flow

Data flows forward through the pipeline via Streamlit session state. Each page depends on outputs from earlier pages:

```
patients_df / zip_df / zip_coords        (Home — ensure_data_loaded)
        ↓
feature_groups                           (page 01 — Neighborhood Features)
        ↓
nbr_index                                (page 02 — Build Neighborhood Index)
        ↓
neighborhood_graph + zip_community       (page 03 — Neighborhood Graph)
        ↓
pf_fused_matrix / pf_fused_table         (page 06 — PSN Feature Selection)
        ↓
patient_graph_cache                      (page 07 — PSN Graph)
        ↓
psn_analysis_results                     (page 08 — PSN Analysis)
        ↓
Publication plots                        (page 10)
```

The graph cache uses a key derived from k, knn_type, patient weight, and zip weight. Only re-runs with different parameters trigger recomputation. Changing upstream inputs (e.g. re-running feature selection with different columns) invalidates downstream cached results.
