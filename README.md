# Context-Aware Patient Similarity Networks (graph-dementia)

A Streamlit application that builds **Patient Similarity Networks (PSNs)** from clinical and demographic data, enriched with neighborhood-level environmental and socioeconomic context. The app supports community detection, signature analysis, and publication-ready visualization — designed for a dementia-adjacent patient population.

---

## Overview

The core idea is to represent patients as nodes in a graph, where edges connect patients who are clinically similar. By grouping patients into profiles (combinations of shared features) and enriching those profiles with ZIP-code context (EPA environmental justice indices), the app constructs a k-nearest-neighbor similarity graph and applies Louvain community detection to identify patient subgroups.

Each community is then characterized by a **signature** — a compact summary of the feature distributions within that community — enabling interpretable comparison across groups.

This approach is related to bipartite network analysis (Bhavnani et al. 2022, 2023) but uses a **unipartite k-NN graph** with Louvain community detection rather than bipartite modularity maximization. See `docs/literature_comparison.md` for a detailed comparison.

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

Optional (for approximate nearest-neighbor search on large graphs):

```bash
pip install pynndescent
```

---

## Quick Start

```bash
streamlit run app/Home.py
```

The app runs at `http://localhost:8501` by default.

---

## Data Setup

### Required Files

Place the following in the paths configured in `configs/default.yaml`:

| File | Default location | Description |
|---|---|---|
| Patient data (visit level) | `data/processed/patients_processed.csv` | Cleaned visit-level records |
| Patient data (patient level) | `data/processed/patients_patient_level.csv` | One row per patient, aggregated |
| ZIP context | `data/raw/zip_context.csv` | EPA EJSCREEN indices per ZIP code |
| ZIP coordinates | `data/raw/uszips.csv` | ZIP code lat/lng lookup |
| WA state boundary | `data/raw/cb_2020_us_state_20m.zip` | US Census TIGER shapefile |

Optional alternative context dataset:

| File | Default location | Description |
|---|---|---|
| Census tract context | `data/raw/census_context.csv` | Census-tract-level EPA indices |
| Census coordinates | `data/raw/uscensus.csv` | Census tract lat/lng |

### Preparing Patient Data

Raw patient data lives in `data/raw/`. Two preprocessing scripts convert raw exports to the format expected by the app.

**Step 1 — Process raw visits:**

```bash
python scripts/process_patient_data.py
```

Reads `data/raw/patients_long.csv` and writes `data/processed/patients_processed.csv`. Creates derived columns:

- `AGE_BIN` — age groups: `<65`, `65-69`, `70-74`, `75-79`, `80-84`, `85-89`, `>=90`
- `LENSTAYD_BIN` — length-of-stay bins: Short / Medium / Long / Extended / Very Long
- `LENSTAYD_LOG` — log-transformed length of stay
- `REVISIT_30` — binary flag: did this patient have a revisit within 30 days?

**Step 2 — Aggregate to patient level:**

```bash
python scripts/aggregate_to_patient_level.py
```

Reads `data/processed/patients_processed.csv` and writes `data/processed/patients_patient_level.csv`. One row per patient. Aggregation rules:

| Column | Method |
|---|---|
| Risk binaries (Hypertension, Diabetes, etc.) | `max` — any positive visit counts |
| REVISIT_30 | `sum` → renamed `READMIT_COUNT` |
| READMIT_RATE | `READMIT_COUNT / max(NUM_VISITS − 1, 1)` |
| Demographics (SEX, Race) | `mode` |
| NUM_VISITS | `count` of visits |

Or run the full pipeline at once:

```bash
bash scripts/update_patient_data.sh
```

---

## Configuration

The app is configured via `configs/default.yaml`:

```yaml
paths:
  wa_state_zip: data/raw/cb_2020_us_state_20m.zip

patient_datasets:
  patient_level:
    label: "Patient Level (84,665)"
    path: data/processed/patients_patient_level.csv
  visit_level:
    label: "Visit Level (135,096)"
    path: data/processed/patients_processed.csv

context_datasets:
  zip_codes:
    label: "ZIP Codes (673)"
    zip_context: data/raw/zip_context.csv
    zip_coords: data/raw/uszips.csv
  census_tracts:
    label: "Census Tracts (71k)"
    zip_context: data/raw/census_context.csv
    zip_coords: data/raw/uscensus.csv
```

Multiple patient datasets and context datasets can be registered. The active dataset is switchable via a sidebar dropdown without restarting the app.

---

## App Walkthrough (Pipeline)

The app is structured as a sequential pipeline. Pages are numbered to indicate the recommended order of operation.

### Home — Data Overview

Loads all configured datasets and displays data summaries:
- Patient demographics and utilization statistics
- Risk factor prevalence (binary columns)
- ZIP context environmental and socioeconomic summaries

Use the **Reload Data** button to refresh datasets without restarting.

---

### 01 — Neighborhood Features

Configure which ZIP-code feature groups to use for the neighborhood context layer. Predefined groups include:

| Group | Variables |
|---|---|
| `all` | All EPA EJSCREEN normalized indices (ENV + SES) |
| `env` | Environmental indices only (ozone, PM, diesel, NPL, TRI, etc.) |
| `ses` | Socioeconomic indices only (poverty, unemployment, uninsured, etc.) |
| `all_raw` / `env_raw` / `ses_raw` | Same groups using raw (non-normalized) values |

Feature groups are saved to `data/config/feature_groups.json` and persist across sessions.

---

### 02 — Build Neighborhood Index

Builds a k-nearest-neighbor index over ZIP codes using the selected feature group. Supports multiple feature group selections for comparison.

- Uses cosine similarity over standardized ZIP-code feature vectors
- Results are cached to `data/cache/nbr_index/`
- Configurable k (default: k=30)

---

### 03 — Neighborhood Graph

Visualizes the ZIP-code similarity graph constructed from the neighborhood index.

- Graph layout: spring (Fruchterman-Reingold) or geographic (lat/lng)
- Node color: Louvain community assignment
- Node size: number of patients in that ZIP code
- Outputs: network figure (PNG), geographic choropleth (interactive)

Graphs are cached to `data/cache/neighborhood_graph/`.

---

### 04 — Feature Group Comparison

Compares community structure across different neighborhood feature groups (e.g. ENV vs. SES vs. ALL). Shows side-by-side heatmaps and summary statistics to help select the most informative feature configuration.

---

### 05 — K Sensitivity Analysis (Neighborhood)

Tests how sensitive the neighborhood graph community structure is to the choice of k in the k-NN construction. Plots the number of communities, modularity, and feature group rankings as k varies.

---

### 06 — PSN Feature Selection

The entry point for the Patient Similarity Network pipeline. Configures which patient features go into the PSN.

**Feature groups available:**

| Group | Columns |
|---|---|
| Demographics | `SEX`, `Race`, `AGE_BIN` |
| Utilization | `LENSTAYD_BIN`, `LENSTAYD_LOG`, `PAYER`, `NUM_VISITS`, `REVISIT_30` |
| Risk binaries | `Hearingloss`, `BrainInjury`, `Hypertension`, `Alcohol`, `Obesity`, `Diabetes` |
| Outcomes | `READMIT_COUNT`, `READMIT_RATE`, `EVER_READMITTED` |

**Encoding:**

- **Standard (default):** One-hot encoding for categorical columns; binary columns kept as 0/1; continuous columns (LENSTAYD_LOG, NUM_VISITS, etc.) standardized with `StandardScaler`.
- **Experimental:** Integer encoding for categoricals, bitflag encoding for comorbidities (packs all binary risk columns into a single integer). Reduces dimensionality significantly.

**Profiles:**

Patients are collapsed into **profiles** — unique combinations of the selected features. Each profile becomes one node in the PSN. `profile_count` records how many patients share each profile.

Neighborhood context (ZIP-code features) is joined to each profile via weighted averaging over the ZIP codes of patients in that profile.

The resulting encoded matrix is stored in session state and passed to page 07.

---

### 07 — PSN Graph

Builds and visualizes the Patient Similarity Network.

**Graph construction:**

1. **Block weighting** — patient features and neighborhood features are weighted independently (slider: 0% = patient only, 100% = neighborhood only).
2. **Similarity computation** — cosine similarity between profile feature vectors. Uses `PyNNDescent` (approximate nearest neighbors) for large graphs, exact cosine for smaller ones.
3. **k-NN graph** — edges connect each profile to its k most similar profiles. Supports `mutual` (edge only if both profiles are in each other's top-k) and `directed` variants.
4. **Community detection** — Louvain algorithm at resolution = 1.0 (fixed; see design rationale below).
5. **Graph metrics** — betweenness centrality, PageRank, degree computed per node.

**Outputs:**

- 2D PCA scatter plot (colored by community, payer, sex, age, or race)
- Network graph figure (spring layout, saved to `data/cache/patient_figs/`)
- Similarity matrix saved to `data/cache/patient_graphs/`
- Community assignments written to the features table

**Design rationale — resolution = 1.0:**
Louvain resolution is fixed at 1.0 across all analyses. Tuning resolution to target a preferred number of communities would amount to post-hoc selection; there is no clinical justification for any particular n. The community count is entirely data-driven.

---

### 08 — PSN Analysis

Characterizes the communities found in the PSN. The analysis focuses on the **top N communities by patient count** (configurable as a percentage; default: top 10%).

**Configuration:**

- **Min patients per community** — communities below this threshold are excluded (default: 0.1% of total patients).
- **Top N% communities** — focus analysis on the N% largest communities; default 10%.

**Signature matrix:**

For each community, a row of summary statistics is computed:

| Feature type | Statistic |
|---|---|
| Binary risk columns (Hypertension, etc.) | Prevalence % (weighted by profile count) |
| SEX | Female % |
| Multi-category (Race, AGE_BIN, PAYER, LENSTAYD_BIN) | Per-category % for every category |
| Continuous (LENSTAYD_LOG, NUM_VISITS, READMIT_COUNT, READMIT_RATE, REVISIT_30) | Weighted mean + SD |

All means are weighted by `profile_count` so profiles with more patients contribute proportionally.

**Outputs (all downloadable):**

- **Signature heatmap** — z-scored community signatures ordered by Ward hierarchical linkage; top-N communities shown by default, all communities available in an expander.
- **Top-N community signatures table** — sorted by patient count.
- **All community signatures table** — full table for all included communities.
- **Outlier communities** — communities with |z| > 2.0 on any feature relative to the top-N grand mean.

Results are cached in session state. Clicking **Run Analysis** again clears and recomputes.

---

### 09 — PSN K Sensitivity

Assesses how the PSN community structure changes as k varies. Plots community count and modularity across a range of k values to help choose a robust k.

---

### 10 — PSN Publication Plots

Publication-ready figures derived from the PSN analysis. All plots are on-demand (click a button to generate). Uses the top-N communities from page 08.

**Tab 1 — Network Graph:**
Spring-layout visualization of the PSN. Nodes colored by community; top-N communities highlighted in distinct colors, remaining communities in grey. Node size reflects patient count. A slider caps the number of profiles displayed (proportional sampling per community).

**Tab 2 — Feature Prevalences:**
Horizontal bar charts showing % prevalence or mean value per community. Filter by feature group prefix. Continuous features (mean) shown on a separate panel.

**Tab 3 — Size vs. Outcome:**
Bubble chart: x-axis = community patient count, y-axis = any outcome metric (READMIT_RATE_mean, REVISIT_30_mean, etc.), bubble size proportional to patient count. Labels show community IDs.

**Tab 4 — Payer Mix:**
Stacked bar chart showing the proportion of each payer category per community (PAYER_*_pct columns). Horizontal or vertical orientation.

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
│   │   └── zip_context_utils.py         # ZIP context join utilities
│   └── pages/
│       ├── 01_Neighborhood_Features.py  # Feature group configuration
│       ├── 02_Build_Neighborhood_Index.py
│       ├── 03_Neighborhood_Graph.py
│       ├── 04_Feature_Group_Comparison.py
│       ├── 05_K_Sensitivity_Analysis.py
│       ├── 06_PSN_Feature_Selection.py  # Feature encoding and profile matrix
│       ├── 07_PSN_Graph.py              # PSN construction and community detection
│       ├── 08_PSN_Analysis.py           # Signature analysis and heatmap
│       ├── 09_PSN_K_Sensitivity.py
│       └── 10_PSN_Publication_Plots.py  # Publication-ready figures
├── configs/
│   └── default.yaml                     # Data paths and dataset registry
├── data/
│   ├── raw/                             # Source data (not committed)
│   ├── processed/                       # Preprocessed patient CSVs (not committed)
│   ├── cache/                           # Computed graphs and figures (not committed)
│   └── config/
│       ├── default_feature_groups.json  # Canonical ZIP feature group definitions
│       ├── feature_groups.json          # User-edited feature groups (persisted by app)
│       └── psn_feature_groups.json      # PSN-specific feature group configuration
├── scripts/
│   ├── process_patient_data.py          # Raw → visit-level processed CSV
│   ├── aggregate_to_patient_level.py    # Visit-level → patient-level CSV
│   ├── compare_patient_files.py         # Audit and compare patient dataset versions
│   └── update_patient_data.sh           # Full preprocessing pipeline script
├── docs/
│   ├── literature_comparison.md         # Comparison with Bhavnani et al. (2022, 2023)
│   ├── community_analysis_framework.md  # Design decisions for community analysis
│   ├── psn_analysis_reference.md        # Reference guide for interpreting PSN outputs
│   └── methodology/                     # LaTeX technical methodology document
├── archive/                             # Retired pages (not in active pipeline)
├── outputs/                             # Figures, tables, logs (not committed)
└── requirements.txt
```

---

## Key Technical Concepts

### Profiles

Rather than making each patient a node, the PSN first collapses patients into **profiles** — unique combinations of selected feature values. For example, all female, White, 70–74, Medicare patients with hypertension and no other comorbidities form one profile. `profile_count` records how many patients match that profile.

This reduces the graph from tens of thousands of patient nodes to hundreds or thousands of profile nodes, making similarity computation tractable.

### Community Detection (Louvain, resolution = 1.0)

Louvain community detection partitions the graph into subsets of profiles that are more densely connected internally than to the rest of the network. The resolution parameter (fixed at 1.0) controls the granularity: lower values produce fewer, larger communities; higher values produce more, smaller ones. Fixing it at 1.0 ensures results are fully data-driven.

### Signature Matrix

The signature matrix has one row per community and one column per summary statistic. It is the foundation for all downstream analysis: heatmap, outlier detection, and publication plots. All statistics are weighted by `profile_count`.

### Community Comparison (Top-N)

Rather than meta-clustering all communities into a predetermined number of groups, the analysis focuses on the **top N communities by patient count** (configurable percentage, default 10%). These are the communities that represent the most patients and are typically the most clinically relevant.

---

## Session State Flow

Session state is used to pass artifacts between pages. Each page depends on the previous:

```
patients_df / zip_df             (loaded by Home / ensure_data_loaded)
        ↓
feature_groups                   (configured on page 01)
        ↓
nbr_index + neighborhood graph   (built on pages 02–03)
        ↓
pf_fused_matrix / pf_fused_table (built on page 06 — feature selection)
        ↓
patient_graph_cache              (built on page 07 — PSN graph)
        ↓
psn_analysis_results             (computed on page 08 — PSN analysis)
        ↓
Publication plots                (page 10)
```

Changing any upstream page (e.g. re-running feature selection with different columns) will invalidate downstream cached results. The graph cache uses a hash key derived from k, knn_type, patient weight, and zip weight, so only re-runs with different parameters trigger recomputation.

---

## Uncommitted Files

The following are currently tracked by git but unstaged, or are new and untracked:

**Modified (not staged):**
- `app/pages/06_PSN_Feature_Selection.py` — NAType bug fix (`X_fused.to_numpy`)
- `app/pages/08_PSN_Analysis.py` — Replaced meta-clustering with top-N community analysis
- `data/config/psn_feature_groups.json` — Feature group config updates

**Deleted (needs staging):**
- `app/pages/10_PSN_Feature_Groups.py` → moved to `archive/`
- `app/pages/11_PSN_Feature_Group_Comparison.py` → moved to `archive/`

**Untracked (new):**
- `app/pages/10_PSN_Publication_Plots.py` — New publication plots page
- `archive/` — Retired pages
- `docs/` — Literature comparison, methodology, PSN reference guide
- `scripts/compare_patient_files.py` — Dataset audit utility
- `scripts/process_patient_data.py` — Raw data preprocessing
- `scripts/update_patient_data.sh` — Full pipeline script
- `.claude/` — Claude Code session memory (not for commit)

---

## Related Literature

This work draws on and extends methods from:

1. **Bhavnani et al. (2022)** — "A Framework for Modeling and Interpreting Patient Subgroups Applied to Hospital Readmission: Visual Analytical Approach." *JMIR Medical Informatics* 10(12):e37239.

2. **Bhavnani et al. (2023)** — "Subtyping Social Determinants of Health in All of Us: Network Analysis and Visualization Approach." *medRxiv* preprint. doi:10.1101/2023.01.27.23285125

Both papers use bipartite network analysis (patients and features as separate node types). This app uses a unipartite k-NN PSN — see `docs/literature_comparison.md` for a detailed comparison of methods, overlaps, and key differences.

---

## Citation Language

If citing this work's methodology:

> Patient subgroup identification using graph-based network analysis has been demonstrated in clinical populations, including hospital readmission studies (Bhavnani et al. 2022) and social determinants of health subtyping (Bhavnani et al. 2023). Our approach extends this tradition by using a unipartite k-nearest-neighbour Patient Similarity Network with Louvain community detection, applied to a broader feature set encompassing demographics, comorbidities, healthcare utilization, and payer information, enriched with ZIP-code-level environmental and socioeconomic context.
