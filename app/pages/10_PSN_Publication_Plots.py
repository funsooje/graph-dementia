# app/pages/10_PSN_Publication_Plots.py
"""
Publication-ready plots drawn from the PSN analysis results.
Requires page 07 (PSN Graph) and page 08 (PSN Analysis) to have been run first.
"""
import math
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
from io import BytesIO

st.set_page_config(page_title="PSN Publication Plots", layout="wide")
st.title("PSN Publication Plots")

# ---------------------------------------------------------------------
# Data checks
# ---------------------------------------------------------------------
from app._logic.loader import ensure_data_loaded

if not ensure_data_loaded():
    st.stop()

if "patient_graph_cache" not in st.session_state or not st.session_state["patient_graph_cache"]:
    st.info("No PSN graph found. Run page 07 (PSN Graph) first.")
    st.stop()

if "psn_analysis_results" not in st.session_state:
    st.info("No analysis results found. Run page 08 (PSN Analysis) first.")
    st.stop()

graph_cache = st.session_state["patient_graph_cache"]
graph_cache_key = st.session_state.get(
    "active_psn_graph_key", list(graph_cache.keys())[-1]
)
if graph_cache_key not in graph_cache:
    graph_cache_key = list(graph_cache.keys())[-1]

G_full       = graph_cache[graph_cache_key]["graph"]
features_tbl = graph_cache[graph_cache_key]["features"].copy()

res             = st.session_state["psn_analysis_results"]
sig             = res["sig"]           # all included communities
top_sig         = res["top_sig"]       # top-N communities
top_communities = res["top_communities"]
n_communities   = res["n_communities"]
n_top           = res["n_top"]

st.caption(
    f"Graph: `{graph_cache_key}`  |  "
    f"{len(features_tbl):,} profiles  |  "
    f"{n_communities} communities  |  "
    f"Top {n_top} shown in plots"
)

# ---------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------
def _community_palette(n: int) -> list:
    cmap = plt.get_cmap("tab20" if n <= 20 else "hsv")
    return [mcolors.to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]


def _save_fig(fig) -> BytesIO:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "1 · Network Graph",
    "2 · Feature Prevalences",
    "3 · Size vs Outcome",
    "4 · Payer Mix",
])

# =====================================================================
# TAB 1 — PSN Network Graph (coloured by community, top-N highlighted)
# =====================================================================
with tab1:
    st.subheader("Patient Similarity Network — community view")
    st.markdown(
        "Nodes are **patient profiles**, coloured by Louvain community. "
        "Top-N communities (by patient count) are highlighted; remaining communities are grey."
    )

    col_a, col_b = st.columns([2, 1])
    with col_a:
        max_nodes = st.slider(
            "Max profiles to display",
            min_value=200, max_value=5000, value=1500, step=100,
        )
    with col_b:
        layout_iters = st.slider("Layout iterations", 20, 200, 60, step=10)

    plot_btn1 = st.button("Plot Network", key="plot_net", type="primary")

    if plot_btn1:
        with st.spinner("Sampling and laying out graph…"):
            feat = features_tbl.copy()
            weight_col = "profile_count" if "profile_count" in feat.columns else None

            top_set = set(top_communities)
            all_comms = sorted(feat["profile_community"].dropna().unique())
            palette = _community_palette(len(top_communities))
            top_color_map = {c: palette[i] for i, c in enumerate(top_communities)}

            # Sample proportionally from each community
            n_total = len(feat)
            if n_total > max_nodes:
                sampled = (
                    feat.groupby("profile_community", group_keys=False)
                    .apply(
                        lambda g: g.sample(
                            n=max(1, math.ceil(len(g) / n_total * max_nodes)),
                            random_state=42,
                        )
                    )
                    .head(max_nodes)
                )
            else:
                sampled = feat

            node_ids = set(sampled.index.tolist())
            G_sub = G_full.subgraph(
                [n for n in node_ids if n in G_full.nodes]
            ).copy()
            for n in node_ids:
                if n not in G_sub.nodes:
                    G_sub.add_node(n)

            k_layout = 1.0 / math.sqrt(max(len(G_sub), 1))
            pos = nx.spring_layout(G_sub, seed=42, k=k_layout, iterations=layout_iters)

            node_colors = []
            node_sizes  = []
            for n in G_sub.nodes:
                row = sampled.loc[n] if n in sampled.index else None
                comm = row["profile_community"] if row is not None else None
                color = top_color_map.get(comm, "#cccccc")
                node_colors.append(color)
                wt = float(row[weight_col]) if (row is not None and weight_col) else 1.0
                node_sizes.append(max(5, min(200, wt * 0.5)))

        fig, ax = plt.subplots(figsize=(12, 10))
        nx.draw_networkx_edges(G_sub, pos, ax=ax, alpha=0.08, width=0.4, edge_color="#888888")
        nx.draw_networkx_nodes(G_sub, pos, ax=ax,
                               node_color=node_colors, node_size=node_sizes, alpha=0.85)
        ax.axis("off")
        ax.set_title(
            f"Patient Similarity Network  ·  {len(G_sub)} profiles  ·  "
            f"Top {n_top} communities highlighted",
            fontsize=12, pad=12,
        )
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=top_color_map[c], markersize=8, label=str(c))
            for c in top_communities
        ]
        handles.append(
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor="#cccccc", markersize=8, label="Other")
        )
        ncol = max(1, len(handles) // 20 + 1)
        ax.legend(handles=handles, title=f"Top {n_top} communities",
                  loc="upper left", bbox_to_anchor=(1.01, 1),
                  fontsize=7, ncol=ncol, frameon=False)
        plt.tight_layout()

        buf = _save_fig(fig)
        st.image(buf, use_container_width=True)
        st.download_button("Download (PNG)", data=buf.getvalue(),
                           file_name="psn_network.png", mime="image/png")

# =====================================================================
# TAB 2 — Feature Prevalence Bar Charts (top-N communities)
# =====================================================================
with tab2:
    st.subheader(f"Feature prevalences — top {n_top} communities")
    st.markdown(
        "Horizontal bar charts showing % prevalence or mean value per community. "
        "Communities ordered by patient count (largest first)."
    )

    pct_cols  = [c for c in top_sig.columns if c.endswith("_pct") and c != "n_patients"]
    mean_cols = [c for c in top_sig.columns if c.endswith("_mean")]

    col_sel, col_feat = st.columns([1, 2])
    with col_sel:
        show_continuous = st.checkbox("Include continuous features (mean)", value=False)
    with col_feat:
        available_prefixes = sorted({c.split("_")[0] for c in pct_cols})
        selected_prefixes = st.multiselect(
            "Feature groups to include",
            options=available_prefixes,
            default=available_prefixes,
        )

    plot_btn2 = st.button("Plot Prevalences", key="plot_prev", type="primary")

    if plot_btn2:
        filtered_pct = [c for c in pct_cols if any(c.startswith(p) for p in selected_prefixes)]

        if not filtered_pct:
            st.warning("No features selected.")
        else:
            with st.spinner("Plotting…"):
                # Sort communities by patient count descending
                ordered = top_sig.sort_values("n_patients", ascending=False)
                groups  = ordered.index.tolist()
                n_groups = len(groups)
                palette  = _community_palette(n_groups)
                group_cols = {g: palette[i] for i, g in enumerate(groups)}

                panels = [("Prevalence (%)", filtered_pct)]
                if show_continuous and mean_cols:
                    panels.append(("Mean value", mean_cols))

                for panel_title, cols in panels:
                    n_cols = len(cols)
                    fig, axes = plt.subplots(
                        1, n_cols,
                        figsize=(max(12, n_cols * 2.2), max(4, n_groups * 0.45 + 1)),
                        sharey=True,
                    )
                    if n_cols == 1:
                        axes = [axes]

                    for ax, col in zip(axes, cols):
                        vals   = ordered[col].fillna(0)
                        colors = [group_cols[g] for g in groups]
                        ax.barh([str(g) for g in groups], vals,
                                color=colors, edgecolor="white", linewidth=0.5)
                        ax.set_xlabel(panel_title, fontsize=8)
                        ax.set_title(
                            col.replace("_pct", "").replace("_mean", ""),
                            fontsize=8, pad=4,
                        )
                        ax.tick_params(axis="y", labelsize=7)
                        ax.tick_params(axis="x", labelsize=7)
                        ax.spines[["top", "right"]].set_visible(False)

                    fig.suptitle(f"{panel_title} by community (top {n_top})",
                                 fontsize=10, y=1.01)
                    plt.tight_layout()

                    buf = _save_fig(fig)
                    st.image(buf, use_container_width=True)
                    st.download_button(
                        f"Download {panel_title} (PNG)",
                        data=buf.getvalue(),
                        file_name=f"prevalence_{panel_title.lower().replace(' ', '_')}.png",
                        mime="image/png",
                        key=f"dl_prev_{panel_title}",
                    )

# =====================================================================
# TAB 3 — Bubble Chart: Community Size vs Outcome (top-N)
# =====================================================================
with tab3:
    st.subheader(f"Community size vs. outcome — top {n_top} communities")
    st.markdown(
        "Each bubble is a community. "
        "X-axis = patients. Y-axis = outcome of choice. Bubble size = patient count."
    )

    outcome_candidates = [c for c in top_sig.columns if c.endswith("_mean")]
    pct_candidates     = [c for c in top_sig.columns if c.endswith("_pct")]
    all_y_options = outcome_candidates + pct_candidates

    if not all_y_options:
        st.warning("No outcome columns found in the signature.")
    else:
        y_col = st.selectbox(
            "Y-axis (outcome)", all_y_options,
            index=next(
                (i for i, c in enumerate(all_y_options) if "READMIT" in c or "REVISIT" in c),
                0,
            ),
        )

        plot_btn3 = st.button("Plot Bubble Chart", key="plot_bubble", type="primary")

        if plot_btn3:
            with st.spinner("Plotting…"):
                df_b = top_sig.reset_index().copy()
                id_col = df_b.columns[0]
                df_b = df_b.dropna(subset=["n_patients", y_col])
                df_b = df_b.sort_values("n_patients", ascending=False)

                palette_b = _community_palette(len(df_b))
                node_cols_b = palette_b

                sizes = (df_b["n_patients"] / df_b["n_patients"].max() * 1200).clip(lower=30)

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(
                    df_b["n_patients"], df_b[y_col],
                    s=sizes, c=node_cols_b, alpha=0.75,
                    edgecolors="white", linewidths=0.5,
                )
                for _, row in df_b.iterrows():
                    ax.annotate(
                        str(row[id_col]),
                        (row["n_patients"], row[y_col]),
                        fontsize=6, ha="center", va="center", color="black",
                    )

                ax.set_xlabel("Patients in community", fontsize=10)
                ax.set_ylabel(
                    y_col.replace("_mean", " (mean)").replace("_pct", " (%)"),
                    fontsize=10,
                )
                ax.set_title(f"Community size vs. {y_col}  (top {n_top})", fontsize=11)
                ax.spines[["top", "right"]].set_visible(False)
                plt.tight_layout()

                buf = _save_fig(fig)
                st.image(buf, use_container_width=True)
                st.download_button("Download (PNG)", data=buf.getvalue(),
                                   file_name="bubble_chart.png", mime="image/png")

# =====================================================================
# TAB 4 — Payer Mix Stacked Bar (top-N)
# =====================================================================
with tab4:
    st.subheader(f"Payer mix — top {n_top} communities")
    st.markdown("Stacked bar showing payer proportion per community.")

    payer_cols = sorted(
        [c for c in top_sig.columns if c.startswith("PAYER_") and c.endswith("_pct")]
    )

    if not payer_cols:
        st.warning(
            "No PAYER columns found in the signature. "
            "Make sure PAYER is selected on page 06 and analysis re-run on page 08."
        )
    else:
        payer_names = [c.replace("PAYER_", "").replace("_pct", "") for c in payer_cols]
        orientation = st.radio("Orientation", ["Horizontal", "Vertical"], horizontal=True)
        plot_btn4 = st.button("Plot Payer Mix", key="plot_payer", type="primary")

        if plot_btn4:
            with st.spinner("Plotting…"):
                df_p = top_sig[payer_cols].sort_values(
                    top_sig["n_patients"].name
                    if top_sig["n_patients"].name in top_sig.columns
                    else top_sig.columns[0],
                    ascending=False,
                )
                # Sort rows by patient count
                df_p = top_sig[payer_cols].loc[
                    top_sig["n_patients"].sort_values(ascending=False).index
                ].copy()
                df_p.columns = payer_names
                df_p.index = [str(i) for i in df_p.index]

                row_sums = df_p.sum(axis=1).replace(0, np.nan)
                df_p = df_p.div(row_sums, axis=0) * 100

                payer_colors = _community_palette(len(payer_names))

                fig, ax = plt.subplots(
                    figsize=(
                        (max(8, len(df_p) * 0.9), 5) if orientation == "Vertical"
                        else (8, max(4, len(df_p) * 0.55))
                    )
                )

                bottom = np.zeros(len(df_p))
                for col, color in zip(payer_names, payer_colors):
                    vals = df_p[col].fillna(0).values
                    if orientation == "Vertical":
                        ax.bar(df_p.index, vals, bottom=bottom, color=color,
                               label=col, edgecolor="white", linewidth=0.4)
                    else:
                        ax.barh(df_p.index, vals, left=bottom, color=color,
                                label=col, edgecolor="white", linewidth=0.4)
                    bottom += vals

                if orientation == "Vertical":
                    ax.set_xlabel("Community", fontsize=10)
                    ax.set_ylabel("Share (%)", fontsize=10)
                    ax.tick_params(axis="x", rotation=45, labelsize=8)
                else:
                    ax.set_xlabel("Share (%)", fontsize=10)
                    ax.set_ylabel("Community", fontsize=10)
                    ax.tick_params(axis="y", labelsize=8)

                ax.set_title(f"Payer mix by community (top {n_top})", fontsize=11)
                ax.spines[["top", "right"]].set_visible(False)
                ax.legend(title="Payer", fontsize=8, frameon=False,
                          bbox_to_anchor=(1.01, 1), loc="upper left")
                plt.tight_layout()

                buf = _save_fig(fig)
                st.image(buf, use_container_width=True)
                st.download_button("Download (PNG)", data=buf.getvalue(),
                                   file_name="payer_mix.png", mime="image/png")
