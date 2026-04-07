import streamlit as st
import pandas as pd
import plotly.express as px

SCORE_COL = "Efficiency Score"

# Canonical complexity level order — used everywhere levels appear
COMPLEXITY_ORDER = ["SIMPLE", "MODERATE", "COMPLEX", "VERY_COMPLEX"]

_SEMANTIC_HELP = (
    "Measures how well the generated SQL matches the intent of the original "
    "natural-language query. Computed by asking the model to explain the generated "
    "SQL in natural language and comparing that explanation against the user's "
    "original query."
)
_F1_HELP = (
    "Measures the F1 score of the generated SQL by comparing the query result sets "
    "of the generated SQL and the gold SQL. F1 is the harmonic mean of precision and recall."
)

# Accuracy metric configuration — defaults to Semantic; call configure_accuracy_metric() to switch
ACCURACY_COL = "Normalized Semantic Score"
ACCURACY_SHORT = "Semantic"

METRIC_COLS = [
    "Normalized Latency (s)",
    "Normalized Complexity Score",
    "Normalized Log Transformed Tokens",
    ACCURACY_COL,
]

_NON_ACCURACY_INFO = {
    "Normalized Log Transformed Tokens": {
        "short": "Token",
        "help": "Measures token usage for the query run. Based on prompt and output token count.",
    },
    "Normalized Latency (s)": {
        "short": "Latency",
        "help": "Measures runtime / response latency for the query run.",
    },
    "Normalized Complexity Score": {
        "short": "Complexity",
        "help": (
            "Measures structural complexity of the generated SQL. The complexity score is based "
            "on SQL characteristics such as joins, subqueries, aggregations, conditions and other operations."
        ),
    },
}

METRIC_INFO = {ACCURACY_COL: {"short": ACCURACY_SHORT, "help": _SEMANTIC_HELP}, **_NON_ACCURACY_INFO}


def configure_accuracy_metric(df: pd.DataFrame):
    """Detect whether F1 Score is available and reconfigure the accuracy metric."""
    global ACCURACY_COL, ACCURACY_SHORT, METRIC_COLS, METRIC_INFO

    if "Normalized F1 Score" in df.columns:
        ACCURACY_COL = "Normalized F1 Score"
        ACCURACY_SHORT = "F1"
        acc_help = _F1_HELP
    else:
        ACCURACY_COL = "Normalized Semantic Score"
        ACCURACY_SHORT = "Semantic"
        acc_help = _SEMANTIC_HELP

    METRIC_COLS[:] = [
        "Normalized Latency (s)",
        "Normalized Complexity Score",
        "Normalized Log Transformed Tokens",
        ACCURACY_COL,
    ]

    METRIC_INFO.clear()
    METRIC_INFO.update({ACCURACY_COL: {"short": ACCURACY_SHORT, "help": acc_help}, **_NON_ACCURACY_INFO})


# -------------------------
# Helper: sort complexity levels in canonical order
# -------------------------
def _sort_complexity_levels(levels: list) -> list:
    """
    Sort a list of complexity level strings in canonical order:
    SIMPLE → MODERATE → COMPLEX → VERY_COMPLEX.
    Any unrecognised values are appended at the end in alphabetical order.
    Case-insensitive matching.
    """
    upper_map = {str(l).strip().upper(): str(l).strip() for l in levels}
    ordered = []
    for canonical in COMPLEXITY_ORDER:
        if canonical in upper_map:
            ordered.append(upper_map[canonical])
    # append anything not in canonical list
    for orig in sorted(upper_map.values()):
        if orig.upper() not in COMPLEXITY_ORDER:
            ordered.append(orig)
    return ordered


# -------------------------
# Filters
# -------------------------
def compute_filters(df_full: pd.DataFrame):
    required = ["Model", SCORE_COL] + METRIC_COLS
    missing = [c for c in required if c not in df_full.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    extra_cols = []
    if "Complexity Level" in df_full.columns:
        extra_cols.append("Complexity Level")
    if "Prompt" in df_full.columns:
        extra_cols.append("Prompt")

    df_scores = df_full[["Model", SCORE_COL] + METRIC_COLS + extra_cols].copy()
    df_scores["Model"] = df_scores["Model"].astype(str)

    for c in METRIC_COLS + [SCORE_COL]:
        df_scores[c] = pd.to_numeric(df_scores[c], errors="coerce")

    df_scores = df_scores.dropna(subset=["Model", SCORE_COL])

    st.sidebar.header("Filters")

    # --- Model filter ---
    models = sorted(df_scores["Model"].unique().tolist())
    selected_models = st.sidebar.multiselect("Models", models, default=models)

    # --- Total score slider ---
    lo, hi = float(df_scores[SCORE_COL].min()), float(df_scores[SCORE_COL].max())
    score_range = st.sidebar.slider("Total Score range", lo, hi, (lo, hi))

    # Base mask
    mask = df_scores["Model"].isin(selected_models) & df_scores[SCORE_COL].between(*score_range)

    # --- Complexity Level filter ---
    if "Complexity Level" in df_scores.columns:
        raw_levels = (
            df_scores["Complexity Level"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )
        levels_ordered = _sort_complexity_levels(raw_levels)

        selected_levels = st.sidebar.multiselect(
            "Complexity levels",
            options=levels_ordered,
            default=levels_ordered,
        )

        # Empty selection = show nothing (not everything)
        mask = mask & df_scores["Complexity Level"].astype(str).str.strip().isin(selected_levels)

    # --- Prompt include filter ---
    if "Prompt" in df_scores.columns:
        st.sidebar.subheader("Prompt Filter")

        q = st.sidebar.text_input(
            "Search prompt by text",
            value="",
            placeholder="type to search...",
        ).strip()

        prompts_series = df_scores.loc[mask, "Prompt"].dropna().astype(str)
        recent_prompts = prompts_series.drop_duplicates(keep="last")

        if q:
            recent_prompts = recent_prompts[recent_prompts.str.contains(q, case=False, na=False)]

        prompt_options = recent_prompts.tail(50).tolist()

        selected_prompts = st.sidebar.multiselect(
            "Select prompts (up to 50 shown)",
            options=prompt_options,
            default=[],
        )

        if selected_prompts:
            mask = mask & df_scores["Prompt"].astype(str).isin(selected_prompts)

    # --- Prompt exclude filter ---
    if "Prompt" in df_scores.columns:
        st.sidebar.subheader("Prompt Exclusion")

        excl_q = st.sidebar.text_input(
            "Search for prompt to exclude",
            value="",
            placeholder="type to search...",
        ).strip()

        all_prompts = df_scores.loc[mask, "Prompt"].dropna().astype(str)
        unique_prompts = all_prompts.drop_duplicates(keep="first").tolist()

        if excl_q:
            unique_prompts = [p for p in unique_prompts if excl_q.lower() in p.lower()]

        excluded_prompts = st.sidebar.multiselect(
            "Exclude prompts (up to 50 shown)",
            options=unique_prompts[:50],
            default=[],
        )

        if excluded_prompts:
            mask = mask & ~df_scores["Prompt"].astype(str).isin(excluded_prompts)

    df_scores_f = df_scores.loc[mask].copy()
    df_full_f = df_full.loc[df_scores.index[mask]].copy()

    if df_scores_f.empty:
        st.warning("No rows match the current filters — try checking all the filters (models / scores / complexity levels).")
        st.stop()

    return df_full_f, df_scores_f


# -------------------------
# Summary row
# -------------------------
def _clamp01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, x))


def render_summary_row(df_scores_f: pd.DataFrame):
    avg_total = float(df_scores_f[SCORE_COL].mean())
    med_total = float(df_scores_f[SCORE_COL].median())
    runs = int(len(df_scores_f))
    models = int(df_scores_f["Model"].nunique())

    c1, c2, c3 = st.columns([1.4, 1.4, 1.0])

    _score_formula = f"Calculated Score = 60% {ACCURACY_SHORT} + 20% Complexity + 10% Latency + 10% Token."

    with c1:
        st.metric(
            "Average Total Score",
            f"{avg_total:.3f}",
            help=f"Average of the final calculated score across the currently selected queries. {_score_formula}",
        )
        st.progress(_clamp01(avg_total), text="0–1 normalised")

    with c2:
        st.metric(
            "Median Total Score",
            f"{med_total:.3f}",
            help=f"Median of the final calculated score across the currently selected queries. {_score_formula}",
        )
        st.progress(_clamp01(med_total), text="0–1 normalised")

    with c3:
        st.metric("Total Number of Queries", runs, help="Number of query runs currently included.")
        st.metric("Total Number of Models", models, help="Number of distinct models currently included.")


def render_metric_name_row():
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(ACCURACY_SHORT, "", help=METRIC_INFO[ACCURACY_COL]["help"])
    with c2:
        st.metric("Latency", "", help=METRIC_INFO["Normalized Latency (s)"]["help"])
    with c3:
        st.metric("Token", "", help=METRIC_INFO["Normalized Log Transformed Tokens"]["help"])
    with c4:
        st.metric("Complexity", "", help=METRIC_INFO["Normalized Complexity Score"]["help"])


# -------------------------
# Aggregation helper
# -------------------------
def _agg_by_model(df_scores_f: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df_scores_f.groupby("Model")
        .agg(
            runs=(SCORE_COL, "count"),
            mean_total=(SCORE_COL, "mean"),
            median_total=(SCORE_COL, "median"),
            mean_latency=("Normalized Latency (s)", "mean"),
            mean_complexity=("Normalized Complexity Score", "mean"),
            mean_token=("Normalized Log Transformed Tokens", "mean"),
            mean_accuracy=(ACCURACY_COL, "mean"),
        )
        .reset_index()
        .sort_values("mean_total", ascending=False)
    )
    return agg


# -------------------------
# Plotly helper: bar chart with labels
# -------------------------
def _bar_with_labels(df, x, y, title, decimals=3, hover_cols=None):
    d = df.copy()
    d["_label"] = d[y].round(decimals).astype(str)
    fig = px.bar(d, x=x, y=y, text="_label", title=title, hover_data=hover_cols)
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        yaxis_range=[0, 1],
        margin=dict(l=10, r=10, t=40, b=10),
        uniformtext_minsize=10,
        uniformtext_mode="hide",
    )
    st.plotly_chart(fig, use_container_width=True)


# -------------------------
# Deeper Analysis dialog
# -------------------------
@st.dialog("Deeper Analysis on Calculated Score", width="large")
def _whatif_dialog(df_scores_f: pd.DataFrame):
    tab1, tab2, tab3 = st.tabs(["Weight Simulator", "Score Calculator", "Metric Correlation"])

    # ----------------------------------------------------------------
    # TAB 1: Weight Simulator
    # ----------------------------------------------------------------
    with tab1:
        st.markdown("Adjust how much each metric contributes to the final score. Must sum to **100%**.")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            w_accuracy = st.slider(f"{ACCURACY_SHORT} %", 0, 100, 60, step=5, key="ws_accuracy",
                                   help=METRIC_INFO[ACCURACY_COL]["help"])
        with c2:
            w_complexity = st.slider("Complexity %", 0, 100, 20, step=5, key="ws_complexity",
                                     help=METRIC_INFO["Normalized Complexity Score"]["help"])
        with c3:
            w_latency = st.slider("Latency %", 0, 100, 10, step=5, key="ws_latency",
                                  help=METRIC_INFO["Normalized Latency (s)"]["help"])
        with c4:
            w_token = st.slider("Token %", 0, 100, 10, step=5, key="ws_token",
                                help=METRIC_INFO["Normalized Log Transformed Tokens"]["help"])

        total = w_accuracy + w_complexity + w_latency + w_token
        if total != 100:
            st.warning(f"Weights sum to **{total}%** — adjust to reach exactly 100%.")
        else:
            st.success("Weights sum to 100% ✓")

        if total == 100:
            df_sim = df_scores_f.copy()
            df_sim["Simulated Score"] = (
                df_sim[ACCURACY_COL] * w_accuracy / 100
                + df_sim["Normalized Complexity Score"] * w_complexity / 100
                + df_sim["Normalized Latency (s)"] * w_latency / 100
                + df_sim["Normalized Log Transformed Tokens"] * w_token / 100
            )

            agg_sim = (
                df_sim.groupby("Model")
                .agg(mean_simulated=("Simulated Score", "mean"), mean_original=(SCORE_COL, "mean"))
                .reset_index()
                .sort_values("mean_simulated", ascending=False)
            )
            agg_sim["_label"] = agg_sim["mean_simulated"].round(3).astype(str)

            fig = px.bar(agg_sim, x="Model", y="mean_simulated", text="_label",
                         title="Simulated Mean Score by Model")
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(yaxis_range=[0, 1], margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)

            orig_rank = (
                agg_sim.sort_values("mean_original", ascending=False)
                .reset_index(drop=True).reset_index()
                .rename(columns={"index": "original_rank"})[["Model", "original_rank", "mean_original"]]
            )
            orig_rank["original_rank"] += 1

            sim_rank = (
                agg_sim.sort_values("mean_simulated", ascending=False)
                .reset_index(drop=True).reset_index()
                .rename(columns={"index": "simulated_rank"})[["Model", "simulated_rank", "mean_simulated"]]
            )
            sim_rank["simulated_rank"] += 1

            rank_df = orig_rank.merge(sim_rank, on="Model")
            rank_df["rank_change"] = rank_df["original_rank"] - rank_df["simulated_rank"]
            rank_df["Rank Change"] = rank_df["rank_change"].apply(
                lambda x: f"▲ {x}" if x > 0 else (f"▼ {abs(x)}" if x < 0 else "—")
            )
            rank_df = rank_df.sort_values("simulated_rank")[
                ["Model", "original_rank", "simulated_rank", "Rank Change", "mean_original", "mean_simulated"]
            ].rename(columns={
                "original_rank": "Original Rank",
                "simulated_rank": "New Rank",
                "mean_original": "Original Score",
                "mean_simulated": "Simulated Score",
            })
            rank_df["Original Score"] = rank_df["Original Score"].round(3)
            rank_df["Simulated Score"] = rank_df["Simulated Score"].round(3)
            st.markdown("**Rank Change vs Original Weights**")
            st.dataframe(rank_df, use_container_width=True, hide_index=True)
        else:
            st.info("Fix weights to sum to 100% to see the chart and rank table.")

    # ----------------------------------------------------------------
    # TAB 2: Score Calculator
    # ----------------------------------------------------------------
    with tab2:
        st.markdown("Enter hypothetical metric values (0–1) and weights to compute what a model's score would be.")

        st.markdown("**Weights**")
        wc1, wc2, wc3, wc4 = st.columns(4)
        with wc1:
            cw_accuracy = st.slider(f"{ACCURACY_SHORT} %", 0, 100, 60, step=5, key="calc_w_accuracy",
                                    help=METRIC_INFO[ACCURACY_COL]["help"])
        with wc2:
            cw_complexity = st.slider("Complexity %", 0, 100, 20, step=5, key="calc_w_complexity",
                                      help=METRIC_INFO["Normalized Complexity Score"]["help"])
        with wc3:
            cw_latency = st.slider("Latency %", 0, 100, 10, step=5, key="calc_w_latency",
                                   help=METRIC_INFO["Normalized Latency (s)"]["help"])
        with wc4:
            cw_token = st.slider("Token %", 0, 100, 10, step=5, key="calc_w_token",
                                 help=METRIC_INFO["Normalized Log Transformed Tokens"]["help"])

        calc_total = cw_accuracy + cw_complexity + cw_latency + cw_token
        if calc_total != 100:
            st.warning(f"Weights sum to **{calc_total}%** — adjust to reach exactly 100%.")
        else:
            st.success("Weights sum to 100% ✓")

        st.markdown("**Metric Values**")
        vc1, vc2, vc3, vc4 = st.columns(4)
        with vc1:
            v_accuracy = st.number_input(f"{ACCURACY_SHORT} Score", 0.0, 1.0, 0.80, step=0.01, key="calc_v_accuracy",
                                         help=METRIC_INFO[ACCURACY_COL]["help"])
        with vc2:
            v_complexity = st.number_input("Complexity Score", 0.0, 1.0, 0.75, step=0.01, key="calc_v_complexity",
                                           help=METRIC_INFO["Normalized Complexity Score"]["help"])
        with vc3:
            v_latency = st.number_input("Latency Score", 0.0, 1.0, 0.70, step=0.01, key="calc_v_latency",
                                        help=METRIC_INFO["Normalized Latency (s)"]["help"])
        with vc4:
            v_token = st.number_input("Token Score", 0.0, 1.0, 0.65, step=0.01, key="calc_v_token",
                                      help=METRIC_INFO["Normalized Log Transformed Tokens"]["help"])

        calculated = (
            v_accuracy * cw_accuracy / 100
            + v_complexity * cw_complexity / 100
            + v_latency * cw_latency / 100
            + v_token * cw_token / 100
        )

        st.divider()
        st.markdown("**Result**")
        res_col, _ = st.columns([1, 2])
        with res_col:
            st.metric(
                label="Calculated Score",
                value=f"{calculated:.4f}",
                help="Weighted sum of all four metrics. All inputs expected on a 0–1 scale.",
            )
            st.progress(_clamp01(calculated))

        breakdown = pd.DataFrame({
            "Metric": [ACCURACY_SHORT, "Complexity", "Latency", "Token"],
            "Contribution": [
                v_accuracy * cw_accuracy / 100,
                v_complexity * cw_complexity / 100,
                v_latency * cw_latency / 100,
                v_token * cw_token / 100,
            ],
        })
        fig_b = px.bar(breakdown, x="Metric", y="Contribution",
                       title="Score Contribution by Metric",
                       text=breakdown["Contribution"].round(4).astype(str),
                       color="Metric")
        fig_b.update_traces(textposition="outside", cliponaxis=False)
        fig_b.update_layout(yaxis_range=[0, 1], showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_b, use_container_width=True)

    # ----------------------------------------------------------------
    # TAB 3: Metric Correlation
    # ----------------------------------------------------------------
    with tab3:
        st.markdown(
            "Shows the **real relationship** between each metric and the final Calculated Score "
            "across all actual data, coloured by model."
        )

        metric_options = {
            f"{ACCURACY_SHORT} Score": ACCURACY_COL,
            "Complexity Score": "Normalized Complexity Score",
            "Latency Score": "Normalized Latency (s)",
            "Token Score": "Normalized Log Transformed Tokens",
        }

        selected_metric_label = st.selectbox(
            "Select metric to plot against Calculated Score",
            options=list(metric_options.keys()),
            key="scatter_metric",
        )
        selected_metric_col = metric_options[selected_metric_label]

        hover_cols = ["Model"]
        if "Prompt" in df_scores_f.columns:
            hover_cols.append("Prompt")

        try:
            fig_scatter = px.scatter(
                df_scores_f, x=selected_metric_col, y=SCORE_COL,
                color="Model", hover_data=hover_cols,
                title=f"{selected_metric_label} vs Calculated Score",
                trendline="ols", trendline_scope="overall",
            )
        except Exception:
            fig_scatter = px.scatter(
                df_scores_f, x=selected_metric_col, y=SCORE_COL,
                color="Model", hover_data=hover_cols,
                title=f"{selected_metric_label} vs Calculated Score",
            )
            st.caption("Install `statsmodels` to show a trendline: `pip install statsmodels`")

        fig_scatter.update_layout(xaxis_range=[0, 1], yaxis_range=[0, 1],
                                  margin=dict(l=10, r=10, t=60, b=10))
        fig_scatter.update_traces(marker=dict(size=7, opacity=0.7))
        st.plotly_chart(fig_scatter, use_container_width=True)

        corr = df_scores_f[[selected_metric_col, SCORE_COL]].dropna().corr().iloc[0, 1]
        st.metric(
            label=f"Pearson Correlation: {selected_metric_label} vs Calculated Score",
            value=f"{corr:.3f}",
            help="1.0 = perfect positive correlation, 0 = no correlation, -1.0 = perfect negative correlation",
        )
        st.caption("Trendline fitted across all models combined. Hover over points to see prompt details.")


# -------------------------
# TAB 1: SUMMARY
# -------------------------
def render_tab_summary(df_scores_f: pd.DataFrame):
    st.subheader("Model Summary Scores")
    st.markdown(
        f"**Calculated Score** is computed with the following weightage: "
        f"**60% Accuracy** ({ACCURACY_SHORT} Score), **20% Complexity**, **10% Latency**, **10% Token**."
    )

    render_metric_name_row()

    if st.button("Deeper Analysis on Calculated Score", key="open_whatif"):
        _whatif_dialog(df_scores_f)

    agg = _agg_by_model(df_scores_f)

    left, right = st.columns(2)
    with left:
        _bar_with_labels(agg, x="Model", y="mean_total", title="Mean Total Score by Model", hover_cols=["runs"])
    with right:
        _bar_with_labels(agg, x="Model", y="median_total", title="Median Total Score by Model", hover_cols=["runs"])

    st.markdown("**Table View**")
    st.dataframe(
        agg[["Model", "runs", "mean_total", "median_total"]],
        use_container_width=True,
        hide_index=True,
    )


# -------------------------
# TAB 2: EFFICIENCY
# -------------------------
def render_tab_efficiency(df_scores_f: pd.DataFrame):
    st.subheader("Efficiency")
    st.markdown(
        "- **Latency**: End-to-end Response Time (Lower time → Higher normalised score).\n"
        "- **Token**: Prompt and Output Tokens (Lower tokens → Higher normalised score).\n"
        "- **Complexity**: Complexity of the Generated SQL (Lower complexity → Higher normalised score)."
    )

    agg = _agg_by_model(df_scores_f)

    c1, c2, c3 = st.columns(3)
    with c1:
        _bar_with_labels(agg, x="Model", y="mean_latency", title="Mean Norm Latency by Model", hover_cols=["runs"])
    with c2:
        _bar_with_labels(agg, x="Model", y="mean_token", title="Mean Norm Token by Model", hover_cols=["runs"])
    with c3:
        _bar_with_labels(agg, x="Model", y="mean_complexity", title="Mean Norm Complexity by Model", hover_cols=["runs"])

    st.markdown("**Efficiency Profile**")
    prof = agg.set_index("Model")[["mean_latency", "mean_token", "mean_complexity"]].copy()
    prof.columns = ["Latency", "Token", "Complexity"]
    prof_long = prof.reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")

    fig = px.line(prof_long, x="Metric", y="Score", color="Model", markers=True,
                  title="Efficiency Profile", hover_data=["Model", "Metric", "Score"])
    fig.update_layout(yaxis_range=[0, 1], margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


# -------------------------
# TAB 3: ACCURACY
# -------------------------
def render_tab_accuracy(df_scores_f: pd.DataFrame):
    st.subheader("Accuracy")
    if ACCURACY_SHORT == "F1":
        st.markdown(
            "- **F1**: Measures correctness by comparing the result sets of the generated SQL "
            "and the gold SQL, shown as a normalised score."
        )
    else:
        st.markdown(
            "- **Semantic**: How well the generated explanation aligns with the NL query intent, "
            "shown as a normalised score."
        )

    agg = _agg_by_model(df_scores_f)
    _bar_with_labels(agg, x="Model", y="mean_accuracy",
                     title=f"Mean Normalized {ACCURACY_SHORT} Score by Model", hover_cols=["runs"])

    st.divider()

    if "Complexity Level" not in df_scores_f.columns:
        st.info("No 'Complexity Level' column found — complexity drilldown is hidden.")
        return

    tmp = df_scores_f[["Model", "Complexity Level", ACCURACY_COL]].copy()
    tmp["Complexity Level"] = tmp["Complexity Level"].astype(str).str.strip()
    tmp = tmp[tmp["Complexity Level"].notna() & (tmp["Complexity Level"] != "")]

    if tmp.empty:
        st.info("Complexity Level exists but no usable values found after filtering.")
        return

    levels_ordered = _sort_complexity_levels(tmp["Complexity Level"].unique().tolist())
    tmp["Complexity Level"] = pd.Categorical(tmp["Complexity Level"], categories=levels_ordered, ordered=True)

    mode = st.radio(
        f"{ACCURACY_SHORT} Accuracy by Query Complexity",
        ["Overall with all selected models combined", "Per model"],
        horizontal=True,
        index=0,
        help=(
            f"Shows how {ACCURACY_SHORT.lower()} score varies across query complexity groupings. "
            "Uses the 'Complexity Level' values in your dataset."
        ),
    )

    if mode.startswith("Overall"):
        by_lvl = (
            tmp.groupby("Complexity Level", as_index=False, observed=True)
            .agg(mean_accuracy=(ACCURACY_COL, "mean"), n=(ACCURACY_COL, "count"))
            .sort_values("Complexity Level")
        )
        by_lvl["_label"] = by_lvl["mean_accuracy"].round(3).astype(str)

        fig = px.line(by_lvl, x="Complexity Level", y="mean_accuracy", markers=True,
                      text="_label",
                      title=f"Mean Normalized {ACCURACY_SHORT} Score vs Complexity Level (Overall)",
                      hover_data=["n"])
        fig.update_traces(textposition="top center")
        fig.update_layout(yaxis_range=[0, 1], margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    else:
        by_model_lvl = (
            tmp.groupby(["Model", "Complexity Level"], as_index=False, observed=True)
            .agg(mean_accuracy=(ACCURACY_COL, "mean"))
        )
        by_model_lvl["Complexity Level"] = by_model_lvl["Complexity Level"].astype(str)
        order_map = {lvl: i for i, lvl in enumerate(levels_ordered)}
        by_model_lvl["_order"] = by_model_lvl["Complexity Level"].map(order_map).fillna(10**9)
        by_model_lvl = by_model_lvl.sort_values(["_order", "Model"]).drop(columns=["_order"])

        fig = px.line(by_model_lvl, x="Complexity Level", y="mean_accuracy", color="Model",
                      markers=True,
                      title=f"Normalized {ACCURACY_SHORT} Score across Complexity Level (per model)",
                      category_orders={"Complexity Level": levels_ordered},
                      hover_data=["Model", "mean_accuracy"])
        fig.update_layout(yaxis_range=[0, 1], margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)


# -------------------------
# Raw data section with checkboxes, instant update via st.rerun()
# -------------------------
def render_raw_data_section(df_full_f: pd.DataFrame) -> pd.Index:
    if "excluded_idx" not in st.session_state:
        st.session_state["excluded_idx"] = set()

    # Clear stale exclusions when filters change
    st.session_state["excluded_idx"] &= set(df_full_f.index.tolist())

    with st.expander("Raw data (All Columns) — uncheck rows to exclude from dashboard", expanded=False):
        df_raw = df_full_f.copy().reset_index(drop=False)
        df_raw = df_raw.rename(columns={"index": "_orig_idx"})

        if ACCURACY_SHORT != "F1":
            drop_candidates = ["Gold SQL", "gold sql", "Gold_SQL", "gold_sql"]
            df_raw = df_raw.drop(columns=[c for c in drop_candidates if c in df_raw.columns])

        df_raw.insert(0, "Include", ~df_raw["_orig_idx"].isin(st.session_state["excluded_idx"]))

        # Column ordering: Include → Complexity Level → rest (no _orig_idx)
        ordered_cols = ["Include"]
        if "Complexity Level" in df_raw.columns:
            ordered_cols.append("Complexity Level")
        ordered_cols += [c for c in df_raw.columns if c not in ("Include", "Complexity Level", "_orig_idx")]

        df_display = df_raw[ordered_cols + ["_orig_idx"]].copy()

        # Sort Complexity Level column categorically if present
        if "Complexity Level" in df_display.columns:
            raw_levels = df_display["Complexity Level"].dropna().astype(str).str.strip().unique().tolist()
            levels_ordered = _sort_complexity_levels(raw_levels)
            df_display["Complexity Level"] = pd.Categorical(
                df_display["Complexity Level"].astype(str).str.strip(),
                categories=levels_ordered,
                ordered=True,
            )

        column_config = {
            "Include": st.column_config.CheckboxColumn(
                "Include",
                help="Untick to exclude this row from the dashboard",
                default=True,
            ),
        }

        numeric_col_config = {
            ACCURACY_COL: METRIC_INFO[ACCURACY_COL]["help"],
            "Normalized Latency (s)": METRIC_INFO["Normalized Latency (s)"]["help"],
            "Normalized Log Transformed Tokens": METRIC_INFO["Normalized Log Transformed Tokens"]["help"],
            "Normalized Complexity Score": METRIC_INFO["Normalized Complexity Score"]["help"],
            SCORE_COL: "Final weighted score used for model comparison.",
        }
        for col, help_text in numeric_col_config.items():
            if col in df_display.columns:
                column_config[col] = st.column_config.NumberColumn(col, help=help_text, format="%.3f")

        edited = st.data_editor(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config=column_config,
            disabled=[c for c in df_display.columns if c != "Include"],
            column_order=ordered_cols,
            key="raw_data_editor",
        )

        new_excluded = set(edited.loc[~edited["Include"], "_orig_idx"].tolist())

        if new_excluded != st.session_state["excluded_idx"]:
            st.session_state["excluded_idx"] = new_excluded
            st.rerun()

        included = edited.loc[edited["Include"], "_orig_idx"]
        st.caption(f"{edited['Include'].sum()} / {len(edited)} rows included in the dashboard.")

    return pd.Index(included)