import streamlit as st
import pandas as pd
import plotly.express as px

# Keep only the metrics you will still use
METRIC_COLS = [
    "Norm Latency",
    "Norm Complexity Score",
    "Norm Token",
    "Norm Semantic",
]

# -------------------------
# Filters
# -------------------------
def compute_filters(df_full: pd.DataFrame):
    missing = [c for c in ["Model"] + METRIC_COLS if c not in df_full.columns]
    if missing:
        st.error(f"Missing required columns in Sheet1: {missing}")
        st.stop()

    extra_cols = []
    if "Complexity Level" in df_full.columns:
        extra_cols.append("Complexity Level")

    df_scores = df_full[["Model"] + METRIC_COLS + extra_cols].copy()
    df_scores["Model"] = df_scores["Model"].astype(str)

    for c in METRIC_COLS:
        df_scores[c] = pd.to_numeric(df_scores[c], errors="coerce")

    # Compute new total score without F1: average of 4 components
    df_scores["Total Score"] = df_scores[METRIC_COLS].mean(axis=1)

    df_scores = df_scores.dropna(subset=["Model", "Total Score"])

    st.sidebar.header("Filters")

    models = sorted(df_scores["Model"].unique().tolist())
    selected_models = st.sidebar.multiselect("Models", models, default=models)

    lo, hi = float(df_scores["Total Score"].min()), float(df_scores["Total Score"].max())
    score_range = st.sidebar.slider("Total Score range", lo, hi, (lo, hi))

    mask = df_scores["Model"].isin(selected_models) & df_scores["Total Score"].between(*score_range)

    df_scores_f = df_scores[mask].copy()
    df_full_f = df_full.loc[df_scores.index[mask]].copy()

    if df_scores_f.empty:
        st.warning("No rows match the current filters.")
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
    avg_total = float(df_scores_f["Total Score"].mean())
    med_total = float(df_scores_f["Total Score"].median())
    runs = int(len(df_scores_f))
    models = int(df_scores_f["Model"].nunique())

    c1, c2, c3 = st.columns([1.4, 1.4, 1.0])

    with c1:
        st.metric("Average Total Score", f"{avg_total:.3f}")
        st.progress(_clamp01(avg_total), text="0–1 normalised")

    with c2:
        st.metric("Median Total Score", f"{med_total:.3f}")
        st.progress(_clamp01(med_total), text="0–1 normalised")

    with c3:
        st.metric("Total Number of Queries", runs)
        st.metric("Total Number of Models", models)


# -------------------------
# Aggregation helper
# -------------------------
def _agg_by_model(df_scores_f: pd.DataFrame) -> pd.DataFrame:
    return (
        df_scores_f.groupby("Model")
        .agg(
            runs=("Total Score", "count"),
            mean_total=("Total Score", "mean"),
            median_total=("Total Score", "median"),
            mean_latency=("Norm Latency", "mean"),
            mean_complexity=("Norm Complexity Score", "mean"),
            mean_token=("Norm Token", "mean"),
            mean_semantic=("Norm Semantic", "mean"),
        )
        .reset_index()
        .sort_values("mean_total", ascending=False)
    )


# -------------------------
# Plotly helper: bar chart with labels
# -------------------------
def _bar_with_labels(df: pd.DataFrame, x: str, y: str, title: str, decimals: int = 3):
    d = df.copy()
    d["_label"] = d[y].round(decimals).astype(str)

    fig = px.bar(d, x=x, y=y, text="_label", title=title)
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        yaxis_range=[0, 1],
        margin=dict(l=10, r=10, t=40, b=10),
        uniformtext_minsize=10,
        uniformtext_mode="hide",
    )
    st.plotly_chart(fig, use_container_width=True)


# -------------------------
# TAB 1: SUMMARY
# -------------------------
def render_tab_summary(df_scores_f: pd.DataFrame):
    st.subheader("Model Summary Scores")
    st.markdown(
        "**Total Score** is computed from **4 components with equal weightage**: Latency, Token, Complexity and Semantic."
    )

    agg = _agg_by_model(df_scores_f)

    left, right = st.columns(2)

    with left:
        _bar_with_labels(
            agg,
            x="Model",
            y="mean_total",
            title="Mean Total Score by Model",
        )

    with right:
        _bar_with_labels(
            agg,
            x="Model",
            y="median_total",
            title="Median Total Score by Model",
        )

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
        _bar_with_labels(agg, x="Model", y="mean_latency", title="Mean Norm Latency by Model")

    with c2:
        _bar_with_labels(agg, x="Model", y="mean_token", title="Mean Norm Token by Model")

    with c3:
        _bar_with_labels(agg, x="Model", y="mean_complexity", title="Mean Norm Complexity by Model")

    st.markdown("**Efficiency Profile**")
    prof = agg.set_index("Model")[["mean_latency", "mean_token", "mean_complexity"]].copy()
    prof.columns = ["Latency", "Token", "Complexity"]
    prof_long = prof.reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")

    fig = px.line(prof_long, x="Metric", y="Score", color="Model", markers=True, title="Efficiency Profile")
    fig.update_layout(yaxis_range=[0, 1], margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


# -------------------------
# TAB 3: ACCURACY (Semantic only)
# -------------------------
def _order_complexity_levels(levels):
    norm = [str(x).strip() for x in levels if pd.notna(x)]
    unique = list(dict.fromkeys(norm))

    canonical_orders = [
        ["low", "medium", "high"],
        ["easy", "medium", "hard"],
        ["simple", "moderate", "complex"],
        ["l1", "l2", "l3"],
    ]

    lower_map = {u: u.lower() for u in unique}
    lower_unique = [lower_map[u] for u in unique]

    for order in canonical_orders:
        if all(any(o == lu for lu in lower_unique) for o in order):
            out = []
            for o in order:
                for orig in unique:
                    if lower_map[orig] == o:
                        out.append(orig)
                        break
            for orig in unique:
                if orig not in out:
                    out.append(orig)
            return out

    return sorted(unique)


def render_tab_accuracy(df_scores_f: pd.DataFrame):
    st.subheader("Accuracy")
    st.markdown(
        "- **Semantic**: How well the generated explanation aligns with the NL query intent, shown as a normalised score."
    )

    agg = _agg_by_model(df_scores_f)

    _bar_with_labels(agg, x="Model", y="mean_semantic", title="Mean Norm Semantic by Model")

    st.divider()

    if "Complexity Level" not in df_scores_f.columns:
        st.info("No 'Complexity Level' column found in the Excel, so complexity drilldown is hidden.")
        return

    tmp = df_scores_f[["Model", "Complexity Level", "Norm Semantic"]].copy()
    tmp["Complexity Level"] = tmp["Complexity Level"].astype(str).str.strip()
    tmp = tmp[tmp["Complexity Level"].notna() & (tmp["Complexity Level"] != "")]

    if tmp.empty:
        st.info("Complexity Level exists, but no usable values found after filtering.")
        return

    levels_ordered = _order_complexity_levels(tmp["Complexity Level"].unique().tolist())
    tmp["Complexity Level"] = pd.Categorical(tmp["Complexity Level"], categories=levels_ordered, ordered=True)

    mode = st.radio(
        "Semantic Accuracy by Query Complexity",
        ["Overall with all selected models combined", "Per model"],
        horizontal=True,
        index=0,
    )

    if mode.startswith("Overall"):
        by_lvl = (
            tmp.groupby("Complexity Level", as_index=False)
            .agg(mean_semantic=("Norm Semantic", "mean"), n=("Norm Semantic", "count"))
            .sort_values("Complexity Level")
        )
        by_lvl["_label"] = by_lvl["mean_semantic"].round(3).astype(str)

        fig = px.line(by_lvl, x="Complexity Level", y="mean_semantic", markers=True, text="_label",
                      title="Mean Norm Semantic vs Complexity Level (Overall)")
        fig.update_traces(textposition="top center")
        fig.update_layout(yaxis_range=[0, 1], margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    else:
        by_model_lvl = (
            tmp.groupby(["Model", "Complexity Level"], as_index=False, observed=True)
            .agg(mean_semantic=("Norm Semantic", "mean"))
        )

        # ensure Plotly doesn't choke on categoricals
        by_model_lvl["Complexity Level"] = by_model_lvl["Complexity Level"].astype(str)

        # enforce the intended ordering
        order_map = {lvl: i for i, lvl in enumerate(levels_ordered)}
        by_model_lvl["_order"] = by_model_lvl["Complexity Level"].map(order_map).fillna(10**9)
        by_model_lvl = by_model_lvl.sort_values(["_order", "Model"]).drop(columns=["_order"])

        fig = px.line(
            by_model_lvl,
            x="Complexity Level",
            y="mean_semantic",
            color="Model",
            markers=True,
            title="Norm Semantic across Complexity Level (per model)",
            category_orders={"Complexity Level": levels_ordered},
        )
        fig.update_layout(yaxis_range=[0, 1], margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)