import streamlit as st
import pandas as pd
from pathlib import Path
from dashboardComponents import (
    compute_filters,
    render_summary_row,
    render_tab_summary,
    render_tab_efficiency,
    render_tab_accuracy,
    render_raw_data_section,
)

st.set_page_config(page_title="LLM Score Dashboard", layout="wide")
st.title("AI Model Performance Dashboard")

ROOT_DIR = Path(__file__).resolve().parents[1]
MASTER_FILE = ROOT_DIR / "eval_files" / "nccs_evaluation_results.csv"

if not MASTER_FILE.exists():
    st.error(f"Missing CSV file: {MASTER_FILE}")
    st.stop()

df_full = pd.read_csv(MASTER_FILE)

# Step 1: sidebar filters
df_full_f, df_scores_f = compute_filters(df_full)

# Step 2: apply any checkbox exclusions from session state
excluded = st.session_state.get("excluded_idx", set())
df_scores_final = df_scores_f.loc[~df_scores_f.index.isin(excluded)].copy()

if df_scores_final.empty:
    st.warning("All rows are unchecked — tick at least one row in the Raw Data table.")
    st.stop()

# Step 3: summary metrics
render_summary_row(df_scores_final)

st.divider()

# Step 4: tabs
tabs = st.tabs(["Summary", "Efficiency", "Accuracy"])
with tabs[0]:
    render_tab_summary(df_scores_final)
with tabs[1]:
    render_tab_efficiency(df_scores_final)
with tabs[2]:
    render_tab_accuracy(df_scores_final)

st.divider()

# Step 5: raw data at the bottom with checkboxes — st.rerun() fires on any change
render_raw_data_section(df_full_f)