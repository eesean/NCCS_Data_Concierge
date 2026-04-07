import streamlit as st
import pandas as pd
from pathlib import Path
from dashboardComponents import (
    configure_accuracy_metric,
    compute_filters,
    render_summary_row,
    render_tab_summary,
    render_tab_efficiency,
    render_tab_accuracy,
    render_raw_data_section,
)

st.set_page_config(page_title="LLM Score Dashboard", layout="wide")
st.title("AI Model Performance Dashboard")

# 1. Setup Directory Path
ROOT_DIR = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT_DIR / "eval_files"

# 2. Scan for all CSV files
if not EVAL_DIR.exists():
    st.error(f"Directory not found: {EVAL_DIR}")
    st.stop()

# Get list of filenames (sorted by newest first usually helps)
csv_files = sorted([f.name for f in EVAL_DIR.glob("*.csv")], reverse=True)

if not csv_files:
    st.warning(f"No CSV files found in {EVAL_DIR}")
    st.stop()

# 3. Add the Dropdown at the top
selected_filename = st.selectbox(
    "Select Evaluation Run",
    options=csv_files,
    index=0,  # Defaults to the first (newest) file
    help="Scanning 'eval_files' directory for results..."
)

# 4. Set the MASTER_FILE based on selection
MASTER_FILE = EVAL_DIR / selected_filename


df_full = pd.read_csv(MASTER_FILE)

# Step 0: detect accuracy metric — use F1 if available, else Semantic
configure_accuracy_metric(df_full)

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