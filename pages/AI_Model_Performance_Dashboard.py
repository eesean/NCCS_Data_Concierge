import streamlit as st
import pandas as pd
from pathlib import Path
from dashboardComponents import (
    compute_filters,
    render_summary_row,
    render_tab_summary,
    render_tab_efficiency,
    render_tab_accuracy,
)

st.set_page_config(page_title="LLM Score Dashboard", layout="wide")
st.title("AI Model Performance Dashboard")

# loading excel
ROOT_DIR = Path(__file__).resolve().parents[1]
MASTER_FILE = ROOT_DIR / "data" / "NCCS_Eval_Own_Test_MasterFile.xlsx"
SHEET_NAME = "Sheet1"

if not MASTER_FILE.exists():
    st.error(f"Missing Excel file: {MASTER_FILE}")
    st.stop()

df_full = pd.read_excel(MASTER_FILE, sheet_name=SHEET_NAME)
df_full_f, df_scores_f = compute_filters(df_full)

# 1 - summary row
render_summary_row(df_scores_f)

st.divider()

# 2 - drilldown for more detailed view
tabs = st.tabs(["Summary", "Efficiency", "Accuracy"])
with tabs[0]:
    render_tab_summary(df_scores_f)
with tabs[1]:
    render_tab_efficiency(df_scores_f)
with tabs[2]:
    render_tab_accuracy(df_scores_f)

st.divider()

# 3 - raw data for debugging
st.subheader("Raw data (All Columns)")
st.dataframe(df_full_f, use_container_width=True, hide_index=True)