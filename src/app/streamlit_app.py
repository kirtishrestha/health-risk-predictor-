"""Streamlit entrypoint and router for the multi-page app."""

from __future__ import annotations

import streamlit as st

from src.app.ui_pipeline import env_ok as pipeline_env_ok


st.set_page_config(page_title="Fitbit Health Risk Predictor", layout="wide")

st.title("Fitbit Health Risk Predictor")
st.caption("Warm, focused insights from your Fitbit data pipeline.")

env_ready, missing = pipeline_env_ok()
st.header("Environment status")
if not env_ready:
    st.warning(
        "Supabase configuration needed for pipeline actions. "
        f"Missing variables: {', '.join(missing)}"
    )
else:
    st.success("Supabase configuration detected. You're ready to run the pipeline.")


def _quick_start_body() -> None:
    st.markdown(
        """
1) Upload your Fitbit export in **Pipeline Runner**.  
2) Run **ETL → Train → Inference** to refresh predictions.  
3) Review insights in **Analytics Dashboard** and **Legacy Dashboard**.
"""
    )


st.markdown("---")
st.header("Quick start")
st.subheader("Three steps to get value fast.")
_quick_start_body()

st.markdown("---")
st.header("Pages")

page_cols = st.columns(3)
with page_cols[0]:
    st.subheader("Pipeline Runner")
    st.write("Run ETL, training, and inference.")
    st.caption("Upload Fitbit data, retrain models, and produce predictions.")
    st.page_link("pages/1_Pipeline_Runner.py", label="Open Pipeline Runner →")
with page_cols[1]:
    st.subheader("Analytics Dashboard")
    st.write("Explore daily predictions.")
    st.caption("Monitor trends, KPIs, and probability insights from Supabase.")
    st.page_link("pages/2_Analytics_Dashboard.py", label="Open Analytics Dashboard →")
with page_cols[2]:
    st.subheader("Legacy (Read-only)")
    st.write("Legacy CSV-based experience.")
    st.caption("Review local metrics and legacy model outputs.")
    st.page_link("pages/3_Legacy_Dashboard.py", label="Open Legacy Dashboard →")

st.caption("Run with `streamlit run src/app/streamlit_app.py`.")
