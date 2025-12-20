"""Streamlit entrypoint and router for the multi-page app."""

from __future__ import annotations

import streamlit as st

from src.app.ui_pipeline import env_ok as pipeline_env_ok


st.set_page_config(
    page_title="Fitbit Health Risk Predictor",
    page_icon="🩺",
    layout="wide",
)

st.title("Fitbit Health Risk Predictor")
st.markdown(
    """
Welcome! This Streamlit app is now split into focused pages for clarity.

**Quickstart**
1) Go to **🚀 Pipeline Runner** to upload a Fitbit ZIP and run ETL → Train → Inference.
2) Visit **📊 Predictions Explorer** to view daily predictions stored in Supabase.
3) (Optional) Use the **🧰 Legacy Dashboard** for local CSV + model outputs (deprecated).
"""
)

env_ready, missing = pipeline_env_ok()
if not env_ready:
    st.info(
        "Supabase environment variables are missing, so pipeline actions are disabled. "
        f"Missing: {', '.join(missing)}"
    )

st.markdown("### Pages")
st.page_link("pages/1_🚀_Pipeline_Runner.py", label="🚀 Pipeline Runner")
st.page_link("pages/2_📊_Predictions_Explorer.py", label="📊 Predictions Explorer")
st.page_link("pages/3_🧰_Legacy_Dashboard.py", label="🧰 Legacy Dashboard (Deprecated)")

st.caption("Run with `streamlit run src/app/streamlit_app.py`.")
