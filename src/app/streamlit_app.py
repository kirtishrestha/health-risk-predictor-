"""Streamlit entrypoint and router for the multi-page app."""

from __future__ import annotations

import streamlit as st

from src.app.ui_pipeline import env_ok as pipeline_env_ok


st.set_page_config(
    page_title="Fitbit Health Risk Predictor",
    layout="wide",
)

st.title("Fitbit Health Risk Predictor")
st.markdown(
    """
Welcome! This Streamlit app is split into focused pages for clarity.

### Quick Start
1) Use the **sidebar** to navigate to the Pipeline Runner or Analytics Dashboard pages.
2) Start with **Pipeline Runner** to upload a Fitbit ZIP and run ETL → Train → Inference.
3) Visit **Analytics Dashboard** to explore daily predictions stored in Supabase.
"""
)

env_ready, missing = pipeline_env_ok()
if not env_ready:
    st.info(
        "Supabase environment variables are missing, so pipeline actions are disabled. "
        f"Missing: {', '.join(missing)}"
    )

st.markdown("### Pages")

st.page_link("pages/1_Pipeline_Runner.py", label="Pipeline Runner")
st.page_link("pages/2_Analytics_Dashboard.py", label="Analytics Dashboard")

with st.sidebar:
    dev_mode = st.checkbox("Developer mode", value=False)
    st.session_state["dev_mode"] = dev_mode

if dev_mode:
    st.page_link(
        "pages/3_Legacy_Dashboard.py",
        label="(Deprecated) Legacy Dashboard",
    )

st.caption("Run with `streamlit run src/app/streamlit_app.py`.")
