"""Streamlit entrypoint and router for the multi-page app."""

from __future__ import annotations

import streamlit as st

from src.app.ui_pipeline import env_ok as pipeline_env_ok
from src.app.ui_style import inject_global_css, render_card


st.set_page_config(
    page_title="Fitbit Health Risk Predictor",
    layout="wide",
)

inject_global_css()

st.markdown('<div class="hrp-title">Fitbit Health Risk Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hrp-subtitle">Warm, focused insights from your Fitbit data pipeline.</div>',
    unsafe_allow_html=True,
)

env_ready, missing = pipeline_env_ok()
if not env_ready:
    render_card(
        "Environment status",
        subtitle="Supabase configuration needed for pipeline actions.",
        body=f"Missing variables: {', '.join(missing)}",
        class_name="filter-card",
    )
else:
    render_card(
        "Environment status",
        subtitle="Supabase configuration detected.",
        body="You're ready to run the pipeline and explore analytics.",
        class_name="filter-card",
    )


def _quick_start_body() -> None:
    st.markdown(
        """
1) Use the **sidebar** to navigate to Pipeline Runner or Analytics Dashboard.  
2) Start with **Pipeline Runner** to upload a Fitbit ZIP and run ETL → Train → Inference.  
3) Visit **Analytics Dashboard** to explore daily predictions stored in Supabase.
"""
    )


render_card(
    "Quick start",
    subtitle="Three steps to get value fast.",
    body_fn=_quick_start_body,
)

st.markdown('<div class="section-title">Pages</div>', unsafe_allow_html=True)

page_cols = st.columns(2)
with page_cols[0]:
    render_card(
        "Pipeline Runner",
        subtitle="Run ETL, training, and inference.",
        body="Upload Fitbit data and keep your models fresh.",
    )
    st.page_link("pages/1_Pipeline_Runner.py", label="Open Pipeline Runner →")
with page_cols[1]:
    render_card(
        "Analytics Dashboard",
        subtitle="Explore daily predictions.",
        body="Monitor trends, insights, and performance KPIs.",
    )
    st.page_link("pages/2_Analytics_Dashboard.py", label="Open Analytics Dashboard →")

with st.sidebar:
    dev_mode = st.checkbox("Developer mode", value=False)
    st.session_state["dev_mode"] = dev_mode

if dev_mode:
    st.page_link(
        "pages/3_Legacy_Dashboard.py",
        label="(Deprecated) Legacy Dashboard",
    )

st.caption("Run with `streamlit run src/app/streamlit_app.py`.")
