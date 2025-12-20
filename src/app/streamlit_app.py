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
1) Upload your Fitbit export in **Pipeline Runner**.  
2) Run **ETL → Train → Inference** to refresh predictions.  
3) Review insights in **Analytics Dashboard** and **Legacy Dashboard**.
"""
    )


render_card(
    "Quick start",
    subtitle="Three steps to get value fast.",
    body_fn=_quick_start_body,
)

st.markdown('<div class="section-title">Pages</div>', unsafe_allow_html=True)

page_cols = st.columns(3)
with page_cols[0]:
    render_card(
        "Pipeline Runner",
        subtitle="Run ETL, training, and inference.",
        body="Upload Fitbit data, retrain models, and produce predictions.",
    )
    st.page_link("pages/1_Pipeline_Runner.py", label="Open Pipeline Runner →")
with page_cols[1]:
    render_card(
        "Analytics Dashboard",
        subtitle="Explore daily predictions.",
        body="Monitor trends, KPIs, and probability insights from Supabase.",
    )
    st.page_link("pages/2_Analytics_Dashboard.py", label="Open Analytics Dashboard →")
with page_cols[2]:
    render_card(
        "Legacy Dashboard",
        subtitle="Legacy CSV-based experience.",
        body="Review local metrics and legacy model outputs.",
    )
    st.page_link("pages/3_Legacy_Dashboard.py", label="Open Legacy Dashboard →")

st.caption("Run with `streamlit run src/app/streamlit_app.py`.")
