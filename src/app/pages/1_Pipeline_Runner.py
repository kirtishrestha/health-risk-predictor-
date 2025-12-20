"""Streamlit page for running the ETL / training / inference pipeline."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

from src.app.ui_pipeline import env_ok, initialize_upload_state, run_command
from src.app.ui_predictions import clear_prediction_cache
from src.app.ui_style import inject_global_css, render_card


st.set_page_config(page_title="Pipeline Runner", page_icon="🚀", layout="wide")

inject_global_css()

st.markdown('<div class="hrp-title">Pipeline Runner</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hrp-subtitle">Run ETL, training, and inference with a clean, guided workflow.</div>',
    unsafe_allow_html=True,
)

env_ready, missing = env_ok()
if not env_ready:
    st.warning(
        "Pipeline actions are disabled because Supabase credentials are missing. "
        f"Set: {', '.join(missing)}"
    )


st.session_state.setdefault("pipeline_last_action", "—")
st.session_state.setdefault("pipeline_last_run_time", "—")
st.session_state.setdefault("pipeline_last_result", "No runs yet")


def _notify(message: str, *, icon: str = "✅", success: bool = True) -> None:
    if hasattr(st, "toast"):
        st.toast(message, icon=icon)
    elif success:
        st.success(message)
    else:
        st.error(message)


def _update_status(action: str, success: bool) -> None:
    st.session_state["pipeline_last_action"] = action
    st.session_state["pipeline_last_run_time"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.session_state["pipeline_last_result"] = "Success" if success else "Needs attention"


def _status_body() -> None:
    status_cols = st.columns(3)
    with status_cols[0]:
        st.markdown("**Last action**")
        st.markdown(st.session_state["pipeline_last_action"])
    with status_cols[1]:
        st.markdown("**Last run time**")
        st.markdown(st.session_state["pipeline_last_run_time"])
    with status_cols[2]:
        st.markdown("**Last result**")
        result = st.session_state["pipeline_last_result"]
        st.markdown(
            f'<span class="status-pill">{result}</span>',
            unsafe_allow_html=True,
        )

render_card(
    "Run status",
    subtitle="Latest pipeline activity at a glance.",
    body_fn=_status_body,
)


def _inputs_body() -> None:
    col_inputs, col_upload = st.columns([2, 1])
    with col_inputs:
        st.text_input("User ID", value="demo_user", key="pipeline_user_id")
        st.selectbox(
            "Source",
            ["fitbit", "fitbit_bella_a", "fitbit_bella_b"],
            index=0,
            key="pipeline_source",
        )

    with col_upload:
        uploaded_zip = st.file_uploader(
            "Upload Fitbit export ZIP",
            type=["zip"],
            accept_multiple_files=False,
            key="pipeline_zip",
        )

    initialize_upload_state(uploaded_zip)

    base_dir = st.session_state.get("uploaded_zip_base_dir")
    csv_count = st.session_state.get("uploaded_zip_csv_count")
    if uploaded_zip is not None:
        if base_dir:
            st.success(f"Detected base folder: {Path(base_dir).name}")
            st.caption(f"CSV files found: {csv_count}")
        else:
            st.error("Could not locate Fitbit CSVs inside the ZIP.")
    else:
        st.info("Upload a Fitbit ZIP export to enable ETL.")


render_card(
    "Inputs",
    subtitle="Provide the user and data source before running the pipeline.",
    body_fn=_inputs_body,
)


def _actions_body() -> None:
    base_dir = st.session_state.get("uploaded_zip_base_dir")
    pipeline_user_id = st.session_state.get("pipeline_user_id", "demo_user")
    pipeline_source = st.session_state.get("pipeline_source", "fitbit")
    col_etl, col_train, col_infer = st.columns(3)
    with col_etl:
        st.markdown('<div class="app-button">', unsafe_allow_html=True)
        run_etl_button = st.button(
            "Run ETL",
            disabled=not env_ready or base_dir is None,
            help="Requires an uploaded Fitbit ZIP.",
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            '<div class="action-helper">Load raw Fitbit exports into Supabase.</div>',
            unsafe_allow_html=True,
        )
    with col_train:
        st.markdown('<div class="app-button">', unsafe_allow_html=True)
        train_button = st.button(
            "Train Models",
            disabled=not env_ready,
            help="Retrain sleep and activity models.",
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            '<div class="action-helper">Refresh sleep + activity models.</div>',
            unsafe_allow_html=True,
        )
    with col_infer:
        st.markdown('<div class="app-button">', unsafe_allow_html=True)
        inference_button = st.button(
            "Run Inference",
            disabled=not env_ready,
            help="Generate new daily predictions.",
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            '<div class="action-helper">Score the latest data for insights.</div>',
            unsafe_allow_html=True,
        )

    if run_etl_button:
        if base_dir is None:
            st.error("ETL requires an uploaded Fitbit ZIP. Please upload a ZIP first.")
        else:
            etl_command = [
                sys.executable,
                "-m",
                "src.pipeline.run_etl",
                "--raw_dir",
                base_dir,
                "--user_id",
                pipeline_user_id,
                "--source",
                pipeline_source,
            ]
            with st.spinner("Running ETL..."):
                success = run_command(etl_command)
            _update_status("ETL", success)
            if success:
                _notify("ETL completed successfully.")
            else:
                _notify("ETL failed. Check the terminal logs for details.", icon="⚠️", success=False)

    if train_button:
        sleep_command = [
            sys.executable,
            "-m",
            "src.ml.train_sleep_quality_model",
            "--source",
            pipeline_source,
            "--all_users",
        ]
        activity_command = [
            sys.executable,
            "-m",
            "src.ml.train_activity_quality_model",
            "--source",
            pipeline_source,
            "--all_users",
        ]
        with st.spinner("Training models..."):
            sleep_success = run_command(sleep_command)
            activity_success = run_command(activity_command)
        success = sleep_success and activity_success
        _update_status("Training", success)
        if success:
            _notify("Model training completed successfully.")
        else:
            _notify("Training finished with errors. Check the terminal logs.", icon="⚠️", success=False)

    if inference_button:
        inference_command = [
            sys.executable,
            "-m",
            "src.ml.run_inference",
            "--source",
            pipeline_source,
            "--user_id",
            pipeline_user_id,
        ]
        with st.spinner("Running inference..."):
            inference_success = run_command(inference_command)
        _update_status("Inference", inference_success)
        if inference_success:
            _notify("Inference completed successfully.")
            clear_prediction_cache()
        else:
            _notify("Inference failed. Check the terminal logs.", icon="⚠️", success=False)


render_card(
    "Run actions",
    subtitle="Choose a step to run. Logs remain in the terminal.",
    body_fn=_actions_body,
)
