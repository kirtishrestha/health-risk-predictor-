"""Streamlit page for running the ETL / training / inference pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

from src.app.ui_pipeline import env_ok, initialize_upload_state, stream_command_logs
from src.app.ui_predictions import clear_prediction_cache


st.set_page_config(page_title="Pipeline Runner", page_icon="🚀", layout="wide")

st.title("🚀 Pipeline Runner")
st.caption(
    "Run the CLI pipeline with streaming logs. Supabase credentials are required for ETL, "
    "training, and inference."
)

env_ready, missing = env_ok()
if not env_ready:
    st.warning(
        "Pipeline actions are disabled because Supabase credentials are missing. "
        f"Set: {', '.join(missing)}"
    )

st.markdown("### Configure inputs")

col_inputs, col_upload = st.columns([2, 1])
with col_inputs:
    pipeline_user_id = st.text_input("user_id", value="demo_user")
    pipeline_source = st.selectbox(
        "source",
        ["fitbit", "fitbit_bella_a", "fitbit_bella_b"],
        index=0,
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

st.markdown("### Run steps")

col_etl, col_train, col_infer = st.columns(3)
run_etl_button = col_etl.button(
    "1) Run ETL",
    disabled=not env_ready or base_dir is None,
    help="Requires an uploaded Fitbit ZIP.",
)
train_button = col_train.button("2) Train Models", disabled=not env_ready)
inference_button = col_infer.button(
    "3) Run Inference",
    disabled=not env_ready,
)

log_container = st.container()

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
        with log_container:
            st.subheader("ETL logs")
            success, _ = stream_command_logs(etl_command, "etl_log")
        if success:
            st.success("ETL completed successfully.")
        else:
            st.error("ETL failed. Check logs above.")

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
    with log_container:
        st.subheader("Sleep model training logs")
        sleep_success, _ = stream_command_logs(sleep_command, "train_sleep_log")
        st.subheader("Activity model training logs")
        activity_success, _ = stream_command_logs(
            activity_command, "train_activity_log"
        )
    if sleep_success and activity_success:
        st.success("All model training runs completed successfully.")
    else:
        st.warning("Training completed with errors. Review the logs for details.")

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
    with log_container:
        st.subheader("Inference logs")
        inference_success, _ = stream_command_logs(
            inference_command, "inference_log"
        )
    if inference_success:
        st.success("Inference completed successfully.")
        clear_prediction_cache()
    else:
        st.error("Inference failed. Check logs above.")

if st.session_state.get("etl_log"):
    with log_container:
        st.subheader("ETL logs (last run)")
        st.code(st.session_state.get("etl_log", ""))
if st.session_state.get("train_sleep_log"):
    with log_container:
        st.subheader("Sleep model training logs (last run)")
        st.code(st.session_state.get("train_sleep_log", ""))
if st.session_state.get("train_activity_log"):
    with log_container:
        st.subheader("Activity model training logs (last run)")
        st.code(st.session_state.get("train_activity_log", ""))
if st.session_state.get("inference_log"):
    with log_container:
        st.subheader("Inference logs (last run)")
        st.code(st.session_state.get("inference_log", ""))
