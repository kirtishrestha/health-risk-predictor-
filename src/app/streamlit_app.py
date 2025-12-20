"""Streamlit dashboard for exploring Fitbit daily metrics and risk predictions."""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

from src.aggregation.monthly_aggregate import run_monthly_aggregation
from src.etl.pandas_metrics import build_daily_metrics_pandas, find_fitbit_base_dir
from src.ingestion.upload_supabase import (
    SupabaseConfigError,
    get_supabase_client,
)
from src.pipeline.run_etl import run_etl


REQUIRED_COLUMNS = [
    "id",
    "date",
    "total_steps",
    "total_distance",
    "very_active_minutes",
    "fairly_active_minutes",
    "lightly_active_minutes",
    "sedentary_minutes",
    "calories",
    "total_minutes_asleep",
    "total_time_in_bed",
    "sleep_efficiency",
    "avg_hr",
    "max_hr",
    "min_hr",
    "active_minutes",
]

FEATURE_COLUMNS = [
    "total_steps",
    "total_distance",
    "very_active_minutes",
    "fairly_active_minutes",
    "lightly_active_minutes",
    "sedentary_minutes",
    "calories",
    "total_minutes_asleep",
    "total_time_in_bed",
    "sleep_efficiency",
    "avg_hr",
    "max_hr",
    "min_hr",
    "active_minutes",
]

MODEL_FILES = {
    "health": "model_health.pkl",
    "health_logreg": "model_health_logreg.pkl",
    "cardio": "model_cardio.pkl",
    "sleep_reg": "model_sleep_reg.pkl",
    "stress": "model_stress.pkl",
}

LABEL_ENCODER_FILES = {
    "health": "le_health.pkl",
    "health_logreg": "le_health_logreg.pkl",
    "cardio": "le_cardio.pkl",
    "stress": "le_stress.pkl",
}

HEALTH_MODEL_OPTIONS = {
    "Random Forest": "health",
    "Logistic Regression": "health_logreg",
}


@st.cache_data(show_spinner=False)
def load_daily_metrics() -> pd.DataFrame:
    """Load processed daily metrics from CSV, preferring combined dataset."""

    combined_path = Path("data/processed/daily_metrics_combined.csv")
    default_path = Path("data/processed/daily_metrics.csv")

    if combined_path.exists():
        data_path = combined_path
    elif default_path.exists():
        st.warning(
            "Combined daily metrics not found; falling back to bella_b-only dataset."
        )
        data_path = default_path
    else:
        st.error(
            "Processed metrics file not found at"
            f" {combined_path} or {default_path}."
        )
        return pd.DataFrame()

    try:
        df = pd.read_csv(data_path, parse_dates=["date"])
    except Exception as exc:  # pragma: no cover - surface any load issues
        st.error(f"Failed to load daily metrics: {exc}")
        return pd.DataFrame()

    if "source" not in df.columns:
        df["source"] = "fitbit_bella_b"

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        st.error(
            "Daily metrics file is missing required columns: "
            + ", ".join(missing_cols)
        )
        return pd.DataFrame()

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.dropna(subset=["date"]).copy()
    return df


@st.cache_resource(show_spinner=False)
def load_models() -> Dict[str, object]:
    """Load trained models from disk."""

    models_dir = Path("models")
    models: Dict[str, object] = {}
    for key, filename in MODEL_FILES.items():
        model_path = models_dir / filename
        if not model_path.exists():
            st.error(f"Model file not found: {model_path}")
            continue
        try:
            with model_path.open("rb") as f:
                models[key] = pickle.load(f)
        except Exception as exc:  # pragma: no cover - display load failure
            st.error(f"Failed to load {key} model: {exc}")
    return models


@st.cache_resource(show_spinner=False)
def load_label_encoders() -> Dict[str, object]:
    """Load label encoders for each prediction target."""

    encoders_dir = Path("models")
    encoders: Dict[str, object] = {}
    for key, filename in LABEL_ENCODER_FILES.items():
        encoder_path = encoders_dir / filename
        if not encoder_path.exists():
            st.error(f"Label encoder not found: {encoder_path}")
            continue
        try:
            with encoder_path.open("rb") as f:
                encoders[key] = pickle.load(f)
        except Exception as exc:  # pragma: no cover - display load failure
            st.error(f"Failed to load {key} label encoder: {exc}")
    return encoders


@st.cache_data(show_spinner=False)
def load_model_metrics() -> Dict | None:
    """Load model metrics JSON if available."""

    metrics_path = Path("models/model_metrics.json")
    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as exc:  # pragma: no cover - surface load issues
        st.info(f"Unable to load model metrics: {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_model_metadata() -> Dict | None:
    """Load model metadata JSON if available."""

    metadata_path = Path("models/model_metadata.json")
    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as exc:  # pragma: no cover - surface load issues
        st.info(f"Unable to load model metadata: {exc}")
        return None


def interpret_sleep_score(score: float) -> str | float:
    """Map numeric sleep quality into a simple interpretation label."""

    if pd.isna(score):
        return np.nan
    if score < 50:
        return "Poor"
    if score < 75:
        return "Moderate"
    return "Good"


def load_uploaded_daily_metrics(uploaded_zip) -> Optional[pd.DataFrame]:
    """Build daily metrics from an uploaded Fitbit ZIP without touching repo data."""

    if uploaded_zip is None:
        return None

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            uploaded_zip.seek(0)
            with zipfile.ZipFile(uploaded_zip) as zip_file:
                zip_file.extractall(tmp_path)

            base_dir = find_fitbit_base_dir(tmp_path)
            if base_dir is None:
                st.error(
                    "Could not locate Fitbit CSVs inside the ZIP. Ensure it contains the Fitabase export folder."
                )
                return None

            df = build_daily_metrics_pandas(base_dir, source_label="uploaded_zip")
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
            if missing_cols:
                st.error(
                    "Uploaded data is missing required columns: " + ", ".join(missing_cols)
                )
                return None

            return df.dropna(subset=["date"]).sort_values("date")
    except zipfile.BadZipFile:
        st.error("The uploaded file is not a valid ZIP archive.")
    except Exception as exc:  # pragma: no cover - display any unexpected parsing failures
        st.error(f"Failed to process uploaded Fitbit data: {exc}")
    return None


def ensure_supabase_env() -> None:
    """Ensure required Supabase credentials are available."""

    if not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_SERVICE_ROLE_KEY"):
        st.error(
            "Supabase credentials are missing. Set SUPABASE_URL and "
            "SUPABASE_SERVICE_ROLE_KEY before using the demo app."
        )
        st.stop()


def _clear_prediction_cache() -> None:
    """Clear cached prediction data."""

    load_daily_predictions.clear()


def _initialize_upload_state(uploaded_zip) -> None:
    """Extract a newly uploaded ZIP and store metadata in session state."""

    if uploaded_zip is None:
        st.session_state.pop("uploaded_zip_name", None)
        st.session_state.pop("uploaded_zip_dir", None)
        st.session_state.pop("uploaded_zip_base_dir", None)
        st.session_state.pop("uploaded_zip_csv_count", None)
        return

    if st.session_state.get("uploaded_zip_name") == uploaded_zip.name:
        return

    tmp_dir = Path(tempfile.mkdtemp(prefix="fitbit_upload_"))
    uploaded_zip.seek(0)
    with zipfile.ZipFile(uploaded_zip) as zip_file:
        zip_file.extractall(tmp_dir)

    base_dir = find_fitbit_base_dir(tmp_dir)
    csv_count = len(list(tmp_dir.rglob("*.csv")))

    st.session_state["uploaded_zip_name"] = uploaded_zip.name
    st.session_state["uploaded_zip_dir"] = str(tmp_dir)
    st.session_state["uploaded_zip_base_dir"] = str(base_dir) if base_dir else None
    st.session_state["uploaded_zip_csv_count"] = csv_count


def _stream_command_logs(command: list[str], log_key: str) -> Tuple[bool, str]:
    """Run a CLI command and stream logs into the UI."""

    st.session_state[log_key] = ""
    placeholder = st.empty()

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines: list[str] = []
    if process.stdout:
        for line in process.stdout:
            output_lines.append(line.rstrip())
            st.session_state[log_key] = "\n".join(output_lines)
            placeholder.code(st.session_state[log_key])

    return_code = process.wait()
    return return_code == 0, st.session_state[log_key]


def get_supabase_client_or_error():
    """Create a Supabase client or report errors in the UI."""

    try:
        return get_supabase_client()
    except SupabaseConfigError as exc:
        st.error(
            "Supabase credentials are missing. "
            "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY before running ingestion."
        )
        st.info(str(exc))
    except Exception as exc:  # pragma: no cover - display unexpected Supabase failures
        st.error(f"Unable to initialize Supabase client: {exc}")
    return None


def fetch_supabase_counts(client, user_id: str, source: str) -> Dict[str, int | None]:
    """Fetch row counts for key tables filtered by user_id/source."""

    tables = ["daily_activity", "daily_sleep", "daily_features", "monthly_metrics"]
    counts: Dict[str, int | None] = {}
    for table in tables:
        try:
            response = (
                client.table(table)
                .select("id", count="exact")
                .eq("user_id", user_id)
                .eq("source", source)
                .execute()
            )
            counts[table] = response.count
        except Exception as exc:  # pragma: no cover - surface query issues
            st.error(f"Failed to fetch count for {table}: {exc}")
            counts[table] = None
    return counts


def fetch_daily_features_preview(client, user_id: str, source: str) -> pd.DataFrame:
    """Fetch a preview of daily_features rows after ingestion."""

    try:
        response = (
            client.table("daily_features")
            .select("*")
            .eq("user_id", user_id)
            .eq("source", source)
            .order("date", desc=True)
            .limit(5)
            .execute()
        )
        return pd.DataFrame(response.data or [])
    except Exception as exc:  # pragma: no cover - surface query issues
        st.error(f"Failed to load daily_features preview: {exc}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_daily_predictions(user_id: str, source: str) -> pd.DataFrame:
    """Load daily predictions from Supabase for a specific user/source."""

    client = get_supabase_client()
    try:
        response = (
            client.table("daily_predictions")
            .select(
                ",".join(
                    [
                        "user_id",
                        "date",
                        "source",
                        "sleep_quality_label",
                        "sleep_quality_proba",
                        "activity_quality_label",
                        "activity_quality_proba",
                        "created_at",
                    ]
                )
            )
            .eq("user_id", user_id)
            .eq("source", source)
            .order("date", desc=False)
            .execute()
        )
        df = pd.DataFrame(response.data or [])
    except Exception as exc:  # pragma: no cover - surface query issues
        st.error(f"Unable to load daily_predictions: {exc}")
        return pd.DataFrame()
    if df.empty:
        return df
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df.sort_values("date")


def add_predictions(
    df: pd.DataFrame, models: Dict[str, object], label_encoders: Dict[str, object], *, health_model_key: str
) -> pd.DataFrame:
    """Generate predictions and append them to the dataframe."""

    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        st.error(
            "Cannot compute predictions because columns are missing: "
            + ", ".join(missing_features)
        )
        return df.copy()

    df_pred = df.copy()
    X_base = df_pred[FEATURE_COLUMNS]
    X_for_lr = X_base.fillna(0.0)

    model_targets = {
        "cardio": "predicted_cardiovascular_strain_risk",
        "stress": "predicted_stress_risk",
    }

    health_encoder_key = "health_logreg" if health_model_key == "health_logreg" else "health"
    health_model = models.get(health_model_key)
    health_encoder = label_encoders.get(health_encoder_key)
    X_for_health = X_for_lr if health_model_key == "health_logreg" else X_base
    if health_model is None or health_encoder is None:
        st.error("Missing model or label encoder for health predictions.")
    else:
        try:
            # Logistic Regression cannot handle NaNs, so fill them for that model only.
            predictions = health_model.predict(X_for_health)
            decoded = health_encoder.inverse_transform(predictions)
            df_pred["predicted_health_risk_level"] = decoded
        except Exception as exc:  # pragma: no cover - display prediction failure
            st.error(f"Failed to compute health predictions: {exc}")

    for target, output_col in model_targets.items():
        model = models.get(target)
        encoder = label_encoders.get(target)
        if model is None or encoder is None:
            st.error(f"Missing model or label encoder for {target} predictions.")
            continue
        try:
            predictions = model.predict(X_base)
            decoded = encoder.inverse_transform(predictions)
            df_pred[output_col] = decoded
        except Exception as exc:  # pragma: no cover - display prediction failure
            st.error(f"Failed to compute {target} predictions: {exc}")

    sleep_model = models.get("sleep_reg")
    if sleep_model is None:
        st.error("Missing sleep quality regression model.")
    else:
        try:
            sleep_scores = sleep_model.predict(X_base.fillna(0.0))
            df_pred["predicted_sleep_quality_score"] = np.clip(sleep_scores, 0, 100)
            df_pred["predicted_sleep_quality_label"] = df_pred[
                "predicted_sleep_quality_score"
            ].apply(interpret_sleep_score)
            # Keep a risk-style column for backward compatibility with logs/visuals.
            df_pred["predicted_sleep_quality_risk"] = df_pred[
                "predicted_sleep_quality_label"
            ]
        except Exception as exc:  # pragma: no cover - display prediction failure
            st.error(f"Failed to compute sleep quality score: {exc}")
    return df_pred


@st.cache_resource(show_spinner=False)
def get_engine():
    """Create and cache a SQLAlchemy engine using SUPABASE_DB_URL."""

    db_url = os.getenv("SUPABASE_DB_URL")
    if not db_url:
        return None
    try:
        return create_engine(db_url)
    except Exception:
        return None


def log_prediction(
    engine,
    user_id,
    health_model_name: str,
    cardio_model_name: str,
    sleep_model_name: str,
    stress_model_name: str,
    last_row: pd.Series,
) -> None:
    """Persist prediction summary to the prediction_logs table."""

    if engine is None:
        return

    required_cols = [
        "predicted_health_risk_level",
        "predicted_cardiovascular_strain_risk",
        "predicted_sleep_quality_risk",
        "predicted_stress_risk",
    ]
    if any(col not in last_row for col in required_cols):
        return

    schema = os.getenv("SUPABASE_SCHEMA", "public")
    try:
        log_df = pd.DataFrame(
            [
                {
                    "source": "dashboard",
                    "user_id": user_id,
                    "model_health": health_model_name,
                    "model_cardio": cardio_model_name,
                    "model_sleep": sleep_model_name,
                    "model_stress": stress_model_name,
                    "predicted_health_risk_level": last_row["predicted_health_risk_level"],
                    "predicted_cardio_risk": last_row[
                        "predicted_cardiovascular_strain_risk"
                    ],
                    "predicted_sleep_risk": last_row["predicted_sleep_quality_risk"],
                    "predicted_stress_risk": last_row["predicted_stress_risk"],
                }
            ]
        )
        log_df.to_sql(
            "prediction_logs",
            engine,
            schema=schema,
            if_exists="append",
            index=False,
        )
    except Exception as exc:  # pragma: no cover - best-effort logging
        st.info(f"Prediction logging skipped: {exc}")


def render_dashboard() -> None:
    """Render the Streamlit dashboard layout."""

    st.title("Fitbit Health Risk Dashboard")

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
        [data-testid="stMetric"] { background: #ffffff; padding: 0.75rem; border-radius: 0.75rem;
            border: 1px solid rgba(79, 70, 229, 0.15); box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05); }
        .section-header { font-size: 1.25rem; font-weight: 700; margin-top: 1.25rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Models are trained on labeled historical days (labels come from heuristic rules) "
        "and then used to predict risk levels on new, unseen days."
    )

    st.markdown(
        "#### Quick start\n"
        "1) Upload a Fitbit ZIP (optional).\n"
        "2) Run ETL → Train → Inference from the sidebar.\n"
        "3) Explore predictions and charts below.\n"
    )

    ensure_supabase_env()

    st.sidebar.markdown("## 🚀 Pipeline Runner")
    pipeline_user_id = st.sidebar.text_input(
        "user_id", value="demo_user", key="pipeline_user_id"
    )
    pipeline_source = st.sidebar.selectbox(
        "source",
        ["fitbit", "fitbit_bella_a", "fitbit_bella_b"],
        index=0,
        key="pipeline_source",
    )

    uploaded_zip = st.sidebar.file_uploader(
        "Upload Fitbit export ZIP",
        type=["zip"],
        accept_multiple_files=False,
        key="pipeline_zip",
    )
    _initialize_upload_state(uploaded_zip)

    base_dir = st.session_state.get("uploaded_zip_base_dir")
    csv_count = st.session_state.get("uploaded_zip_csv_count")
    if uploaded_zip is not None:
        if base_dir:
            st.sidebar.success(f"Detected base folder: {Path(base_dir).name}")
            st.sidebar.caption(f"CSV files found: {csv_count}")
        else:
            st.sidebar.error("Could not locate Fitbit CSVs inside the ZIP.")

    st.sidebar.caption("Run each step in order. Logs appear below.")

    run_etl_button = st.sidebar.button(
        "1) Run ETL to Supabase", disabled=base_dir is None
    )
    train_button = st.sidebar.button("2) Train Models (All Users)")
    inference_button = st.sidebar.button("3) Run Inference (This user_id)")

    ingest_mode = st.sidebar.radio(
        "Upload handling",
        ["Preview only (in-memory)", "Ingest to Supabase (run ETL)"],
    )
    ingest_user_id = st.sidebar.text_input(
        "user_id", value=pipeline_user_id, key="ingest_user_id"
    )
    ingest_source = st.sidebar.text_input(
        "source", value=pipeline_source, key="ingest_source"
    )
    run_monthly = st.sidebar.checkbox("Run monthly aggregation after ETL", value=True)

    using_uploaded = uploaded_zip is not None and ingest_mode == "Preview only (in-memory)"

    st.markdown('<div class="section-header">Pipeline logs</div>', unsafe_allow_html=True)
    pipeline_log_container = st.container()

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
            with pipeline_log_container:
                st.subheader("ETL logs")
                success, _ = _stream_command_logs(etl_command, "etl_log")
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
        with pipeline_log_container:
            st.subheader("Sleep model training logs")
            sleep_success, _ = _stream_command_logs(
                sleep_command, "train_sleep_log"
            )
            st.subheader("Activity model training logs")
            activity_success, _ = _stream_command_logs(
                activity_command, "train_activity_log"
            )
        if sleep_success and activity_success:
            st.success("All model training runs completed successfully.")
        else:
            st.warning(
                "Training completed with errors. Review the logs for details."
            )

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
        with pipeline_log_container:
            st.subheader("Inference logs")
            inference_success, _ = _stream_command_logs(
                inference_command, "inference_log"
            )
        if inference_success:
            st.success("Inference completed successfully.")
            _clear_prediction_cache()
            st.session_state["force_refresh"] = True
        else:
            st.error("Inference failed. Check logs above.")

    if st.session_state.get("etl_log"):
        with pipeline_log_container:
            st.subheader("ETL logs (last run)")
            st.code(st.session_state.get("etl_log", ""))
    if st.session_state.get("train_sleep_log"):
        with pipeline_log_container:
            st.subheader("Sleep model training logs (last run)")
            st.code(st.session_state.get("train_sleep_log", ""))
    if st.session_state.get("train_activity_log"):
        with pipeline_log_container:
            st.subheader("Activity model training logs (last run)")
            st.code(st.session_state.get("train_activity_log", ""))
    if st.session_state.get("inference_log"):
        with pipeline_log_container:
            st.subheader("Inference logs (last run)")
            st.code(st.session_state.get("inference_log", ""))

    if st.session_state.get("force_refresh"):
        st.session_state["force_refresh"] = False
        st.rerun()

    if using_uploaded:
        st.sidebar.info("Processing only the data inside your uploaded ZIP (kept in-memory).")
        df = load_uploaded_daily_metrics(uploaded_zip)
        if df is None or df.empty:
            st.warning("Uploaded ZIP could not be processed or contained no records.")
            return
    else:
        df = load_daily_metrics()
        if df.empty:
            st.warning("No data available to display.")
            return

    df = df.sort_values("date")
    metrics_data = load_model_metrics()
    metadata = load_model_metadata()
    engine = get_engine()

    if using_uploaded:
        st.sidebar.success("Using uploaded Fitbit data.")
    else:
        source_options = ["All"] + sorted(df["source"].dropna().unique())
        selected_source = st.sidebar.selectbox("Select data source", source_options)
        if selected_source != "All":
            df = df[df["source"] == selected_source]

        if df.empty:
            st.warning("No data available for the selected source.")
            return

    user_ids = sorted(df["id"].unique())
    selected_user = st.sidebar.selectbox("Select user ID", user_ids)

    if uploaded_zip is not None and ingest_mode == "Ingest to Supabase (run ETL)":
        st.subheader("Supabase ingestion")
        st.write(
            "Upload a Fitbit ZIP and run the ETL pipeline to load canonical tables into Supabase."
        )
        if st.button("Run ETL", type="primary"):
            with st.spinner("Running ETL..."):
                try:
                    supabase_client = get_supabase_client_or_error()
                    if supabase_client is None:
                        raise RuntimeError("Supabase client unavailable.")

                    with tempfile.TemporaryDirectory() as tmpdir:
                        tmp_path = Path(tmpdir)
                        uploaded_zip.seek(0)
                        with zipfile.ZipFile(uploaded_zip) as zip_file:
                            zip_file.extractall(tmp_path)

                        base_dir = find_fitbit_base_dir(tmp_path)
                        if base_dir is None:
                            raise ValueError(
                                "Could not locate Fitbit CSVs inside the ZIP. "
                                "Ensure it contains the Fitabase export folder with "
                                "dailyActivity_merged.csv or sleepDay_merged.csv."
                            )

                        run_etl(raw_dir=str(base_dir), user_id=ingest_user_id, source=ingest_source)

                    if run_monthly:
                        run_monthly_aggregation(user_id=ingest_user_id, source=ingest_source)

                    st.session_state["etl_status"] = "success"
                except Exception as exc:  # pragma: no cover - surface ETL issues
                    st.session_state["etl_status"] = "failed"
                    st.session_state["etl_error"] = str(exc)

        status = st.session_state.get("etl_status")
        if status == "success":
            st.success("ETL succeeded.")
            supabase_client = get_supabase_client_or_error()
            if supabase_client is not None:
                counts = fetch_supabase_counts(
                    supabase_client, ingest_user_id, ingest_source
                )
                st.write("Supabase row counts (filtered by user_id/source):")
                st.json(counts)

                st.write("daily_features preview (first 5 rows):")
                preview_df = fetch_daily_features_preview(
                    supabase_client, ingest_user_id, ingest_source
                )
                if preview_df.empty:
                    st.info("No daily_features rows found for the selected user/source.")
                else:
                    st.dataframe(preview_df)
        elif status == "failed":
            st.error("ETL failed.")
            error_msg = st.session_state.get("etl_error")
            if error_msg:
                st.info(f"Error details: {error_msg}")

    st.sidebar.markdown("### Daily Predictions Filters")
    prediction_source = st.sidebar.text_input(
        "Prediction source", value=pipeline_source, key="prediction_source"
    )
    prediction_user_id = st.sidebar.text_input(
        "Prediction user_id", value=pipeline_user_id, key="prediction_user_id"
    )
    if st.sidebar.button("Refresh predictions"):
        _clear_prediction_cache()

    prediction_df = load_daily_predictions(prediction_user_id, prediction_source)

    pred_start = pred_end = None
    if not prediction_df.empty:
        min_pred_date = prediction_df["date"].min().date()
        max_pred_date = prediction_df["date"].max().date()
        pred_dates = st.sidebar.date_input(
            "Prediction date range",
            value=(min_pred_date, max_pred_date),
            min_value=min_pred_date,
            max_value=max_pred_date,
            key="prediction_date_range",
        )
        if isinstance(pred_dates, tuple) and len(pred_dates) == 2:
            pred_start, pred_end = pred_dates
        else:
            pred_start = pred_end = pred_dates

    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    selected_dates = st.sidebar.date_input(
        "Select date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
    )

    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date = end_date = selected_dates

    mask_user = df["id"] == selected_user
    mask_dates = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
    df_filtered = df[mask_user & mask_dates]

    if df_filtered.empty:
        st.warning("No records found for the selected filters.")
        return

    models = load_models()
    encoders = load_label_encoders()
    if not models or not encoders:
        st.error("Models or label encoders are unavailable. Please check the models directory.")
        return

    available_health_options = {
        label: key
        for label, key in HEALTH_MODEL_OPTIONS.items()
        if key in models and key in encoders
    }
    if not available_health_options:
        st.error("No health risk models are available. Please verify training outputs.")
        return

    health_model_label = st.sidebar.selectbox(
        "Select health risk model",
        options=list(available_health_options.keys()),
    )
    selected_health_model = available_health_options[health_model_label]

    df_pred = add_predictions(
        df_filtered, models, encoders, health_model_key=selected_health_model
    )

    with st.expander("Model performance summary", expanded=False):
        if metrics_data and metrics_data.get("metrics"):
            perf_df = pd.DataFrame(metrics_data["metrics"])
            summary_cols = [
                col
                for col in ["target", "model", "accuracy", "macro_f1", "roc_auc_ovr_macro"]
                if col in perf_df.columns
            ]
            st.subheader("Model summary (per target and model)")
            st.dataframe(perf_df[summary_cols])
            st.caption(
                "Accuracy can mask class imbalance. Macro F1, macro ROC AUC (one-vs-rest), "
                "and confusion matrices provide a more balanced view across classes."
            )
            if not perf_df.empty:
                try:
                    classification_df = perf_df.dropna(subset=["accuracy"])
                    if not classification_df.empty:
                        st.subheader("Model accuracy by target")
                        accuracy_pivot = classification_df.pivot(
                            index="model", columns="target", values="accuracy"
                        )
                        st.bar_chart(accuracy_pivot)

                        st.subheader("Macro F1 by target")
                        f1_pivot = classification_df.pivot(
                            index="model", columns="target", values="macro_f1"
                        )
                        st.bar_chart(f1_pivot)
                    else:
                        st.info("Classification metrics not available in metrics file.")

                    regression_df = perf_df.dropna(subset=["mae"])
                    if not regression_df.empty:
                        st.subheader("Regression metrics")
                        st.dataframe(
                            regression_df[
                                ["target", "model", "mae", "rmse", "r2", "train_size", "test_size"]
                            ]
                        )

                    st.subheader("ROC AUC (macro, one-vs-rest) by target")
                    if "roc_auc_macro_ovr" in perf_df:
                        roc_pivot = classification_df.pivot(
                            index="model",
                            columns="target",
                            values="roc_auc_ovr_macro",
                        )
                        st.bar_chart(roc_pivot)
                    else:
                        st.info("ROC AUC values not available; retrain models to populate.")

                    if {
                        "confusion_matrix",
                        "labels",
                    }.issubset(set(perf_df.columns)):
                        st.subheader("Confusion matrix (test set)")
                        target_to_view = st.selectbox(
                            "Select target for confusion matrix",
                            sorted(perf_df["target"].unique()),
                        )
                        models_for_target = perf_df[
                            perf_df["target"] == target_to_view
                        ]["model"].unique()
                        model_to_view = st.selectbox(
                            "Select model", models_for_target
                        )
                        row = perf_df[
                            (perf_df["target"] == target_to_view)
                            & (perf_df["model"] == model_to_view)
                        ].iloc[0]
                        labels = row.get("labels", [])
                        cm = row.get("confusion_matrix")
                        if labels and cm is not None:
                            try:
                                cm_array = np.array(cm)
                                fig, ax = plt.subplots()
                                im = ax.imshow(cm_array, cmap="Blues")
                                ax.set_xticks(range(len(labels)))
                                ax.set_yticks(range(len(labels)))
                                ax.set_xticklabels(labels)
                                ax.set_yticklabels(labels)
                                ax.set_xlabel("Predicted")
                                ax.set_ylabel("True")
                                plt.colorbar(im, ax=ax)
                                st.pyplot(fig)
                            except Exception as exc:
                                st.info(f"Unable to display confusion matrix: {exc}")
                        if row.get("class_counts"):
                            st.write("Class counts used for training/evaluation:")
                            st.json(row["class_counts"])
                except Exception as exc:
                    st.info(f"Unable to plot performance summary: {exc}")
        else:
            st.info("Model metrics not available. Run training to generate metrics.")

    with st.expander("Model metadata", expanded=False):
        if metadata:
            st.write(f"Version: {metadata.get('version', 'unknown')}")
            st.write(f"Trained at: {metadata.get('trained_at', 'unknown')}")
            st.write(f"Dataset path: {metadata.get('dataset_path', 'unknown')}")
            st.write(
                "Models are trained on an 80% stratified training split and evaluated on a "
                "20% holdout test split to approximate performance on new, unseen data."
            )
            models_meta = metadata.get("models", [])
            if models_meta:
                st.write("Models:")
                meta_df = pd.DataFrame(models_meta)
                st.dataframe(meta_df)
        else:
            st.info("Model metadata not available.")

    display_columns = [
        "date",
        "total_steps",
        "total_minutes_asleep",
        "avg_hr",
        "predicted_health_risk_level",
        "predicted_sleep_quality_score",
        "predicted_sleep_quality_label",
        "predicted_cardiovascular_strain_risk",
        "predicted_stress_risk",
        "stress_quality_label",
        "stress_quality_proba",
    ]

    available_columns = [col for col in display_columns if col in df_pred.columns]

    health_model_name = (
        models[selected_health_model].__class__.__name__
        if selected_health_model in models
        else "Unknown"
    )
    cardio_model_name = models.get("cardio").__class__.__name__ if models.get("cardio") else "Unknown"
    sleep_model_name = (
        models.get("sleep_reg").__class__.__name__ if models.get("sleep_reg") else "Unknown"
    )
    stress_model_name = models.get("stress").__class__.__name__ if models.get("stress") else "Unknown"

    if engine is not None and not df_pred.empty:
        try:
            last_row = df_pred.iloc[-1]
            log_prediction(
                engine,
                selected_user,
                health_model_name,
                cardio_model_name,
                sleep_model_name,
                stress_model_name,
                last_row,
            )
        except Exception as exc:  # pragma: no cover - best-effort logging
            st.info(f"Prediction logging skipped: {exc}")

    with st.expander("Model explainability (feature importance)", expanded=False):
        rf_model = models.get("health")
        if rf_model is not None and hasattr(rf_model, "feature_importances_"):
            importances = pd.DataFrame(
                {
                    "feature": FEATURE_COLUMNS,
                    "importance": rf_model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)
            st.subheader("RandomForest feature importances")
            top_rf = importances.head(10)
            st.dataframe(top_rf)
            st.bar_chart(top_rf.set_index("feature"))
        else:
            st.info("RandomForest model for health risk not available.")

        logreg_model = models.get("health_logreg")
        if logreg_model is not None and hasattr(logreg_model, "coef_"):
            coefs = np.mean(np.abs(logreg_model.coef_), axis=0)
            coef_df = pd.DataFrame(
                {
                    "feature": FEATURE_COLUMNS,
                    "coef_importance": coefs,
                }
            ).sort_values("coef_importance", ascending=False)
            st.subheader("Logistic Regression coefficient magnitudes")
            top_lr = coef_df.head(10)
            st.dataframe(top_lr)
            st.bar_chart(top_lr.set_index("feature"))
        else:
            st.info("Logistic Regression model for health risk not available.")

        st.caption(
            "RandomForest importance is based on impurity decrease; Logistic Regression importance is based on coefficient magnitude."
        )

    # How to test:
    # 1) python -m src.ml.run_inference --source fitbit --user_id demo_user
    # 2) streamlit run src/app/streamlit_app.py
    # 3) Select user_id=demo_user, source=fitbit and confirm 62 rows shown.
    st.markdown('<div class="section-header">📊 Predictions Explorer</div>', unsafe_allow_html=True)
    if prediction_df.empty:
        st.info(
            "No daily predictions found for the selected filters. "
            "Run inference from the sidebar to populate this table."
        )
    else:
        if pred_start and pred_end:
            pred_mask = (prediction_df["date"].dt.date >= pred_start) & (
                prediction_df["date"].dt.date <= pred_end
            )
            prediction_df = prediction_df[pred_mask].copy()

        prediction_df = prediction_df.sort_values("date")
        total_days = len(prediction_df)
        activity_share = pd.to_numeric(
            prediction_df["activity_quality_label"], errors="coerce"
        )
        sleep_share = pd.to_numeric(
            prediction_df["sleep_quality_label"], errors="coerce"
        )
        activity_good_pct = (
            activity_share.eq(1).mean() * 100 if activity_share.notna().any() else 0.0
        )
        sleep_good_pct = (
            sleep_share.eq(1).mean() * 100 if sleep_share.notna().any() else 0.0
        )
        date_range = (
            f"{prediction_df['date'].min().date()} → {prediction_df['date'].max().date()}"
        )

        kpi_cols = st.columns(4)
        kpi_cols[0].metric("Total days predicted", f"{total_days}")
        kpi_cols[1].metric("Good activity days", f"{activity_good_pct:.0f}%")
        kpi_cols[2].metric("Good sleep days", f"{sleep_good_pct:.0f}%")
        kpi_cols[3].metric("Date range", date_range)

        chart_df = prediction_df.set_index("date")
        activity_chart = chart_df[["activity_quality_proba"]].copy()
        sleep_chart = chart_df[["sleep_quality_proba"]].copy()
        if len(chart_df) >= 7:
            activity_chart["activity_quality_proba_7d_avg"] = activity_chart[
                "activity_quality_proba"
            ].rolling(7).mean()
            sleep_chart["sleep_quality_proba_7d_avg"] = sleep_chart[
                "sleep_quality_proba"
            ].rolling(7).mean()

        st.subheader("Activity quality probability")
        if activity_chart.notna().any().any():
            st.line_chart(activity_chart)
        else:
            st.info("Activity probabilities not available to plot.")

        st.subheader("Sleep quality probability")
        if sleep_chart.notna().any().any():
            st.line_chart(sleep_chart)
        else:
            st.info("Sleep probabilities not available to plot.")

        label_counts = pd.DataFrame(
            {
                "activity": prediction_df["activity_quality_label"]
                .value_counts(dropna=True)
                .sort_index(),
                "sleep": prediction_df["sleep_quality_label"]
                .value_counts(dropna=True)
                .sort_index(),
            }
        ).fillna(0)
        st.subheader("Label distribution")
        if not label_counts.empty:
            st.bar_chart(label_counts)
        else:
            st.info("No label distribution data available.")

        prediction_columns = [
            "date",
            "sleep_quality_label",
            "sleep_quality_proba",
            "activity_quality_label",
            "activity_quality_proba",
            "created_at",
        ]
        for col in prediction_columns:
            if col not in prediction_df.columns:
                prediction_df[col] = pd.NA
        prediction_display = prediction_df[prediction_columns].copy()
        st.dataframe(prediction_display.reset_index(drop=True), use_container_width=True)
        csv_bytes = prediction_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download predictions CSV",
            data=csv_bytes,
            file_name="daily_predictions.csv",
            mime="text/csv",
        )

    st.subheader("Daily Metrics and Predicted Risks")
    st.dataframe(df_pred[available_columns].reset_index(drop=True))

    df_pred = df_pred.set_index("date")

    st.subheader("Steps over time")
    st.line_chart(df_pred["total_steps"])

    st.subheader("Sleep duration over time (minutes asleep)")
    st.line_chart(df_pred["total_minutes_asleep"])

    risk_map = {"low": 0, "moderate": 1, "high": 2}
    health_risk_numeric = df_pred.get("predicted_health_risk_level", pd.Series(dtype="float"))
    health_risk_numeric = health_risk_numeric.map(risk_map)

    st.subheader("Predicted overall health risk (0=low,1=moderate,2=high)")
    st.line_chart(health_risk_numeric)

    if "predicted_sleep_quality_score" in df_pred.columns:
        st.subheader("Predicted sleep quality score (0–100)")
        st.line_chart(df_pred["predicted_sleep_quality_score"])

    if "stress_quality_label" in df_pred.columns:
        st.subheader("Predicted stress quality label (1=good, 0=high stress)")
        st.line_chart(df_pred["stress_quality_label"])

    if "stress_quality_proba" in df_pred.columns:
        st.subheader("Predicted stress quality probability")
        st.line_chart(df_pred["stress_quality_proba"])

    with st.expander("Try custom inputs", expanded=False):
        st.write("Manually enter daily metrics to see predicted risks.")
        c1, c2, c3 = st.columns(3)
        total_steps = c1.number_input(
            "Total steps", min_value=0, max_value=50000, value=10000, step=500
        )
        total_distance = c2.number_input(
            "Total distance (km)", min_value=0.0, max_value=50.0, value=7.0, step=0.5
        )
        very_active_minutes = c3.number_input(
            "Very active minutes", min_value=0, max_value=600, value=30, step=5
        )
        fairly_active_minutes = c1.number_input(
            "Fairly active minutes", min_value=0, max_value=600, value=30, step=5
        )
        lightly_active_minutes = c2.number_input(
            "Lightly active minutes", min_value=0, max_value=800, value=200, step=10
        )
        sedentary_minutes = c3.number_input(
            "Sedentary minutes", min_value=0, max_value=1440, value=600, step=10
        )
        calories = c1.number_input(
            "Calories", min_value=0, max_value=6000, value=2200, step=50
        )
        total_minutes_asleep = c2.number_input(
            "Total minutes asleep", min_value=0, max_value=1200, value=420, step=10
        )
        sleep_efficiency = c3.number_input(
            "Sleep efficiency", min_value=0.0, max_value=1.0, value=0.9, step=0.01
        )
        avg_hr = c1.number_input(
            "Average heart rate", min_value=40, max_value=200, value=75, step=1
        )
        max_hr = c2.number_input(
            "Max heart rate", min_value=40, max_value=220, value=150, step=1
        )
        min_hr = c3.number_input(
            "Min heart rate", min_value=30, max_value=200, value=55, step=1
        )
        active_minutes = c1.number_input(
            "Active minutes", min_value=0, max_value=600, value=60, step=5
        )

        if st.button("Predict risks for custom input"):
            custom_df = pd.DataFrame(
                [
                    {
                        "total_steps": total_steps,
                        "total_distance": total_distance,
                        "very_active_minutes": very_active_minutes,
                        "fairly_active_minutes": fairly_active_minutes,
                        "lightly_active_minutes": lightly_active_minutes,
                        "sedentary_minutes": sedentary_minutes,
                        "calories": calories,
                        "total_minutes_asleep": total_minutes_asleep,
                        "sleep_efficiency": sleep_efficiency,
                        "avg_hr": avg_hr,
                        "max_hr": max_hr,
                        "min_hr": min_hr,
                        "active_minutes": active_minutes,
                    }
                ]
            )

            custom_predictions = add_predictions(
                custom_df, models, encoders, health_model_key=selected_health_model
            )
            prediction_cols = [
                "predicted_health_risk_level",
                "predicted_cardiovascular_strain_risk",
                "predicted_sleep_quality_score",
                "predicted_sleep_quality_label",
                "predicted_stress_risk",
            ]
            available_preds = [col for col in prediction_cols if col in custom_predictions.columns]
            if available_preds:
                st.write("Predicted risks:")
                st.dataframe(custom_predictions[available_preds])
            else:
                st.info("Predictions unavailable for the provided input.")


def main() -> None:
    """Entrypoint for running with `streamlit run`."""

    render_dashboard()


if __name__ == "__main__":
    main()
