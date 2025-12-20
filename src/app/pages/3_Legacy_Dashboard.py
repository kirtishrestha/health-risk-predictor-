"""Deprecated legacy dashboard based on local CSVs and models."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Legacy (Read-only) Dashboard",
    page_icon="🧪",
    layout="wide",
)

st.title("Legacy (Read-only) Dashboard")
st.caption("Deprecated workflow powered by local CSVs and model artifacts.")
st.info(
    "This legacy dashboard is deprecated and maintained for reference. "
    "Use Pipeline Runner and Analytics Dashboard for the current experience."
)

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


def add_predictions(
    df: pd.DataFrame,
    models: Dict[str, object],
    label_encoders: Dict[str, object],
    *,
    health_model_key: str,
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
            df_pred["predicted_sleep_quality_risk"] = df_pred[
                "predicted_sleep_quality_label"
            ]
        except Exception as exc:  # pragma: no cover - display prediction failure
            st.error(f"Failed to compute sleep quality score: {exc}")
    return df_pred


def normalize_date_range(value) -> tuple[pd.Timestamp, pd.Timestamp]:
    if isinstance(value, tuple) and len(value) == 2:
        return value[0], value[1]
    return value, value

df = load_daily_metrics()
if df.empty:
    st.info("No local data available to display.")
    st.stop()

source_options = ["All"] + sorted(df["source"].dropna().unique())
selected_source = st.selectbox("Select data source", source_options)
if selected_source != "All":
    df = df[df["source"] == selected_source]

if df.empty:
    st.info("No data available for the selected source.")
    st.stop()

user_ids = sorted(df["id"].unique())
selected_user = st.selectbox("Select user ID", user_ids)

min_date = df["date"].min().date()
max_date = df["date"].max().date()
selected_dates = st.date_input(
    "Select date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
)

start_date, end_date = normalize_date_range(selected_dates)

mask_user = df["id"] == selected_user
mask_dates = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
df_filtered = df[mask_user & mask_dates]

if df_filtered.empty:
    st.info("No records found for the selected filters.")
    st.stop()

models = load_models()
encoders = load_label_encoders()
if not models or not encoders:
    st.error("Models or label encoders are unavailable. Please check the models directory.")
    st.stop()

available_health_options = {
    label: key
    for label, key in HEALTH_MODEL_OPTIONS.items()
    if key in models and key in encoders
}
if not available_health_options:
    st.error("No health risk models are available. Please verify training outputs.")
    st.stop()

health_model_label = st.selectbox(
    "Select health risk model",
    options=list(available_health_options.keys()),
)
selected_health_model = available_health_options[health_model_label]

df_pred = add_predictions(
    df_filtered, models, encoders, health_model_key=selected_health_model
)

metadata = load_model_metadata()
with st.expander("Model metadata", expanded=False):
    if metadata:
        st.write(f"Version: {metadata.get('version', 'unknown')}")
        st.write(f"Trained at: {metadata.get('trained_at', 'unknown')}")
        st.write(f"Dataset path: {metadata.get('dataset_path', 'unknown')}")
    else:
        st.info("Model metadata not available.")

st.subheader("Daily Metrics and Predicted Risks")

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
]
available_columns = [col for col in display_columns if col in df_pred.columns]
st.dataframe(df_pred[available_columns].reset_index(drop=True), use_container_width=True)

st.subheader("Steps over time")
if df_pred["total_steps"].notna().any():
    st.line_chart(df_pred.set_index("date")["total_steps"])
else:
    st.info("No step data available for the selected filters.")

st.subheader("Sleep duration over time (minutes asleep)")
if "total_minutes_asleep" in df_pred.columns and df_pred[
    "total_minutes_asleep"
].notna().any():
    st.line_chart(df_pred.set_index("date")["total_minutes_asleep"])
else:
    st.info("No sleep duration data available for the selected filters.")
