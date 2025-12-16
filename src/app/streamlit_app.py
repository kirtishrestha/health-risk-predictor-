"""Streamlit dashboard for exploring Fitbit daily metrics and risk predictions."""

from __future__ import annotations

import json
import os
import pickle
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

from src.etl.pandas_metrics import (
    build_daily_metrics_pandas,
    find_fitbit_base_dir,
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

    st.caption(
        "Models are trained on labeled historical days (labels come from heuristic rules) "
        "and then used to predict risk levels on new, unseen days."
    )

    uploaded_zip = st.sidebar.file_uploader(
        "Upload your Fitbit export ZIP", type=["zip"], accept_multiple_files=False
    )
    # Uploaded data is extracted into a temporary folder and kept in-memory only.
    using_uploaded = uploaded_zip is not None

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

    st.subheader("Daily Metrics and Predicted Risks")
    st.dataframe(df_pred[available_columns].reset_index(drop=True))

    df_pred = df_pred.set_index("date")

    st.subheader("Steps over time")
    st.line_chart(df_pred["total_steps"])

    st.subheader("Sleep duration over time (minutes asleep)")
    st.line_chart(df_pred["total_minutes_asleep"])

    st.subheader("Average heart rate over time")
    st.line_chart(df_pred["avg_hr"])

    risk_map = {"low": 0, "moderate": 1, "high": 2}
    health_risk_numeric = df_pred.get("predicted_health_risk_level", pd.Series(dtype="float"))
    health_risk_numeric = health_risk_numeric.map(risk_map)

    st.subheader("Predicted overall health risk (0=low,1=moderate,2=high)")
    st.line_chart(health_risk_numeric)

    if "predicted_sleep_quality_score" in df_pred.columns:
        st.subheader("Predicted sleep quality score (0–100)")
        st.line_chart(df_pred["predicted_sleep_quality_score"])

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
