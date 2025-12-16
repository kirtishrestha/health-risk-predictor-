"""
Train health risk models using labeled daily metrics.

Note: running this script will overwrite previously saved model_*.pkl files
that may only contain feature names. The labels are heuristically derived in
``risk_labeling.py`` and treated as ground truth for supervised training.

The workflow mirrors a standard supervised pipeline: heuristic labels are
generated, data is split into an 80% stratified training set and a 20%
stratified test set to simulate unseen days, models are fitted on the training
portion, and evaluation happens only on the held-out test portion.
"""
from __future__ import annotations

import datetime
import math
import json
import pickle
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .risk_labeling import add_risk_labels

DATA_PATH = Path("data/processed/daily_metrics.csv")
DATA_PATH_COMBINED = Path("data/processed/daily_metrics_combined.csv")
MODELS_DIR = Path("models")


def load_data(data_path: Path) -> pd.DataFrame:
    """Load and label the daily metrics data."""
    df = pd.read_csv(data_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = add_risk_labels(df)
    return df


def train_and_save_model(
    df: pd.DataFrame,
    target_col: str,
    model_name: str,
    feature_cols: List[str],
    *,
    model: object,
    dataset_path: Path,
    model_filename: Optional[str] = None,
    encoder_filename: Optional[str] = None,
    feature_filename: Optional[str] = None,
) -> Optional[dict]:
    """
    Train a classification model for the provided target and persist artifacts.
    """
    prepared = df.dropna(subset=feature_cols + [target_col])
    if prepared.empty:
        print(f"No data available to train target '{target_col}'. Skipping.")
        return None

    X = prepared[feature_cols]
    y = prepared[target_col]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Explicitly hold out 20% of labeled rows as unseen test data to simulate
    # evaluating on new days that were not part of model fitting. Stratification
    # preserves the class distribution across splits.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    roc_auc = None
    cm = confusion_matrix(y_test, y_pred)

    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
            roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
        except Exception:
            roc_auc = None
    print(
        " | ".join(
            [
                f"Target: {target_col}",
                f"Model: {model.__class__.__name__}",
                f"Accuracy: {acc:.3f}",
                f"Macro F1: {macro_f1:.3f}",
                f"ROC AUC (macro OVR): {roc_auc:.3f}" if roc_auc is not None else "ROC AUC: n/a",
            ]
        )
    )

    models_dir = MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / (model_filename or f"model_{model_name}.pkl")
    encoder_path = models_dir / (encoder_filename or f"le_{model_name}.pkl")
    feature_path = models_dir / (feature_filename or f"feature_cols_{model_name}.pkl")

    with model_path.open("wb") as f:
        pickle.dump(model, f)
    with encoder_path.open("wb") as f:
        pickle.dump(encoder, f)
    with feature_path.open("wb") as f:
        pickle.dump(feature_cols, f)

    class_counts = y.value_counts().to_dict()

    metrics_record = {
        "target": target_col,
        "model": model.__class__.__name__,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "roc_auc_ovr_macro": float(roc_auc) if roc_auc is not None else None,
        "confusion_matrix": cm.tolist(),
        "labels": encoder.classes_.tolist(),
        "feature_cols": feature_cols,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "dataset_path": str(dataset_path),
        "class_counts": class_counts,
    }

    return metrics_record


def train_and_save_regression(
    df: pd.DataFrame,
    target_col: str,
    model_name: str,
    feature_cols: List[str],
    *,
    model: object,
    dataset_path: Path,
    model_filename: Optional[str] = None,
    feature_filename: Optional[str] = None,
) -> Optional[dict]:
    """Train a regression model for the provided target and persist artifacts."""

    prepared = df.dropna(subset=feature_cols + [target_col])
    if prepared.empty:
        print(f"No data available to train target '{target_col}'. Skipping.")
        return None

    X = prepared[feature_cols]
    y = prepared[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(
        " | ".join(
            [
                f"Target: {target_col}",
                f"Model: {model.__class__.__name__}",
                f"MAE: {mae:.3f}",
                f"RMSE: {rmse:.3f}",
                f"R^2: {r2:.3f}",
            ]
        )
    )

    models_dir = MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / (model_filename or f"model_{model_name}.pkl")
    feature_path = models_dir / (feature_filename or f"feature_cols_{model_name}.pkl")

    with model_path.open("wb") as f:
        pickle.dump(model, f)
    with feature_path.open("wb") as f:
        pickle.dump(feature_cols, f)

    metrics_record = {
        "target": target_col,
        "model": model.__class__.__name__,
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "feature_cols": feature_cols,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "dataset_path": str(dataset_path),
    }

    return metrics_record


def main() -> None:
    feature_cols = [
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

    combined_path = DATA_PATH_COMBINED
    default_path = DATA_PATH

    if combined_path.exists():
        data_path = combined_path
    else:
        print(
            f"Combined dataset not found at {combined_path}, falling back to {default_path}."
        )
        data_path = default_path

    df = load_data(data_path)

    all_metrics = []

    rf_targets = [
        ("health_risk_level", "health"),
        ("cardiovascular_strain_risk", "cardio"),
        ("stress_risk", "stress"),
    ]

    for target_col, model_name in rf_targets:
        metrics = train_and_save_model(
            df,
            target_col,
            model_name,
            feature_cols,
            model=RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            ),
            dataset_path=data_path,
        )
        if metrics:
            all_metrics.append(metrics)

    logreg = LogisticRegression(max_iter=2000, class_weight="balanced")
    logreg_metrics = train_and_save_model(
        df,
        "health_risk_level",
        "health_logreg",
        feature_cols,
        model=logreg,
        dataset_path=data_path,
        model_filename="model_health_logreg.pkl",
        encoder_filename="le_health_logreg.pkl",
        feature_filename="feature_cols_health_logreg.pkl",
    )
    if logreg_metrics:
        all_metrics.append(logreg_metrics)

    sleep_features = feature_cols
    sleep_rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    sleep_metrics = train_and_save_regression(
        df,
        "sleep_quality_score",
        "sleep_reg",
        sleep_features,
        model=sleep_rf,
        dataset_path=data_path,
        model_filename="model_sleep_reg.pkl",
        feature_filename="feature_cols_sleep_reg.pkl",
    )
    if sleep_metrics:
        all_metrics.append(sleep_metrics)

    sleep_linear = LinearRegression()
    sleep_lr_metrics = train_and_save_regression(
        df,
        "sleep_quality_score",
        "sleep_reg_linear",
        sleep_features,
        model=sleep_linear,
        dataset_path=data_path,
        model_filename="model_sleep_linear.pkl",
        feature_filename="feature_cols_sleep_linear.pkl",
    )
    if sleep_lr_metrics:
        all_metrics.append(sleep_lr_metrics)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_payload = {"metrics": all_metrics}
    metrics_path = MODELS_DIR / "model_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    metadata_payload = {
        "version": "1.0.0",
        "trained_at": datetime.datetime.now().isoformat(),
        "dataset_path": str(data_path),
        "train_test_split": {"train_frac": 0.8, "test_frac": 0.2, "stratified": True},
        "models": [
            {
                "target": "health_risk_level",
                "model": "RandomForestClassifier",
                "pickle_path": "models/model_health.pkl",
            },
            {
                "target": "health_risk_level",
                "model": "LogisticRegression",
                "pickle_path": "models/model_health_logreg.pkl",
            },
            {
                "target": "cardiovascular_strain_risk",
                "model": "RandomForestClassifier",
                "pickle_path": "models/model_cardio.pkl",
            },
            {
                "target": "sleep_quality_score",
                "model": "RandomForestRegressor",
                "pickle_path": "models/model_sleep_reg.pkl",
            },
            {
                "target": "sleep_quality_score",
                "model": "LinearRegression",
                "pickle_path": "models/model_sleep_linear.pkl",
            },
            {
                "target": "stress_risk",
                "model": "RandomForestClassifier",
                "pickle_path": "models/model_stress.pkl",
            },
        ],
    }

    metadata_path = MODELS_DIR / "model_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)


if __name__ == "__main__":
    main()
