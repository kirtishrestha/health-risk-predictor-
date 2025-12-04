"""
Train health risk models using labeled daily metrics.

Note: running this script will overwrite previously saved model_*.pkl files
that may only contain feature names.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
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
    model_filename: Optional[str] = None,
    encoder_filename: Optional[str] = None,
    feature_filename: Optional[str] = None,
) -> None:
    """
    Train a classification model for the provided target and persist artifacts.
    """
    prepared = df.dropna(subset=feature_cols + [target_col])
    if prepared.empty:
        print(f"No data available to train target '{target_col}'. Skipping.")
        return

    X = prepared[feature_cols]
    y = prepared[target_col]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(
        " | ".join(
            [
                f"Target: {target_col}",
                f"Model: {model.__class__.__name__}",
                f"Accuracy: {acc:.3f}",
                f"Macro F1: {macro_f1:.3f}",
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

    rf_targets = [
        ("health_risk_level", "health"),
        ("cardiovascular_strain_risk", "cardio"),
        ("sleep_quality_risk", "sleep"),
        ("stress_risk", "stress"),
    ]

    for target_col, model_name in rf_targets:
        train_and_save_model(
            df,
            target_col,
            model_name,
            feature_cols,
            model=RandomForestClassifier(n_estimators=100, random_state=42),
        )

    logreg = LogisticRegression(max_iter=1000, multi_class="auto", n_jobs=-1)
    train_and_save_model(
        df,
        "health_risk_level",
        "health_logreg",
        feature_cols,
        model=logreg,
        model_filename="model_health_logreg.pkl",
        encoder_filename="le_health_logreg.pkl",
        feature_filename="feature_cols_health_logreg.pkl",
    )


if __name__ == "__main__":
    main()
