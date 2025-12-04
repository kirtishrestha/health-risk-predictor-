"""Train health risk models using labeled daily metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from .risk_labeling import add_risk_labels

DATA_PATH = Path("data/processed/daily_metrics.csv")
MODELS_DIR = Path("models")

FEATURE_COLUMNS = [
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

TARGETS = {
    "health_risk_level": "health",
    "cardiovascular_strain_risk": "cardio",
    "sleep_quality_risk": "sleep",
    "stress_risk": "stress",
}


def _prepare_features(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    labeled = df.dropna(subset=[target] + FEATURE_COLUMNS)
    X = labeled[FEATURE_COLUMNS]
    y = labeled[target]
    return X, y


def _train_model(X: pd.DataFrame, y: pd.Series) -> Tuple[RandomForestClassifier, LabelEncoder]:
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"Target: {y.name} | Accuracy: {acc:.3f} | Macro F1: {macro_f1:.3f}")

    return model, encoder


def train_models(training_data_path: Path | None = None, models_dir: Path | None = None) -> None:
    data_path = training_data_path or DATA_PATH
    output_dir = models_dir or MODELS_DIR

    df = pd.read_csv(data_path)
    df = add_risk_labels(df)

    output_dir.mkdir(parents=True, exist_ok=True)

    for target, prefix in TARGETS.items():
        X, y = _prepare_features(df, target)
        if X.empty:
            print(f"No data available to train target '{target}'. Skipping.")
            continue

        model, encoder = _train_model(X, y)

        joblib.dump(model, output_dir / f"model_{prefix}.pkl")
        joblib.dump(encoder, output_dir / f"le_{prefix}.pkl")


if __name__ == "__main__":
    train_models()
