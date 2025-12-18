"""Train and evaluate a binary classifier for sleep quality."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from src.ingestion.upload_supabase import get_supabase_client
from src.ml.label_sleep_quality import label_sleep_quality

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/sleep_quality_model.pkl")
METRICS_PATH = Path("reports/sleep_quality_metrics.json")


def _load_features(
    user_id: str | None, source: str, all_users: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    client = get_supabase_client()
    feature_query = client.table("daily_features").select("*").eq("source", source)
    sleep_query = client.table("daily_sleep").select("user_id,date,source,awakenings_count,sleep_efficiency").eq("source", source)
    if user_id and not all_users:
        feature_query = feature_query.eq("user_id", user_id)
        sleep_query = sleep_query.eq("user_id", user_id)

    features = pd.DataFrame(feature_query.execute().data or [])
    sleep = pd.DataFrame(sleep_query.execute().data or [])
    return features, sleep


def _prepare_dataset(features: pd.DataFrame, sleep: pd.DataFrame) -> pd.DataFrame:
    data = features.copy()
    if not sleep.empty:
        data = data.merge(
            sleep,
            on=["user_id", "date", "source"],
            how="left",
            suffixes=("", "_sleep"),
        )
    if data.empty:
        raise ValueError("No data available to train the model.")

    data["label"] = data.apply(label_sleep_quality, axis=1)
    return data


def _build_pipeline(model_type: str) -> Pipeline:
    numeric_features = ["sleep_minutes", "steps", "distance_km", "active_minutes", "calories", "awakenings_count", "sleep_efficiency"]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        ]
    )

    if model_type == "logistic_regression":
        clf = LogisticRegression(max_iter=500)
    elif model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float | list[list[int]]]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def train_model(user_id: str | None = None, source: str = "fitbit", all_users: bool = False) -> None:
    features, sleep = _load_features(user_id, source, all_users=all_users)
    data = _prepare_dataset(features, sleep)

    feature_cols = ["sleep_minutes", "steps", "distance_km", "active_minutes", "calories", "awakenings_count", "sleep_efficiency"]
    X = data[feature_cols]
    y = data["label"]

    n_total = len(y)
    class_distribution = y.value_counts().to_dict()
    unique_classes = y.nunique()

    logger.info(
        "Dataset summary - total samples: %s, class distribution: %s, unique classes: %s",
        n_total,
        class_distribution,
        unique_classes,
    )

    if unique_classes < 2:
        logger.warning(
            "Training skipped due to single-class labels. Class distribution: %s",
            class_distribution,
        )

        METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        skip_metrics = {
            "skipped": True,
            "reason": "Only one class present",
            "class_distribution": class_distribution,
        }
        with METRICS_PATH.open("w", encoding="utf-8") as f:
            json.dump(skip_metrics, f, indent=2)
        return

    try:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    except ValueError as exc:
        logger.warning(
            "Stratified split failed (%s). Falling back to random split without stratification.",
            exc,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    models = {
        "logistic_regression": _build_pipeline("logistic_regression"),
        "random_forest": _build_pipeline("random_forest"),
    }

    best_model_name = None
    best_model = None
    best_f1 = -1.0
    all_metrics: Dict[str, Dict[str, float | list[list[int]]]] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        metrics = _evaluate(y_test, y_pred, proba)
        all_metrics[name] = metrics
        logger.info("%s metrics: %s", name, metrics)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model_name = name
            best_model = model

    if best_model is None:
        raise RuntimeError("Failed to train any model")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as f:
        pickle.dump(best_model, f)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = {"best_model": best_model_name, "metrics": all_metrics}
    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    logger.info("Saved best model (%s) to %s", best_model_name, MODEL_PATH)
    logger.info("Metrics written to %s", METRICS_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sleep quality classifier")
    parser.add_argument("--user_id", help="User identifier to filter", required=False)
    parser.add_argument("--source", default="fitbit", help="Data source label")
    parser.add_argument(
        "--all_users",
        action="store_true",
        help="Train on all users for the specified source, ignoring user filter",
    )
    args = parser.parse_args()

    train_model(user_id=args.user_id, source=args.source, all_users=args.all_users)


if __name__ == "__main__":
    main()
