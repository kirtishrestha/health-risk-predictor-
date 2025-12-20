"""Train and evaluate a binary classifier for stress quality."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ingestion.upload_supabase import get_supabase_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/stress_quality_model.pkl")
METRICS_PATH = Path("reports/stress_quality_metrics.json")

REQUIRED_COLUMNS = [
    "user_id",
    "date",
    "source",
    "steps",
    "distance_km",
    "active_minutes",
    "calories",
    "sleep_minutes",
    "stress_avg",
]

FEATURE_COLUMNS = [
    "steps",
    "distance_km",
    "active_minutes",
    "calories",
    "sleep_minutes",
]


def _load_features(user_id: str | None, source: str, all_users: bool = False) -> pd.DataFrame:
    client = get_supabase_client()
    query = client.table("daily_features").select("*").eq("source", source)
    if not all_users:
        query = query.eq("user_id", user_id)
    return pd.DataFrame(query.execute().data or [])


def _build_pipeline(model_type: str) -> Pipeline:
    if model_type == "logistic_regression":
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
                    FEATURE_COLUMNS,
                )
            ]
        )
        clf = LogisticRegression(max_iter=500, class_weight="balanced")
    elif model_type == "random_forest":
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    SimpleImputer(strategy="median"),
                    FEATURE_COLUMNS,
                )
            ]
        )
        clf = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])


def _safe_roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float | None:
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return roc_auc_score(y_true, y_proba)
    except ValueError:
        return None


def _evaluate(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> Dict[str, float | list[list[int]] | None]:
    roc_auc = _safe_roc_auc(y_true, y_proba)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def _select_best_model(
    metrics_by_model: Dict[str, Dict[str, float | list[list[int]] | None]]
) -> str:
    best_by_roc_auc = [
        (name, metrics["roc_auc"])
        for name, metrics in metrics_by_model.items()
        if metrics["roc_auc"] is not None
    ]
    if best_by_roc_auc:
        return max(best_by_roc_auc, key=lambda item: item[1])[0]
    return max(metrics_by_model.items(), key=lambda item: item[1]["f1"])[0]


def train_model(
    user_id: str | None = None, source: str = "fitbit", all_users: bool = False
) -> None:
    if not all_users and not user_id:
        raise ValueError("Provide --user_id or set --all_users to train on all users.")

    features = _load_features(user_id, source, all_users=all_users)
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in features.columns]
    if missing_cols:
        logger.warning(
            "daily_features is missing required columns (%s). Skipping stress model training.",
            ", ".join(missing_cols),
        )
        METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        skip_metrics = {
            "skipped": True,
            "reason": "Missing required columns",
            "missing_columns": missing_cols,
            "available_columns": list(features.columns),
            "source": source,
            "user_scope": "all_users" if all_users else "single_user",
            "total_rows_loaded": len(features),
        }
        with METRICS_PATH.open("w", encoding="utf-8") as f:
            json.dump(skip_metrics, f, indent=2)
        return

    if features.empty:
        raise ValueError("No data available to train the model.")

    labeled = features.dropna(subset=["stress_avg"]).copy()
    if labeled.empty:
        raise ValueError("No stress_avg values available to label stress quality.")

    labeled["stress_quality_label"] = (labeled["stress_avg"] <= 40).astype(int)

    X = labeled[FEATURE_COLUMNS]
    y = labeled["stress_quality_label"]

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
            y.value_counts().to_dict(),
        )
        METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        skip_metrics = {
            "skipped": True,
            "reason": "Only one class present",
            "class_distribution": class_distribution,
            "total_samples": n_total,
        }
        with METRICS_PATH.open("w", encoding="utf-8") as f:
            json.dump(skip_metrics, f, indent=2)
        return

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42,
        )
    except ValueError as exc:
        logger.warning(
            "Stratified split failed (%s). Falling back to random split without stratification.",
            exc,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

    models = {
        "logistic_regression": _build_pipeline("logistic_regression"),
        "random_forest": _build_pipeline("random_forest"),
    }

    all_metrics: Dict[str, Dict[str, float | list[list[int]] | None]] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        metrics = _evaluate(y_test, y_pred, proba)
        all_metrics[name] = metrics
        logger.info("%s metrics: %s", name, metrics)

    best_model_name = _select_best_model(all_metrics)
    best_model = models[best_model_name]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as f:
        pickle.dump(best_model, f)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "best_model": best_model_name,
        "metrics": all_metrics,
        "total_samples": n_total,
        "class_distribution": class_distribution,
        "unique_classes": unique_classes,
        "user_id": user_id,
        "source": source,
    }
    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    logger.info("Saved best model (%s) to %s", best_model_name, MODEL_PATH)
    logger.info("Metrics written to %s", METRICS_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train stress quality classifier")
    parser.add_argument("--user_id", help="User identifier to filter", required=False)
    parser.add_argument("--source", default="fitbit", help="Data source label")
    parser.add_argument(
        "--all_users",
        action="store_true",
        help="Train model using data from all users",
    )
    args = parser.parse_args()

    train_model(user_id=args.user_id, source=args.source, all_users=args.all_users)


if __name__ == "__main__":
    main()
