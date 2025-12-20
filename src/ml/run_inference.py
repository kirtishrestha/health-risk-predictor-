"""Run daily feature inference and upsert predictions to Supabase.

SQL to create the required table:

create table if not exists daily_predictions (
  user_id text not null,
  date date not null,
  source text not null,
  sleep_quality_label int,
  sleep_quality_proba double precision,
  activity_quality_label int,
  activity_quality_proba double precision,
  created_at timestamptz not null default now(),
  primary key (user_id, date, source)
);
"""

from __future__ import annotations

import argparse
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from src.ingestion.upload_supabase import get_supabase_client

try:
    from src.ingestion.upload_supabase import upsert_dataframe as _upsert_dataframe
except ImportError:  # pragma: no cover - fallback if helper is unavailable
    _upsert_dataframe = None


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SLEEP_MODEL_PATH = Path("models/sleep_quality_model.pkl")
ACTIVITY_MODEL_PATH = Path("models/activity_quality_model.pkl")

REQUIRED_COLUMNS = [
    "user_id",
    "date",
    "source",
    "steps",
    "distance_km",
    "active_minutes",
    "calories",
    "sleep_minutes",
]

SLEEP_COLUMNS = [
    "user_id",
    "date",
    "source",
    "sleep_efficiency",
    "awakenings_count",
]

FEATURE_COLUMNS = [
    "steps",
    "distance_km",
    "active_minutes",
    "calories",
    "sleep_minutes",
]

SLEEP_FEATURE_COLUMNS = [
    "steps",
    "distance_km",
    "active_minutes",
    "calories",
    "sleep_minutes",
    "sleep_efficiency",
    "awakenings_count",
]


def _load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    missing = [col for col in columns if col not in df.columns]
    for col in missing:
        logger.warning("Missing column '%s' in daily_features; filling with NaN", col)
        df[col] = np.nan
    return df


def _fetch_features(source: str, user_id: str | None, all_users: bool) -> pd.DataFrame:
    client = get_supabase_client()
    query = client.table("daily_features").select("*").eq("source", source)
    if not all_users:
        query = query.eq("user_id", user_id)
    data = query.execute().data or []
    df = pd.DataFrame(data)
    df = _ensure_columns(df, REQUIRED_COLUMNS)
    return df[REQUIRED_COLUMNS]


def _fetch_sleep(source: str, user_id: str | None, all_users: bool) -> pd.DataFrame:
    client = get_supabase_client()
    query = (
        client.table("daily_sleep")
        .select("user_id,date,source,sleep_efficiency,awakenings_count")
        .eq("source", source)
    )
    if not all_users:
        query = query.eq("user_id", user_id)
    data = query.execute().data or []
    df = pd.DataFrame(data)
    df = _ensure_columns(df, SLEEP_COLUMNS)
    return df[SLEEP_COLUMNS]


def _prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date", "user_id"]).reset_index(drop=True)
    return df


def _dataset_summary(df: pd.DataFrame) -> None:
    row_count = len(df)
    user_count = df["user_id"].nunique() if not df.empty else 0
    if df.empty:
        date_range = "n/a"
    else:
        min_date = df["date"].min()
        max_date = df["date"].max()
        date_range = f"{min_date} to {max_date}"
    logger.info("Dataset summary: %d rows, %d users, date range %s", row_count, user_count, date_range)


def _predict_with_model(model, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    predictions = model.predict(features)
    proba = None
    if hasattr(model, "predict_proba"):
        proba_values = model.predict_proba(features)
        if proba_values is not None:
            if proba_values.ndim == 2 and proba_values.shape[1] > 1:
                proba = proba_values[:, 1]
            else:
                proba = np.ravel(proba_values)
    return predictions, proba


def _log_label_distribution(name: str, labels: np.ndarray) -> None:
    series = pd.Series(labels)
    distribution = series.value_counts().sort_index().to_dict()
    logger.info("%s label distribution: %s", name, distribution)


def _local_upsert(client, table: str, df: pd.DataFrame, conflict_columns: Iterable[str]) -> list[dict]:
    payload = df.to_dict(orient="records")
    response = (
        client.table(table)
        .upsert(payload, on_conflict=",".join(conflict_columns))
        .execute()
    )
    return response.data or []


def _run_inference(source: str, user_id: str | None, all_users: bool) -> int:
    sleep_model = _load_model(SLEEP_MODEL_PATH)
    activity_model = _load_model(ACTIVITY_MODEL_PATH)

    data = _fetch_features(source=source, user_id=user_id, all_users=all_users)
    data = _prepare_dataset(data)
    sleep_data = _fetch_sleep(source=source, user_id=user_id, all_users=all_users)
    sleep_data = _prepare_dataset(sleep_data)
    _dataset_summary(data)

    if data.empty:
        logger.info("No rows found for inference; exiting.")
        return 0

    total_rows = len(data)
    merged = data.merge(
        sleep_data[["user_id", "date", "source", "sleep_efficiency", "awakenings_count"]],
        on=["user_id", "date", "source"],
        how="left",
    )

    sleep_rows = int(merged["sleep_efficiency"].notna().sum())
    missing_sleep_rows = total_rows - sleep_rows
    logger.info(
        "Sleep coverage: total=%d, has_sleep_fields=%d, missing_sleep_fields=%d",
        total_rows,
        sleep_rows,
        missing_sleep_rows,
    )

    merged = _ensure_columns(merged, SLEEP_FEATURE_COLUMNS)
    sleep_efficiency_median = merged["sleep_efficiency"].median(skipna=True)
    if pd.isna(sleep_efficiency_median):
        sleep_efficiency_median = 0.0
    awakenings_median = merged["awakenings_count"].median(skipna=True)
    if pd.isna(awakenings_median):
        awakenings_median = 0
    merged["sleep_efficiency"] = merged["sleep_efficiency"].fillna(sleep_efficiency_median)
    merged["awakenings_count"] = merged["awakenings_count"].fillna(awakenings_median)

    sleep_features = merged[SLEEP_FEATURE_COLUMNS].fillna(0.0)
    activity_features = merged[FEATURE_COLUMNS].fillna(0.0)
    logger.info("Sleep inference columns: %s", list(sleep_features.columns))

    activity_labels, activity_proba = _predict_with_model(activity_model, activity_features)
    sleep_labels, sleep_proba = _predict_with_model(sleep_model, sleep_features)

    _log_label_distribution("sleep_quality", sleep_labels)
    _log_label_distribution("activity_quality", activity_labels)

    results = merged[["user_id", "date", "source"]].copy()
    results["sleep_quality_label"] = sleep_labels.astype(int)
    results["sleep_quality_proba"] = sleep_proba if sleep_proba is not None else None
    results["activity_quality_label"] = activity_labels.astype(int)
    results["activity_quality_proba"] = activity_proba if activity_proba is not None else None
    results["created_at"] = datetime.now(timezone.utc)

    client = get_supabase_client()
    upsert_func: Callable[..., list[dict]]
    try:
        if _upsert_dataframe is not None:
            upsert_func = _upsert_dataframe
            upserted = upsert_func(client, "daily_predictions", results, ["user_id", "date", "source"])
        else:
            upserted = _local_upsert(client, "daily_predictions", results, ["user_id", "date", "source"])
    except Exception as exc:
        message = (
            "Failed to upsert into daily_predictions. Ensure the table exists by running the SQL in "
            "sql/schema.sql."
        )
        raise RuntimeError(message) from exc

    logger.info("Upserted %d rows into daily_predictions", len(upserted))
    return len(upserted)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference on daily features and upsert predictions.")
    parser.add_argument("--source", default="fitbit", help="Source name to filter daily_features")
    parser.add_argument("--user_id", help="User ID to filter daily_features")
    parser.add_argument("--all_users", action="store_true", help="Run inference for all users")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.all_users and not args.user_id:
        parser.error("--user_id is required unless --all_users is set")

    _run_inference(source=args.source, user_id=args.user_id, all_users=args.all_users)


if __name__ == "__main__":
    main()
