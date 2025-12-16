"""Pandas helpers for per-user Fitbit uploads (no disk writes)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def find_fitbit_base_dir(extracted_root: Path) -> Optional[Path]:
    """Return the folder that contains Fitbit CSV exports inside an extracted ZIP."""

    for csv_path in extracted_root.rglob("dailyActivity_merged.csv"):
        return csv_path.parent
    return None


def _load_daily_activity(base_path: Path) -> pd.DataFrame:
    path = base_path / "dailyActivity_merged.csv"
    if not path.exists():
        raise FileNotFoundError(f"daily activity file not found at {path}")

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["ActivityDate"], errors="coerce")
    return df.rename(
        columns={
            "Id": "id",
            "TotalSteps": "total_steps",
            "TotalDistance": "total_distance",
            "VeryActiveMinutes": "very_active_minutes",
            "FairlyActiveMinutes": "fairly_active_minutes",
            "LightlyActiveMinutes": "lightly_active_minutes",
            "SedentaryMinutes": "sedentary_minutes",
            "Calories": "calories",
        }
    )[
        [
            "id",
            "date",
            "total_steps",
            "total_distance",
            "very_active_minutes",
            "fairly_active_minutes",
            "lightly_active_minutes",
            "sedentary_minutes",
            "calories",
        ]
    ]


def _load_sleep(base_path: Path) -> Optional[pd.DataFrame]:
    path = base_path / "sleepDay_merged.csv"
    if not path.exists():
        logging.info("Sleep file not found at %s; continuing without sleep data", path)
        return None

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["SleepDay"], errors="coerce")
    df = df.rename(
        columns={
            "Id": "id",
            "TotalSleepRecords": "total_sleep_records",
            "TotalMinutesAsleep": "total_minutes_asleep",
            "TotalTimeInBed": "total_time_in_bed",
        }
    )
    df["sleep_efficiency"] = np.where(
        (df["total_time_in_bed"] > 0) & df["total_minutes_asleep"].notna(),
        df["total_minutes_asleep"] / df["total_time_in_bed"],
        np.nan,
    )
    return df[["id", "date", "total_minutes_asleep", "total_time_in_bed", "sleep_efficiency"]]


def _load_heart_rate(base_path: Path) -> Optional[pd.DataFrame]:
    path = base_path / "heartrate_seconds_merged.csv"
    if not path.exists():
        logging.info("Heart rate file not found at %s; continuing without heart rate", path)
        return None

    df = pd.read_csv(path, parse_dates=["Time"], infer_datetime_format=True)
    df["date"] = df["Time"].dt.normalize()
    grouped = df.groupby(["Id", "date"])  # daily aggregates
    return grouped["Value"].agg(avg_hr="mean", max_hr="max", min_hr="min").reset_index().rename(
        columns={"Id": "id"}
    )


def build_daily_metrics_pandas(base_path: Path, *, source_label: str = "uploaded_fitbit") -> pd.DataFrame:
    """Construct daily metrics using pandas for an already-extracted Fitbit folder."""

    daily_activity = _load_daily_activity(base_path)
    sleep = _load_sleep(base_path)
    heart_rate = _load_heart_rate(base_path)

    merged = daily_activity.copy()
    if heart_rate is not None:
        merged = merged.merge(heart_rate, on=["id", "date"], how="left")
    else:
        merged[["avg_hr", "max_hr", "min_hr"]] = np.nan

    if sleep is not None:
        merged = merged.merge(sleep, on=["id", "date"], how="left")
    else:
        merged[["total_minutes_asleep", "total_time_in_bed", "sleep_efficiency"]] = np.nan

    merged["active_minutes"] = merged[["very_active_minutes", "fairly_active_minutes"]].sum(axis=1)
    merged["source"] = source_label

    return merged[
        [
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
            "source",
        ]
    ]
