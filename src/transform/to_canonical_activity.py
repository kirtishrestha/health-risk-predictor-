"""Transform raw Fitbit activity exports into the canonical daily schema."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

KM_PER_MILE = 1.60934


def _coerce_user_id(df: pd.DataFrame, user_id_override: Optional[str], user_id: Optional[str]) -> pd.Series:
    if user_id_override is not None:
        return pd.Series([user_id_override] * len(df))
    if "Id" in df.columns:
        return df["Id"].astype(str)
    if user_id is None:
        raise ValueError("user_id must be provided when raw data does not include an Id column")
    return pd.Series([user_id] * len(df))


def _raw_user_id(df: pd.DataFrame, user_id_override: Optional[str], user_id: Optional[str]) -> pd.Series:
    if "Id" in df.columns:
        return df["Id"].astype(str)
    fallback = user_id_override if user_id_override is not None else user_id
    if fallback is None:
        return pd.Series([pd.NA] * len(df))
    return pd.Series([fallback] * len(df))


def _parse_date(df: pd.DataFrame) -> pd.Series:
    date_col = next((col for col in df.columns if col.lower() in {"activitydate", "date"}), None)
    parsed = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
    return pd.Series(parsed).dt.date


def _get_numeric(df: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series([np.nan] * len(df))


def transform_activity(
    raw_activity: pd.DataFrame,
    user_id: Optional[str] = None,
    source: str = "fitbit",
    user_id_override: Optional[str] = None,
) -> pd.DataFrame:
    """Normalize raw activity data to the canonical schema."""

    df = raw_activity.copy()
    df.columns = [col.strip().replace(" ", "").replace("-", "").replace("_", "").title() for col in df.columns]

    df["user_id"] = _coerce_user_id(df, user_id_override, user_id)
    df["raw_user_id"] = _raw_user_id(df, user_id_override, user_id)
    df["date"] = _parse_date(df)
    df["steps"] = _get_numeric(df, ("TotalSteps", "Steps", "Totalsteps")).fillna(0).astype(int)

    distance_series = _get_numeric(df, ("TotalDistance", "Distance", "Totaldistance"))
    distance_km = distance_series * KM_PER_MILE
    df["distance_km"] = distance_km.astype(float)

    active_minutes = _get_numeric(
        df,
        (
            "VeryActiveMinutes",
            "ActiveMinutes",
            "TotalActiveMinutes",
            "Veryactiveminutes",
            "ActiveMinutesTotal",
        ),
    )
    # If detailed activity minutes exist, sum multiple columns
    detailed_minutes = [
        col
        for col in ("FairlyActiveMinutes", "LightlyActiveMinutes", "ModeratelyActiveMinutes")
        if col in df.columns
    ]
    if detailed_minutes:
        active_minutes = active_minutes.fillna(0)
        for col in detailed_minutes:
            active_minutes = active_minutes.add(pd.to_numeric(df[col], errors="coerce").fillna(0))

    df["active_minutes"] = active_minutes.astype("Int64")
    df["calories"] = _get_numeric(df, ("Calories", "TotalCalories", "Caloriestotal")).astype(float)

    df["source"] = source
    df["created_at"] = datetime.utcnow()

    canonical = df[
        [
            "user_id",
            "raw_user_id",
            "date",
            "source",
            "steps",
            "distance_km",
            "active_minutes",
            "calories",
            "created_at",
        ]
    ].dropna(subset=["date"])

    logger.info("Transformed %d activity records", len(canonical))
    return canonical
