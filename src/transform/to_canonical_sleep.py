"""Transform raw Fitbit sleep exports into the canonical daily schema."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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
    date_col = next((col for col in df.columns if col.lower() in {"sleepday", "date", "sleep_date"}), None)
    if date_col:
        parsed = pd.to_datetime(df[date_col], errors="coerce")
        return pd.Series(parsed).dt.date
    return pd.Series([pd.NaT] * len(df))


def _get_numeric(df: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series([np.nan] * len(df))


def transform_sleep(
    raw_sleep: pd.DataFrame,
    user_id: Optional[str] = None,
    source: str = "fitbit",
    user_id_override: Optional[str] = None,
) -> pd.DataFrame:
    """Normalize raw sleep data to the canonical schema."""

    df = raw_sleep.copy()
    df.columns = [col.strip().replace(" ", "").replace("-", "").replace("_", "").title() for col in df.columns]

    df["user_id"] = _coerce_user_id(df, user_id_override, user_id)
    df["raw_user_id"] = _raw_user_id(df, user_id_override, user_id)
    df["date"] = _parse_date(df)
    df["sleep_minutes"] = _get_numeric(df, ("TotalMinutesAsleep", "Totalminutesasleep")).astype("Int64")
    df["time_in_bed_minutes"] = _get_numeric(df, ("TotalTimeInBed", "Totaltimeinbed")).astype("Int64")

    awakenings = _get_numeric(
        df,
        (
            "NumberOfAwakenings",
            "AwakeningsCount",
            "TotalSleepRecords",
            "Numberofawakenings",
            "Totalsleeprecords",
        ),
    )
    df["awakenings_count"] = awakenings.astype("Int64")

    efficiency_series = _get_numeric(df, ("SleepEfficiency", "Sleepefficiency"))
    if efficiency_series.notna().any():
        df["sleep_efficiency"] = efficiency_series / 100 if efficiency_series.max() > 1 else efficiency_series
    else:
        df["sleep_efficiency"] = df["sleep_minutes"] / df["time_in_bed_minutes"]

    df["source"] = source
    df["created_at"] = datetime.utcnow()

    canonical = df[
        [
            "user_id",
            "raw_user_id",
            "date",
            "source",
            "sleep_minutes",
            "time_in_bed_minutes",
            "awakenings_count",
            "sleep_efficiency",
            "created_at",
        ]
    ].dropna(subset=["date", "sleep_minutes"])

    canonical["sleep_efficiency"] = canonical["sleep_efficiency"].astype(float)
    canonical["sleep_minutes"] = canonical["sleep_minutes"].astype(int)
    canonical["time_in_bed_minutes"] = canonical["time_in_bed_minutes"].astype("Int64")
    canonical["awakenings_count"] = canonical["awakenings_count"].astype("Int64")

    logger.info("Transformed %d sleep records", len(canonical))
    return canonical
