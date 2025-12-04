"""Risk labeling utilities for Fitbit-derived daily metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sleep_quality_label(row: pd.Series) -> str | float:
    minutes_asleep = row.get("total_minutes_asleep")
    efficiency = row.get("sleep_efficiency")

    if pd.isna(minutes_asleep) or pd.isna(efficiency):
        return np.nan

    if 420 <= minutes_asleep <= 540 and efficiency >= 0.85:
        return "low"

    if minutes_asleep < 360 or minutes_asleep > 600 or efficiency < 0.75:
        return "high"

    return "moderate"


def _cardio_strain_label(row: pd.Series) -> str | float:
    avg_hr = row.get("avg_hr")
    active_minutes = row.get("active_minutes")
    max_hr = row.get("max_hr")

    if pd.isna(avg_hr):
        return np.nan

    if (not pd.isna(active_minutes) and avg_hr >= 85 and active_minutes < 20) or (
        not pd.isna(max_hr) and not pd.isna(active_minutes) and max_hr >= 170 and active_minutes >= 60
    ):
        return "high"

    if not pd.isna(active_minutes) and avg_hr < 75 and active_minutes >= 30:
        return "low"

    return "moderate"


def _stress_label(row: pd.Series) -> str | float:
    min_hr = row.get("min_hr")
    minutes_asleep = row.get("total_minutes_asleep")
    sedentary_minutes = row.get("sedentary_minutes")

    if pd.isna(min_hr) and pd.isna(minutes_asleep) and pd.isna(sedentary_minutes):
        return np.nan

    if (
        not pd.isna(min_hr)
        and not pd.isna(minutes_asleep)
        and not pd.isna(sedentary_minutes)
        and min_hr >= 80
        and minutes_asleep < 360
        and sedentary_minutes > 900
    ):
        return "high"

    if not pd.isna(min_hr) and not pd.isna(minutes_asleep) and not pd.isna(
        sedentary_minutes
    ):
        if not (min_hr >= 80 and minutes_asleep < 360 and sedentary_minutes > 900):
            return "low"

    return "moderate"


def _map_label_to_int(label: str | float) -> int | float:
    if pd.isna(label):
        return np.nan
    mapping = {"low": 0, "moderate": 1, "high": 2}
    return mapping[label]


def _health_risk_label(row: pd.Series) -> str | float:
    sleep = row.get("sleep_quality_risk")
    cardio = row.get("cardiovascular_strain_risk")
    stress = row.get("stress_risk")

    if pd.isna(sleep) or pd.isna(cardio) or pd.isna(stress):
        return np.nan

    scores = [_map_label_to_int(sleep), _map_label_to_int(cardio), _map_label_to_int(stress)]
    high_count = sum(score == 2 for score in scores)
    moderate_count = sum(score == 1 for score in scores)

    if high_count >= 2 or (high_count == 1 and moderate_count == 2):
        return "high"

    if all(score == 0 for score in scores):
        return "low"

    return "moderate"


def add_risk_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with the 4 new label columns added."""

    df = df.copy()

    df["sleep_quality_risk"] = df.apply(_sleep_quality_label, axis=1)
    df["cardiovascular_strain_risk"] = df.apply(_cardio_strain_label, axis=1)
    df["stress_risk"] = df.apply(_stress_label, axis=1)
    df["health_risk_level"] = df.apply(_health_risk_label, axis=1)

    return df


__all__ = ["add_risk_labels"]
