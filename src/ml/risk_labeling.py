"""Risk labeling utilities for Fitbit-derived daily metrics.

Because the Fitbit dataset does not include ground-truth clinical labels for
these outcomes, we generate heuristic pseudo-labels based on simple thresholds
and combinations of wearable-derived metrics (steps, sleep, heart rate, etc.).
These labels act as weak supervision for the machine learning models. In a
real-world system, these rules would ideally be replaced or validated against
expert-annotated or clinically validated labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_sleep_quality_score(df: pd.DataFrame) -> pd.Series:
    """Compute a deterministic sleep quality score (0–100).

    The score blends duration and efficiency so it is explainable:
    - Duration peaks for 7–9 hours (420–540 minutes) and declines outside.
    - Efficiency uses provided sleep_efficiency or estimates asleep/bed.
    - Final score = 0.6 * duration_component + 0.4 * efficiency_component.
    """

    df_local = df.copy()

    if "total_minutes_asleep" not in df_local and "total_sleep_minutes" not in df_local:
        return pd.Series(0.0, index=df_local.index)

    minutes_asleep = df_local.get("total_minutes_asleep")
    if minutes_asleep is None:
        minutes_asleep = df_local.get("total_sleep_minutes")
    if minutes_asleep is None:
        minutes_asleep = pd.Series(np.nan, index=df_local.index)
    minutes_asleep = pd.to_numeric(minutes_asleep, errors="coerce")

    time_in_bed = df_local.get("total_time_in_bed")
    if time_in_bed is None:
        time_in_bed = minutes_asleep
    else:
        time_in_bed = pd.to_numeric(time_in_bed, errors="coerce").fillna(minutes_asleep)

    sleep_efficiency = df_local.get("sleep_efficiency")
    if sleep_efficiency is None:
        sleep_efficiency = pd.Series(np.nan, index=df_local.index)
    sleep_efficiency = pd.to_numeric(sleep_efficiency, errors="coerce")

    # If efficiency is missing, estimate from asleep/bed; interpret values <=1 as ratios.
    estimated_efficiency = None
    if time_in_bed is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            estimated_efficiency = (minutes_asleep / time_in_bed) * 100
    efficiency_pct = sleep_efficiency.copy()
    efficiency_pct = efficiency_pct.where(~efficiency_pct.isna(), estimated_efficiency)
    efficiency_pct = efficiency_pct.where(efficiency_pct <= 1, efficiency_pct * 1.0)
    efficiency_pct = efficiency_pct.where(efficiency_pct > 1, efficiency_pct * 100)
    efficiency_pct = efficiency_pct.clip(0, 100)

    # Duration component: full credit in 7–9 hour window, linearly lower outside.
    lower_ideal, upper_ideal, long_sleep_floor = 420, 540, 720
    duration_component = pd.Series(0.0, index=df_local.index)
    duration_component = np.where(
        minutes_asleep.isna(),
        np.nan,
        np.where(
            minutes_asleep < lower_ideal,
            100 * (minutes_asleep / lower_ideal),
            np.where(
                minutes_asleep <= upper_ideal,
                100,
                100 * ((long_sleep_floor - minutes_asleep) / (long_sleep_floor - upper_ideal)),
            ),
        ),
    )
    duration_component = pd.Series(duration_component, index=df_local.index).clip(0, 100)

    # Efficiency component: raw percent with an extra penalty below 85% efficiency.
    efficiency_component = pd.Series(0.0, index=df_local.index)
    efficiency_component = np.where(
        pd.isna(efficiency_pct),
        np.nan,
        np.where(
            efficiency_pct >= 85,
            efficiency_pct,
            efficiency_pct * (efficiency_pct / 85),
        ),
    )
    efficiency_component = pd.Series(efficiency_component, index=df_local.index).clip(0, 100)

    score = 0.6 * duration_component + 0.4 * efficiency_component
    return pd.Series(score, index=df_local.index).clip(0, 100)


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
    """Return df with the 4 new label columns and numeric sleep score added."""

    df = df.copy()

    df["sleep_quality_score"] = compute_sleep_quality_score(df)
    df["sleep_quality_risk"] = df.apply(_sleep_quality_label, axis=1)
    df["cardiovascular_strain_risk"] = df.apply(_cardio_strain_label, axis=1)
    df["stress_risk"] = df.apply(_stress_label, axis=1)
    df["health_risk_level"] = df.apply(_health_risk_label, axis=1)

    return df


__all__ = ["add_risk_labels", "compute_sleep_quality_score"]
