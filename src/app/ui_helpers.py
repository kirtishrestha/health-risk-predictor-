"""Shared UI helper utilities for Streamlit pages."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def safe_has_cols(df: pd.DataFrame, cols: Iterable[str]) -> bool:
    """Return True when dataframe has all requested columns and is not empty."""

    if df is None or df.empty:
        return False
    return all(col in df.columns for col in cols)


def add_rolling(series: pd.Series, window: int = 7) -> pd.Series:
    """Add a rolling average to a numeric series."""

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return numeric
    return numeric.rolling(window=window, min_periods=1).mean()


def compute_risk_bucket(
    df: pd.DataFrame,
    *,
    low_threshold: float = 0.35,
    high_threshold: float = 0.65,
    score_columns: Iterable[str] | None = None,
    existing_column: str | None = None,
    invert_scores: bool = False,
) -> pd.DataFrame:
    """Return a dataframe with risk score and bucket columns for display."""

    df_copy = df.copy()
    if existing_column and existing_column in df_copy.columns:
        df_copy["risk_bucket"] = df_copy[existing_column].fillna("Unknown").astype(str)
        df_copy["risk_score"] = np.nan
        return df_copy

    score_columns = list(score_columns or [])
    available = [col for col in score_columns if col in df_copy.columns]
    if not available:
        df_copy["risk_score"] = np.nan
        df_copy["risk_bucket"] = "Unknown"
        return df_copy

    score = pd.concat(
        [pd.to_numeric(df_copy[col], errors="coerce") for col in available], axis=1
    ).mean(axis=1, skipna=True)
    if invert_scores:
        score = 1 - score
    df_copy["risk_score"] = score

    low_condition = score <= low_threshold
    high_condition = score >= high_threshold
    df_copy["risk_bucket"] = np.select(
        [low_condition, high_condition], ["Low", "High"], default="Moderate"
    )
    df_copy.loc[score.isna(), "risk_bucket"] = "Unknown"
    return df_copy
