"""Helpers for displaying Supabase-backed prediction results."""

from __future__ import annotations

import os
from typing import Tuple

import pandas as pd
import streamlit as st

from src.ingestion.upload_supabase import SupabaseConfigError, get_supabase_client


REQUIRED_ENV_VARS = ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY")


def env_ok() -> Tuple[bool, list[str]]:
    """Return whether required Supabase env vars are available."""

    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
    return len(missing) == 0, missing


@st.cache_data(show_spinner=False)
def load_daily_predictions(user_id: str, source: str) -> pd.DataFrame:
    """Load daily predictions from Supabase for a specific user/source."""

    try:
        client = get_supabase_client()
    except SupabaseConfigError:
        return pd.DataFrame()
    except Exception as exc:  # pragma: no cover - surface query issues
        st.error(f"Unable to initialize Supabase client: {exc}")
        return pd.DataFrame()

    try:
        response = (
            client.table("daily_predictions")
            .select(
                ",".join(
                    [
                        "user_id",
                        "date",
                        "source",
                        "sleep_quality_label",
                        "sleep_quality_proba",
                        "activity_quality_label",
                        "activity_quality_proba",
                        "created_at",
                    ]
                )
            )
            .eq("user_id", user_id)
            .eq("source", source)
            .order("date", desc=False)
            .execute()
        )
        df = pd.DataFrame(response.data or [])
    except Exception as exc:  # pragma: no cover - surface query issues
        st.error(f"Unable to load daily_predictions: {exc}")
        return pd.DataFrame()

    if df.empty:
        return df

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df.sort_values("date")


@st.cache_data(show_spinner=False)
def load_daily_features(user_id: str, source: str) -> pd.DataFrame:
    """Load daily feature metrics from Supabase for a specific user/source."""

    try:
        client = get_supabase_client()
    except SupabaseConfigError:
        return pd.DataFrame()
    except Exception as exc:  # pragma: no cover - surface query issues
        st.error(f"Unable to initialize Supabase client: {exc}")
        return pd.DataFrame()

    try:
        response = (
            client.table("daily_features")
            .select(
                ",".join(
                    [
                        "user_id",
                        "date",
                        "source",
                        "sleep_minutes",
                        "steps",
                    ]
                )
            )
            .eq("user_id", user_id)
            .eq("source", source)
            .order("date", desc=False)
            .execute()
        )
        df = pd.DataFrame(response.data or [])
    except Exception as exc:  # pragma: no cover - surface query issues
        st.error(f"Unable to load daily_features: {exc}")
        return pd.DataFrame()

    if df.empty:
        return df

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df.sort_values("date")


@st.cache_data(show_spinner=False)
def load_prediction_options() -> dict[str, list[str]]:
    """Load distinct user and source options from Supabase predictions."""

    try:
        client = get_supabase_client()
    except SupabaseConfigError:
        return {"user_ids": [], "sources": []}
    except Exception as exc:  # pragma: no cover - surface query issues
        st.error(f"Unable to initialize Supabase client: {exc}")
        return {"user_ids": [], "sources": []}

    try:
        response = (
            client.table("daily_predictions")
            .select("user_id,source")
            .order("user_id", desc=False)
            .execute()
        )
        df = pd.DataFrame(response.data or [])
    except Exception as exc:  # pragma: no cover - surface query issues
        st.error(f"Unable to load prediction options: {exc}")
        return {"user_ids": [], "sources": []}

    if df.empty:
        return {"user_ids": [], "sources": []}

    user_ids = sorted(df["user_id"].dropna().unique().tolist())
    sources = sorted(df["source"].dropna().unique().tolist())
    return {"user_ids": user_ids, "sources": sources}


def clear_prediction_cache() -> None:
    """Clear cached predictions."""

    load_daily_predictions.clear()


def clear_features_cache() -> None:
    """Clear cached daily features."""

    load_daily_features.clear()


def clear_prediction_options_cache() -> None:
    """Clear cached prediction filter options."""

    load_prediction_options.clear()


def compute_kpis(prediction_df: pd.DataFrame) -> dict[str, object]:
    """Compute KPI summary values from predictions."""

    total_days = len(prediction_df)
    activity_share = pd.to_numeric(
        prediction_df.get("activity_quality_label"), errors="coerce"
    )
    sleep_share = pd.to_numeric(
        prediction_df.get("sleep_quality_label"), errors="coerce"
    )
    activity_good_pct = (
        activity_share.eq(1).mean() * 100 if activity_share.notna().any() else 0.0
    )
    sleep_good_pct = (
        sleep_share.eq(1).mean() * 100 if sleep_share.notna().any() else 0.0
    )
    date_range = (
        f"{prediction_df['date'].min().date()} → {prediction_df['date'].max().date()}"
        if total_days
        else ""
    )

    return {
        "total_days": total_days,
        "activity_good_pct": activity_good_pct,
        "sleep_good_pct": sleep_good_pct,
        "date_range": date_range,
    }


def build_probability_trends(
    prediction_df: pd.DataFrame, *, window: int = 7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create dataframes for probability trend charts with rolling averages."""

    chart_df = prediction_df.set_index("date")
    activity_chart = chart_df[["activity_quality_proba"]].copy()
    sleep_chart = chart_df[["sleep_quality_proba"]].copy()

    if len(chart_df) >= window:
        activity_chart[f"activity_quality_proba_{window}d_avg"] = activity_chart[
            "activity_quality_proba"
        ].rolling(window).mean()
        sleep_chart[f"sleep_quality_proba_{window}d_avg"] = sleep_chart[
            "sleep_quality_proba"
        ].rolling(window).mean()

    return activity_chart, sleep_chart


def build_label_distribution(prediction_df: pd.DataFrame) -> pd.DataFrame:
    """Build a label distribution dataframe for charting."""

    return pd.DataFrame(
        {
            "activity": prediction_df.get("activity_quality_label", pd.Series(dtype=int))
            .value_counts(dropna=True)
            .sort_index(),
            "sleep": prediction_df.get("sleep_quality_label", pd.Series(dtype=int))
            .value_counts(dropna=True)
            .sort_index(),
        }
    ).fillna(0)
