"""Aggregate daily metrics into monthly metrics and upsert into Supabase."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime

import pandas as pd

from src.ingestion.upload_supabase import get_supabase_client, upsert_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_table(client, table: str, user_id: str | None, source: str) -> pd.DataFrame:
    query = client.table(table).select("*").eq("source", source)
    if user_id:
        query = query.eq("user_id", user_id)
    response = query.execute()
    return pd.DataFrame(response.data or [])


def _aggregate_monthly(user_id: str | None, source: str, sleep_df: pd.DataFrame, activity_df: pd.DataFrame) -> pd.DataFrame:
    def month_start(series: pd.Series) -> pd.Series:
        dates = pd.to_datetime(series)
        return dates.dt.to_period("M").dt.to_timestamp()

    sleep_df = sleep_df.copy()
    activity_df = activity_df.copy()
    if not sleep_df.empty:
        sleep_df["month"] = month_start(sleep_df["date"]).dt.date
    if not activity_df.empty:
        activity_df["month"] = month_start(activity_df["date"]).dt.date

    sleep_grouped = (
        sleep_df.groupby(["user_id", "month", "source"])
        .agg(avg_sleep_minutes=("sleep_minutes", "mean"), sleep_days_count=("sleep_minutes", "count"))
        if not sleep_df.empty
        else pd.DataFrame()
    )

    activity_grouped = (
        activity_df.groupby(["user_id", "month", "source"])
        .agg(
            avg_steps=("steps", "mean"),
            avg_distance_km=("distance_km", "mean"),
            avg_active_minutes=("active_minutes", "mean"),
            total_steps=("steps", "sum"),
            total_distance_km=("distance_km", "sum"),
            total_active_minutes=("active_minutes", "sum"),
            activity_days_count=("steps", "count"),
        )
        if not activity_df.empty
        else pd.DataFrame()
    )

    monthly = pd.merge(
        activity_grouped.reset_index(),
        sleep_grouped.reset_index(),
        on=["user_id", "month", "source"],
        how="outer",
    )

    if user_id:
        monthly["user_id"] = user_id
    monthly["source"] = source
    monthly["created_at"] = datetime.utcnow()

    columns = [
        "user_id",
        "month",
        "source",
        "avg_sleep_minutes",
        "avg_steps",
        "avg_distance_km",
        "avg_active_minutes",
        "total_steps",
        "total_distance_km",
        "total_active_minutes",
        "sleep_days_count",
        "activity_days_count",
        "created_at",
    ]
    return monthly[columns].fillna(0)


def run_monthly_aggregation(user_id: str | None, source: str = "fitbit") -> None:
    client = get_supabase_client()
    sleep_df = _load_table(client, "daily_sleep", user_id, source)
    activity_df = _load_table(client, "daily_activity", user_id, source)

    if sleep_df.empty and activity_df.empty:
        logger.warning("No daily data available for aggregation")
        return

    monthly_df = _aggregate_monthly(user_id, source, sleep_df, activity_df)
    upsert_dataframe(client, "monthly_metrics", monthly_df, ["user_id", "month", "source"])
    logger.info("Monthly aggregation complete for user %s", user_id or "all")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate daily metrics into monthly metrics")
    parser.add_argument("--user_id", help="User identifier to filter by", required=False)
    parser.add_argument("--source", default="fitbit", help="Source label")
    args = parser.parse_args()

    run_monthly_aggregation(user_id=args.user_id, source=args.source)


if __name__ == "__main__":
    main()
