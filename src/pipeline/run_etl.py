"""ETL pipeline to normalize Fitbit CSV exports and load them into Supabase."""

from __future__ import annotations

import argparse
import logging
from typing import List

import pandas as pd

from src.ingestion.read_fitbit_raw import RawFitbitFile, load_fitbit_datasets
from src.ingestion.upload_supabase import get_supabase_client, upsert_dataframe
from src.transform.to_canonical_activity import transform_activity
from src.transform.to_canonical_sleep import transform_sleep

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _merge_daily_features(sleep_df: pd.DataFrame, activity_df: pd.DataFrame) -> pd.DataFrame:
    sleep_df = sleep_df.copy()
    activity_df = activity_df.copy()
    if sleep_df is None or sleep_df.empty:
        sleep_df = pd.DataFrame(columns=["user_id", "date", "source", "sleep_minutes"])
    if activity_df is None or activity_df.empty:
        activity_df = pd.DataFrame(columns=["user_id", "date", "source", "steps", "distance_km", "active_minutes", "calories"])

    merged = pd.merge(
        activity_df,
        sleep_df[["user_id", "date", "source", "sleep_minutes"]],
        on=["user_id", "date", "source"],
        how="outer",
    )
    merged["created_at"] = pd.Timestamp.utcnow()
    return merged


def run_etl(raw_dir: str, user_id: str, source: str = "fitbit") -> None:
    datasets: List[RawFitbitFile] = load_fitbit_datasets(raw_dir)
    if not datasets:
        logger.warning("No Fitbit CSV files found in %s", raw_dir)
        return

    sleep_frames = []
    activity_frames = []

    for dataset in datasets:
        if dataset.kind == "sleep":
            sleep_frames.append(transform_sleep(dataset.dataframe, user_id=user_id, source=source))
        elif dataset.kind == "activity":
            activity_frames.append(transform_activity(dataset.dataframe, user_id=user_id, source=source))

    sleep_df = pd.concat(sleep_frames, ignore_index=True) if sleep_frames else pd.DataFrame()
    activity_df = pd.concat(activity_frames, ignore_index=True) if activity_frames else pd.DataFrame()

    client = get_supabase_client()
    upsert_dataframe(client, "daily_sleep", sleep_df, ["user_id", "date", "source"])
    upsert_dataframe(client, "daily_activity", activity_df, ["user_id", "date", "source"])

    features_df = _merge_daily_features(sleep_df, activity_df)
    upsert_dataframe(client, "daily_features", features_df, ["user_id", "date", "source"])
    logger.info("ETL complete for user %s", user_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Fitbit ETL pipeline")
    parser.add_argument("--raw_dir", required=True, help="Directory containing raw Fitbit CSV files")
    parser.add_argument("--user_id", required=True, help="User identifier to associate with records")
    parser.add_argument("--source", default="fitbit", help="Source label for the data")
    args = parser.parse_args()

    run_etl(raw_dir=args.raw_dir, user_id=args.user_id, source=args.source)


if __name__ == "__main__":
    main()
