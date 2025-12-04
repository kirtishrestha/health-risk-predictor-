"""Load raw Fitbit CSVs into Supabase/Postgres tables."""

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

from src.config import Settings

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure basic logging for the ingestion script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def load_daily_activity(base_path: Path) -> pd.DataFrame:
    """Load and normalize the daily activity CSV."""
    path = base_path / "dailyActivity_merged.csv"
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "Id": "id",
            "ActivityDate": "activity_date",
            "TotalSteps": "total_steps",
            "TotalDistance": "total_distance",
            "VeryActiveMinutes": "very_active_minutes",
            "FairlyActiveMinutes": "fairly_active_minutes",
            "LightlyActiveMinutes": "lightly_active_minutes",
            "SedentaryMinutes": "sedentary_minutes",
            "Calories": "calories",
        }
    )
    df["activity_date"] = pd.to_datetime(df["activity_date"], errors="raise").dt.date
    df["source"] = "fitbit_bella_b"
    LOGGER.info("Loaded dailyActivity_merged: %d rows", len(df))
    return df


def load_sleep_day(base_path: Path) -> pd.DataFrame:
    """Load and normalize the sleep day CSV."""
    path = base_path / "sleepDay_merged.csv"
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "Id": "id",
            "SleepDay": "sleep_date",
            "TotalSleepRecords": "total_sleep_records",
            "TotalMinutesAsleep": "total_minutes_asleep",
            "TotalTimeInBed": "total_time_in_bed",
        }
    )
    df["sleep_date"] = pd.to_datetime(df["sleep_date"], errors="raise").dt.date
    df["source"] = "fitbit_bella_b"
    LOGGER.info("Loaded sleepDay_merged: %d rows", len(df))
    return df


def load_heart_rate_seconds(base_path: Path) -> pd.DataFrame:
    """Load and normalize the heart rate per second CSV."""
    path = base_path / "heartrate_seconds_merged.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={"Id": "id", "Time": "ts", "Value": "heart_rate"})
    df["ts"] = pd.to_datetime(df["ts"], errors="raise")
    df["source"] = "fitbit_bella_b"
    LOGGER.info("Loaded heartrate_seconds_merged: %d rows", len(df))
    return df


def load_dataframe(df: pd.DataFrame, table_name: str, engine, schema: str) -> None:
    """Persist a dataframe into the target database table."""
    df.to_sql(table_name, engine, schema=schema, if_exists="append", index=False)
    LOGGER.info("Inserted %d rows into %s.%s", len(df), schema, table_name)


def main() -> None:
    """Entry point for loading raw Fitbit data into Supabase."""
    configure_logging()
    settings = Settings.load()

    if not settings.supabase_db_url:
        raise ValueError("SUPABASE_DB_URL environment variable is required to load data.")

    engine = create_engine(settings.supabase_db_url)
    base_path = Path("data/raw/bella_b/Fitabase Data 4.12.16-5.12.16/")

    daily_activity_df = load_daily_activity(base_path)
    sleep_day_df = load_sleep_day(base_path)
    heart_rate_df = load_heart_rate_seconds(base_path)

    load_dataframe(daily_activity_df, "raw_fitbit_daily_activity", engine, settings.supabase_schema)
    load_dataframe(sleep_day_df, "raw_fitbit_sleep_day", engine, settings.supabase_schema)
    load_dataframe(heart_rate_df, "raw_fitbit_hr_seconds", engine, settings.supabase_schema)


if __name__ == "__main__":
    main()
