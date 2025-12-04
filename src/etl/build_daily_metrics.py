"""ETL pipeline for building daily Fitbit metrics with PySpark."""

import logging
import shutil
from pathlib import Path
from typing import Optional

from pyspark.sql import DataFrame, SparkSession, functions as F

RAW_DIR_BELLA_B = Path("data/raw/bella_b/Fitabase Data 4.12.16-5.12.16")
RAW_DIR_BELLA_A = Path("data/raw/bella_a/Fitabase Data 4.12.16-5.12.16")
OUTPUT_PATH = Path("data/processed/daily_metrics.csv")
OUTPUT_PATH_COMBINED = Path("data/processed/daily_metrics_combined.csv")


def _read_csv(spark: SparkSession, path: Path, timestamp: bool = False) -> DataFrame:
    """Read a CSV file with headers and inferred schema."""
    options = {"header": True, "inferSchema": True}
    if timestamp:
        options["timestampFormat"] = "M/d/yyyy h:mm:ss a"
    return spark.read.csv(str(path), **options)


def _load_daily_activity(spark: SparkSession, data_dir: Path) -> DataFrame:
    df = _read_csv(spark, data_dir / "dailyActivity_merged.csv")
    return df.select(
        F.col("Id").alias("id"),
        F.to_date("ActivityDate", "M/d/yyyy").alias("date"),
        F.col("TotalSteps").alias("total_steps"),
        F.col("TotalDistance").alias("total_distance"),
        F.col("VeryActiveMinutes").alias("very_active_minutes"),
        F.col("FairlyActiveMinutes").alias("fairly_active_minutes"),
        F.col("LightlyActiveMinutes").alias("lightly_active_minutes"),
        F.col("SedentaryMinutes").alias("sedentary_minutes"),
        F.col("Calories").alias("calories"),
    )


def _load_sleep(spark: SparkSession, data_dir: Path) -> DataFrame:
    df = _read_csv(spark, data_dir / "sleepDay_merged.csv")
    return df.select(
        F.col("Id").alias("id"),
        F.to_date("SleepDay", "M/d/yyyy h:mm:ss a").alias("date"),
        F.col("TotalSleepRecords").alias("total_sleep_records"),
        F.col("TotalMinutesAsleep").alias("total_minutes_asleep"),
        F.col("TotalTimeInBed").alias("total_time_in_bed"),
    )


def _load_heart_rate(spark: SparkSession, data_dir: Path) -> DataFrame:
    df = _read_csv(spark, data_dir / "heartrate_seconds_merged.csv", timestamp=True)
    df = df.select(
        F.col("Id").alias("id"),
        F.to_timestamp("Time", "M/d/yyyy h:mm:ss a").alias("ts"),
        F.col("Value").alias("heart_rate"),
    )
    return df.withColumn("date", F.to_date("ts"))


def _aggregate_heart_rate(df: DataFrame) -> DataFrame:
    return (
        df.groupBy("id", "date")
        .agg(
            F.mean("heart_rate").alias("avg_hr"),
            F.max("heart_rate").alias("max_hr"),
            F.min("heart_rate").alias("min_hr"),
        )
    )


def build_daily_metrics_for_folder(
    spark: SparkSession, base_path: Path, source_label: str
) -> DataFrame:
    """
    Build daily metrics for a single Fitbit export folder.

    Reads dailyActivity_merged.csv, sleepDay_merged.csv, heartrate_seconds_merged.csv
    from the given base_path (the 'Fitabase Data 4.12.16-5.12.16' folder), applies
    the standard transformations, and returns a Spark DataFrame with a `source`
    column set to source_label.
    """

    daily_activity = _load_daily_activity(spark, base_path)
    sleep = _load_sleep(spark, base_path)
    heart_rate = _aggregate_heart_rate(_load_heart_rate(spark, base_path))

    joined = (
        daily_activity.alias("da")
        .join(sleep.alias("sl"), on=["id", "date"], how="left")
        .join(heart_rate.alias("hr"), on=["id", "date"], how="left")
    )

    return joined.select(
        F.col("id"),
        F.col("date"),
        F.col("total_steps"),
        F.col("total_distance"),
        F.col("very_active_minutes"),
        F.col("fairly_active_minutes"),
        F.col("lightly_active_minutes"),
        F.col("sedentary_minutes"),
        F.col("calories"),
        F.col("total_minutes_asleep"),
        F.col("total_time_in_bed"),
        F.when(
            (F.col("total_time_in_bed") > 0) & F.col("total_minutes_asleep").isNotNull(),
            F.col("total_minutes_asleep") / F.col("total_time_in_bed"),
        ).alias("sleep_efficiency"),
        F.col("avg_hr"),
        F.col("max_hr"),
        F.col("min_hr"),
        (F.col("very_active_minutes") + F.col("fairly_active_minutes")).alias(
            "active_minutes"
        ),
        F.lit(source_label).alias("source"),
    )


def _write_single_csv(df: DataFrame, output_file: Path) -> None:
    """Write a Spark dataframe as a single CSV file with headers."""

    output_dir = output_file.parent
    temp_dir = output_dir / f"{output_file.stem}_tmp"

    output_dir.mkdir(parents=True, exist_ok=True)
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    df.coalesce(1).write.mode("overwrite").option("header", True).csv(str(temp_dir))

    part_files = list(temp_dir.glob("part-*.csv"))
    if not part_files:
        raise FileNotFoundError("No CSV output produced by Spark.")

    part_files[0].rename(output_file)
    shutil.rmtree(temp_dir)


def build_daily_metrics(
    raw_data_dir_bella_b: Optional[Path] = None,
    raw_data_dir_bella_a: Optional[Path] = None,
    output_path: Optional[Path] = None,
    output_path_combined: Optional[Path] = None,
) -> None:
    """Transform raw Fitbit data into aggregated daily metrics across cohorts."""

    data_dir_b = raw_data_dir_bella_b or RAW_DIR_BELLA_B
    data_dir_a = raw_data_dir_bella_a or RAW_DIR_BELLA_A
    output_file_single = output_path or OUTPUT_PATH
    output_file_combined = output_path_combined or OUTPUT_PATH_COMBINED

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    spark = SparkSession.builder.appName("BuildDailyMetrics").getOrCreate()

    df_b = None
    if data_dir_b.exists():
        logging.info("Building daily metrics for bella_b source at %s", data_dir_b)
        df_b = build_daily_metrics_for_folder(spark, data_dir_b, "fitbit_bella_b")
        logging.info("Rows for bella_b: %s", df_b.count())
    else:
        logging.warning("Bella_b data directory not found at %s", data_dir_b)

    df_a = None
    if data_dir_a.exists():
        logging.info("Building daily metrics for bella_a source at %s", data_dir_a)
        df_a = build_daily_metrics_for_folder(spark, data_dir_a, "fitbit_bella_a")
        logging.info("Rows for bella_a: %s", df_a.count())
    else:
        logging.warning("Bella_a data directory not found at %s; skipping", data_dir_a)

    if df_a is not None and df_b is not None:
        df_all = df_a.unionByName(df_b)
    else:
        df_all = df_a or df_b

    if df_all is None:
        logging.error("No data found for any source. Exiting without writing output.")
        spark.stop()
        return

    logging.info("Total rows across sources: %s", df_all.count())

    # For backward compatibility we keep the single-source file primarily using bella_b
    # data when available; otherwise we fall back to all available data.
    if df_b is not None:
        _write_single_csv(df_b, output_file_single)
        logging.info("Wrote bella_b daily metrics to %s", output_file_single)
    else:
        _write_single_csv(df_all, output_file_single)
        logging.info(
            "Wrote fallback daily metrics (non-bella_b) to %s", output_file_single
        )

    _write_single_csv(df_all, output_file_combined)
    logging.info("Wrote combined daily metrics to %s", output_file_combined)

    spark.stop()


if __name__ == "__main__":
    build_daily_metrics()
