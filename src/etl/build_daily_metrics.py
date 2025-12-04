"""ETL pipeline for building daily Fitbit metrics with PySpark."""

import shutil
from pathlib import Path
from typing import Optional

from pyspark.sql import DataFrame, SparkSession, functions as F

RAW_DIR = Path("data/raw/bella_b/Fitabase Data 4.12.16-5.12.16")
OUTPUT_PATH = Path("data/processed/daily_metrics.csv")


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


def build_daily_metrics(
    raw_data_dir: Optional[Path] = None, output_path: Optional[Path] = None
) -> None:
    """Transform raw Fitbit data into aggregated daily metrics."""

    data_dir = raw_data_dir or RAW_DIR
    output_file = output_path or OUTPUT_PATH

    spark = SparkSession.builder.appName("BuildDailyMetrics").getOrCreate()

    daily_activity = _load_daily_activity(spark, data_dir)
    sleep = _load_sleep(spark, data_dir)
    heart_rate = _aggregate_heart_rate(_load_heart_rate(spark, data_dir))

    joined = (
        daily_activity.alias("da")
        .join(sleep.alias("sl"), on=["id", "date"], how="left")
        .join(heart_rate.alias("hr"), on=["id", "date"], how="left")
    )

    result = joined.select(
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
        F.lit("fitbit_bella_b").alias("source"),
    )

    output_dir = output_file.parent
    temp_dir = output_dir / "daily_metrics_tmp"

    output_dir.mkdir(parents=True, exist_ok=True)
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    result.coalesce(1).write.mode("overwrite").option("header", True).csv(
        str(temp_dir)
    )

    part_files = list(temp_dir.glob("part-*.csv"))
    if not part_files:
        raise FileNotFoundError("No CSV output produced by Spark.")

    part_files[0].rename(output_file)
    shutil.rmtree(temp_dir)
    spark.stop()


if __name__ == "__main__":
    build_daily_metrics()
