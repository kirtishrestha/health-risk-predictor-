"""Placeholder ETL pipeline for building daily Fitbit metrics."""

from pathlib import Path


def build_daily_metrics(raw_data_dir: Path, output_path: Path) -> None:
    """Transform raw Fitbit data into aggregated daily metrics.

    Args:
        raw_data_dir: Directory containing raw Fitbit CSV files.
        output_path: Destination for the processed daily metrics CSV.
    """
    # PySpark-based transformations will be implemented in a future PR.
    del raw_data_dir, output_path
