"""Utilities for discovering and reading raw Fitbit CSV exports."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


SLEEP_COLUMNS = {"sleepday", "totalminutesasleep", "totaltimeinbed", "totalsleeprecords", "numberofawakenings"}
ACTIVITY_COLUMNS = {"activitydate", "totalsteps", "totaldistance", "veryactiveminutes", "calories"}


@dataclass
class RawFitbitFile:
    """Container describing a discovered Fitbit CSV file."""

    path: Path
    kind: str
    dataframe: pd.DataFrame


def _normalize_columns(columns: Iterable[str]) -> List[str]:
    return [col.strip().lower().replace(" ", "") for col in columns]


def _detect_kind(df: pd.DataFrame) -> Optional[str]:
    normalized = set(_normalize_columns(df.columns))
    if normalized & SLEEP_COLUMNS:
        if {"sleepday", "totalminutesasleep"}.issubset(normalized):
            return "sleep"
    if normalized & ACTIVITY_COLUMNS:
        if {"activitydate", "totalsteps"}.issubset(normalized):
            return "activity"
    return None


def load_fitbit_datasets(raw_dir: str) -> List[RawFitbitFile]:
    """Discover and read Fitbit CSV files from a directory.

    Args:
        raw_dir: Directory containing Fitbit CSV exports.

    Returns:
        A list of RawFitbitFile entries with detected type and DataFrames.
    """

    base_path = Path(raw_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")

    datasets: List[RawFitbitFile] = []
    for path in sorted(base_path.glob("*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - logging only
            logger.error("Failed to read %s: %s", path, exc)
            continue

        kind = _detect_kind(df)
        if kind is None:
            logger.warning("Skipping unrecognized Fitbit file: %s", path)
            continue

        datasets.append(RawFitbitFile(path=path, kind=kind, dataframe=df))
        logger.info("Loaded %s file: %s with %d rows", kind, path.name, len(df))

    return datasets
