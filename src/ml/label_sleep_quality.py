"""Rule-based sleep quality labeling utilities."""

from __future__ import annotations

import pandas as pd


def label_sleep_quality(row: pd.Series) -> int:
    """Assign a binary sleep quality label based on heuristic rules."""

    score = 0
    sleep_minutes = row.get("sleep_minutes")
    if sleep_minutes is not None and 420 <= float(sleep_minutes) <= 540:
        score += 1

    sleep_efficiency = row.get("sleep_efficiency")
    if pd.notna(sleep_efficiency) and float(sleep_efficiency) >= 0.85:
        score += 1

    awakenings = row.get("awakenings_count")
    if pd.notna(awakenings) and float(awakenings) <= 2:
        score += 1

    return 1 if score >= 2 else 0
