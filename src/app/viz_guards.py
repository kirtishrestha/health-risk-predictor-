"""Helper utilities for guarding visualizations."""

from __future__ import annotations

from typing import Callable, Iterable

import pandas as pd
import streamlit as st


def require_columns(df: pd.DataFrame, cols: Iterable[str]) -> bool:
    """Return True when dataframe includes all requested columns."""

    if df is None:
        return False
    return all(col in df.columns for col in cols)


def has_rows(df: pd.DataFrame, min_rows: int = 1) -> bool:
    """Return True when dataframe has at least min_rows rows."""

    if df is None:
        return False
    return len(df) >= min_rows


def maybe_render(
    key: str, enabled: bool, render_fn: Callable[[], None], fallback_message: str
) -> bool:
    """Render a chart block conditionally, with optional debug logging."""

    if enabled:
        render_fn()
        return True

    if st.session_state.get("show_debug_info", False):
        st.sidebar.write(f"Skipped {key}: {fallback_message}")
    return False
