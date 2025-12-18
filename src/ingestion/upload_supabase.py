"""Utilities for upserting canonical datasets into Supabase/PostgreSQL."""

from __future__ import annotations

import logging
import os
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from supabase import Client, create_client

logger = logging.getLogger(__name__)


class SupabaseConfigError(RuntimeError):
    """Raised when Supabase credentials are missing."""


def get_supabase_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise SupabaseConfigError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
    return create_client(url, key)


def _chunk_dataframe(df: pd.DataFrame, chunk_size: int = 500) -> Iterable[pd.DataFrame]:
    for start in range(0, len(df), chunk_size):
        yield df.iloc[start : start + chunk_size]


def _jsonable(v: Any) -> Any:
    """Convert common pandas/numpy/python types to JSON-serializable primitives."""
    # NaN / NaT -> None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass

    # numpy scalar -> python primitive
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        v = v.item()

    # ✅ numeric strings like "327.0" -> 327
    if isinstance(v, str):
        s = v.strip()
        # basic numeric check (handles "123", "123.4", "-10", "-10.0")
        try:
            f = float(s)
            if f.is_integer():
                return int(f)
            return f
        except ValueError:
            return v  # leave non-numeric strings unchanged

    # ✅ integer-like floats -> int (prevents Postgres integer parse issues)
    if isinstance(v, float):
        if v.is_integer():
            return int(v)
        return v

    # timestamps/dates -> ISO string
    if isinstance(v, (pd.Timestamp, datetime, date)):
        return v.isoformat()

    return v


def _sanitize_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{k: _jsonable(v) for k, v in row.items()} for row in records]


def upsert_dataframe(client: Client, table: str, df: pd.DataFrame, conflict_columns: Sequence[str]) -> List[dict]:
    """Upsert a DataFrame into Supabase in manageable chunks."""

    if df is None or df.empty:
        logger.info("No rows to upsert into %s", table)
        return []

    # ✅ Deduplicate by conflict key to avoid "cannot affect row a second time"
    original_len = len(df)
    df = df.drop_duplicates(subset=list(conflict_columns), keep="last")
    removed = original_len - len(df)
    if removed > 0:
        logger.warning(
            "Removed %d duplicate rows for %s based on conflict key %s",
            removed,
            table,
            ",".join(conflict_columns),
        )

    all_results: List[dict] = []
    for chunk in _chunk_dataframe(df):
        payload = _sanitize_records(chunk.to_dict(orient="records"))
        response = client.table(table).upsert(payload, on_conflict=",".join(conflict_columns)).execute()
        all_results.extend(response.data or [])
        logger.info("Upserted %d rows into %s", len(payload), table)

    return all_results
