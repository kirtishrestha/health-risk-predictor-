"""Utilities for upserting canonical datasets into Supabase/PostgreSQL."""

from __future__ import annotations

import logging
import os
import datetime
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


def _json_safe_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (datetime.date, datetime.datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_safe_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {key: _json_safe_value(value) for key, value in record.items()}


def upsert_dataframe(client: Client, table: str, df: pd.DataFrame, conflict_columns: Sequence[str]) -> List[dict]:
    """Upsert a DataFrame into Supabase in manageable chunks."""

    if df.empty:
        logger.info("No rows to upsert into %s", table)
        return []

    all_results: List[dict] = []
    for chunk in _chunk_dataframe(df):
        payload = [_json_safe_record(record) for record in chunk.to_dict(orient="records")]
        response = client.table(table).upsert(payload, on_conflict=",".join(conflict_columns)).execute()
        all_results.extend(response.data or [])
        logger.info("Upserted %d rows into %s", len(payload), table)
    return all_results
