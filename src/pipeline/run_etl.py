"""Run the end-to-end Fitbit ETL: raw -> canonical -> Supabase."""

from __future__ import annotations

import argparse
import logging
from typing import List, Optional

import pandas as pd

from src.ingestion.read_fitbit_raw import RawFitbitFile, load_fitbit_datasets
from src.ingestion.upload_supabase import get_supabase_client, upsert_dataframe
from src.transform.to_canonical_activity import transform_activity

# Sleep transform import (repo may use different function names)
try:
    from src.transform.to_canonical_sleep import transform_sleep  # type: ignore
except Exception:
    transform_sleep = None  # type: ignore
    try:
        from src.transform.to_canonical_sleep import to_canonical_sleep  # type: ignore
    except Exception:
        to_canonical_sleep = None  # type: ignore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _concat_or_empty(frames: List[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _get_dataset_df(ds: RawFitbitFile) -> pd.DataFrame:
    """
    RawFitbitFile schema can differ by implementation.
    Try common attribute names for the loaded pandas DataFrame.
    """
    for attr in ("df", "data", "dataframe", "frame"):
        if hasattr(ds, attr):
            val = getattr(ds, attr)
            if isinstance(val, pd.DataFrame):
                return val
    raise AttributeError(
        "RawFitbitFile does not expose a pandas DataFrame. "
        "Tried attributes: df, data, dataframe, frame."
    )


def _get_dataset_path_str(ds: RawFitbitFile) -> str:
    # ds.path might be pathlib.Path
    p = getattr(ds, "path", "")
    return str(p)


def _get_dataset_file_type(ds: RawFitbitFile) -> Optional[str]:
    # different loaders may call this field file_type/kind/type
    for attr in ("file_type", "kind", "type"):
        if hasattr(ds, attr):
            v = getattr(ds, attr)
            if v is not None:
                return str(v).lower()
    return None


def run_etl(raw_dir: str, user_id: str, source: str = "fitbit") -> None:
    datasets: List[RawFitbitFile] = load_fitbit_datasets(raw_dir)

    raw_activity_frames: List[pd.DataFrame] = []
    raw_sleep_frames: List[pd.DataFrame] = []

    for ds in datasets:
        path_str = _get_dataset_path_str(ds)
        path_lower = path_str.lower()
        file_type = _get_dataset_file_type(ds)

        # Decide bucket based on explicit type (if available) or filename
        if file_type in {"activity", "daily_activity"} or "dailyactivity" in path_lower:
            raw_activity_frames.append(_get_dataset_df(ds))
        elif file_type in {"sleep", "daily_sleep"} or "sleepday" in path_lower:
            raw_sleep_frames.append(_get_dataset_df(ds))
        else:
            logger.warning("Skipping unrecognized Fitbit file: %s", path_str)

    raw_activity_df = _concat_or_empty(raw_activity_frames)
    raw_sleep_df = _concat_or_empty(raw_sleep_frames)

    # --- Transform to canonical (force canonical user_id = CLI user_id)
    activity_df = pd.DataFrame()
    if not raw_activity_df.empty:
        activity_df = transform_activity(
            raw_activity_df,
            user_id=user_id,
            source=source,
            user_id_override=user_id,  # ✅ override to demo_user
        )

    sleep_df = pd.DataFrame()
    if not raw_sleep_df.empty:
        if transform_sleep is not None:
            sleep_df = transform_sleep(
                raw_sleep_df,
                user_id=user_id,
                source=source,
                user_id_override=user_id,  # ✅ override to demo_user
            )
        elif to_canonical_sleep is not None:
            # older implementation fallback
            sleep_df = to_canonical_sleep(raw_sleep_df)  # type: ignore
            # force override even if old transform doesn't support it
            if "raw_user_id" not in sleep_df.columns:
                sleep_df["raw_user_id"] = sleep_df.get("user_id")
            sleep_df["user_id"] = user_id
            sleep_df["source"] = source
        else:
            raise ImportError("Could not import a sleep transform (transform_sleep or to_canonical_sleep).")

    client = get_supabase_client()

    # --- Upsert canonical tables
    upsert_dataframe(client, "daily_sleep", sleep_df, ["user_id", "date", "source"])
    upsert_dataframe(client, "daily_activity", activity_df, ["user_id", "date", "source"])

    # --- Build daily_features (join day-level sleep + activity on user_id/date/source)
    if not activity_df.empty or not sleep_df.empty:
        if activity_df.empty:
            features = sleep_df.copy()
            for col in ("steps", "distance_km", "active_minutes", "calories"):
                if col not in features.columns:
                    features[col] = pd.NA
            if "sleep_minutes" not in features.columns:
                features["sleep_minutes"] = pd.NA
        elif sleep_df.empty:
            features = activity_df.copy()
            if "sleep_minutes" not in features.columns:
                features["sleep_minutes"] = pd.NA
        else:
            features = pd.merge(
                activity_df,
                sleep_df[["user_id", "raw_user_id", "date", "source", "sleep_minutes"]],
                on=["user_id", "date", "source"],
                how="outer",
            )

        if "raw_user_id" not in features.columns:
            features["raw_user_id"] = pd.NA

        keep_cols = [
            "user_id",
            "raw_user_id",
            "date",
            "source",
            "steps",
            "distance_km",
            "active_minutes",
            "calories",
            "sleep_minutes",
            "created_at",
        ]
        for c in keep_cols:
            if c not in features.columns:
                features[c] = pd.NA

        features_df = features[keep_cols]
        upsert_dataframe(client, "daily_features", features_df, ["user_id", "date", "source"])

    logger.info("ETL complete for user %s", user_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--user_id", required=True)
    parser.add_argument("--source", default="fitbit")
    args = parser.parse_args()

    run_etl(raw_dir=args.raw_dir, user_id=args.user_id, source=args.source)


if __name__ == "__main__":
    main()
