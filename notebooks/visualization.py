"""Basic visualizations for Fitbit-derived datasets."""

from __future__ import annotations

import logging
import matplotlib.pyplot as plt
import pandas as pd

from src.ingestion.upload_supabase import get_supabase_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_features(user_id: str | None, source: str):
    client = get_supabase_client()
    feature_query = client.table("daily_features").select("*").eq("source", source)
    monthly_query = client.table("monthly_metrics").select("*").eq("source", source)
    if user_id:
        feature_query = feature_query.eq("user_id", user_id)
        monthly_query = monthly_query.eq("user_id", user_id)
    features = pd.DataFrame(feature_query.execute().data or [])
    monthly = pd.DataFrame(monthly_query.execute().data or [])
    return features, monthly


def plot_visualizations(user_id: str | None = None, source: str = "fitbit") -> None:
    features, monthly = _load_features(user_id, source)
    if features.empty:
        logger.warning("No daily_features data available for visualization")
        return

    plt.figure(figsize=(8, 4))
    features["sleep_minutes"].dropna().astype(float).plot(kind="hist", bins=20, title="Sleep Minutes Distribution")
    plt.xlabel("Sleep Minutes")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.scatter(features.get("steps", []), features.get("sleep_minutes", []), alpha=0.6)
    plt.xlabel("Steps")
    plt.ylabel("Sleep Minutes")
    plt.title("Steps vs Sleep Minutes")
    plt.tight_layout()
    plt.show()

    if monthly.empty:
        logger.warning("No monthly_metrics data available for line plot")
        return

    monthly_sorted = monthly.sort_values("month")
    plt.figure(figsize=(8, 4))
    plt.plot(pd.to_datetime(monthly_sorted["month"]), monthly_sorted["avg_sleep_minutes"], marker="o")
    plt.title("Monthly Average Sleep Minutes")
    plt.xlabel("Month")
    plt.ylabel("Avg Sleep Minutes")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_visualizations()
