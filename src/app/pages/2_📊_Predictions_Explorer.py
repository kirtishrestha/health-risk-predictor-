"""Streamlit page for exploring daily predictions."""

from __future__ import annotations

import streamlit as st

from src.app.ui_predictions import (
    build_label_distribution,
    build_probability_trends,
    clear_prediction_cache,
    compute_kpis,
    env_ok,
    load_daily_predictions,
)


st.set_page_config(page_title="Predictions Explorer", page_icon="📊", layout="wide")

st.title("📊 Predictions Explorer")
st.caption("Daily prediction KPIs and charts from Supabase.")

env_ready, missing = env_ok()
if not env_ready:
    st.warning(
        "Supabase credentials are missing, so predictions cannot be loaded. "
        f"Set: {', '.join(missing)}"
    )
    st.stop()

filters = st.columns(3)
with filters[0]:
    prediction_user_id = st.text_input("user_id", value="demo_user")
with filters[1]:
    prediction_source = st.text_input("source", value="fitbit")
with filters[2]:
    if st.button("Refresh predictions"):
        clear_prediction_cache()

prediction_df = load_daily_predictions(prediction_user_id, prediction_source)

if prediction_df.empty:
    st.info(
        "No daily predictions found for the selected filters. "
        "Run inference from the Pipeline Runner to populate this table."
    )
    st.stop()

pred_start = pred_end = None
min_pred_date = prediction_df["date"].min().date()
max_pred_date = prediction_df["date"].max().date()
pred_dates = st.date_input(
    "Prediction date range",
    value=(min_pred_date, max_pred_date),
    min_value=min_pred_date,
    max_value=max_pred_date,
)
if isinstance(pred_dates, tuple) and len(pred_dates) == 2:
    pred_start, pred_end = pred_dates
else:
    pred_start = pred_end = pred_dates

if pred_start and pred_end:
    pred_mask = (prediction_df["date"].dt.date >= pred_start) & (
        prediction_df["date"].dt.date <= pred_end
    )
    prediction_df = prediction_df[pred_mask].copy()

if prediction_df.empty:
    st.info("No predictions found for the selected date range.")
    st.stop()

kpis = compute_kpis(prediction_df)
kpi_cols = st.columns(4)
kpi_cols[0].metric("Total days predicted", f"{kpis['total_days']}")
kpi_cols[1].metric("Good activity days", f"{kpis['activity_good_pct']:.0f}%")
kpi_cols[2].metric("Good sleep days", f"{kpis['sleep_good_pct']:.0f}%")
kpi_cols[3].metric("Date range", kpis["date_range"])

activity_chart, sleep_chart = build_probability_trends(prediction_df)

st.subheader("Activity quality probability")
if activity_chart.notna().any().any():
    st.line_chart(activity_chart)
else:
    st.info("Activity probabilities not available to plot.")

st.subheader("Sleep quality probability")
if sleep_chart.notna().any().any():
    st.line_chart(sleep_chart)
else:
    st.info("Sleep probabilities not available to plot.")

label_counts = build_label_distribution(prediction_df)
st.subheader("Label distribution")
if not label_counts.empty:
    st.bar_chart(label_counts)
else:
    st.info("No label distribution data available.")

prediction_columns = [
    "date",
    "sleep_quality_label",
    "sleep_quality_proba",
    "activity_quality_label",
    "activity_quality_proba",
    "created_at",
]
for col in prediction_columns:
    if col not in prediction_df.columns:
        prediction_df[col] = None
prediction_display = prediction_df[prediction_columns].copy()

st.subheader("Daily predictions")
st.dataframe(prediction_display.reset_index(drop=True), use_container_width=True)

csv_bytes = prediction_display.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download predictions CSV",
    data=csv_bytes,
    file_name="daily_predictions.csv",
    mime="text/csv",
)
