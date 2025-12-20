"""Production analytics dashboard for daily predictions."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from src.app.ui_predictions import (
    clear_features_cache,
    clear_prediction_cache,
    clear_prediction_options_cache,
    env_ok,
    load_daily_features,
    load_daily_predictions,
    load_prediction_options,
)


st.set_page_config(page_title="Analytics Dashboard", page_icon="📊", layout="wide")

st.title("📊 Analytics Dashboard")
st.caption("Personal health analytics from daily predictions and activity insights.")

env_ready, missing = env_ok()
if not env_ready:
    st.warning(
        "Supabase credentials are missing, so analytics cannot be loaded. "
        f"Set: {', '.join(missing)}"
    )
    st.stop()

options = load_prediction_options()
user_options = options["user_ids"] or ["demo_user"]
source_options = options["sources"] or ["fitbit"]

top_filters = st.columns([2, 2, 2, 1])
with top_filters[0]:
    selected_user = st.selectbox("user_id", options=user_options)
with top_filters[1]:
    selected_source = st.selectbox("source", options=source_options)
with top_filters[2]:
    granularity = st.radio(
        "Granularity",
        options=["Daily", "Weekly", "Monthly"],
        horizontal=True,
    )
with top_filters[3]:
    if st.button("Refresh"):
        clear_prediction_cache()
        clear_features_cache()
        clear_prediction_options_cache()

prediction_df = load_daily_predictions(selected_user, selected_source)
features_df = load_daily_features(selected_user, selected_source)

if prediction_df.empty:
    st.info("No predictions available yet for the selected user and source.")
    st.stop()

min_date = prediction_df["date"].min().date()
max_date = prediction_df["date"].max().date()
selected_dates = st.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
else:
    start_date = end_date = selected_dates

mask = (prediction_df["date"].dt.date >= start_date) & (
    prediction_df["date"].dt.date <= end_date
)
prediction_df = prediction_df.loc[mask].copy()
if not features_df.empty:
    feature_mask = (features_df["date"].dt.date >= start_date) & (
        features_df["date"].dt.date <= end_date
    )
    features_df = features_df.loc[feature_mask].copy()

if prediction_df.empty:
    st.info("No predictions found for the selected date range.")
    st.stop()


def _coalesce_probability(label_series: pd.Series, proba_series: pd.Series) -> pd.Series:
    label_numeric = pd.to_numeric(label_series, errors="coerce")
    proba_numeric = pd.to_numeric(proba_series, errors="coerce")
    return proba_numeric.fillna(label_numeric)


sleep_proba = _coalesce_probability(
    prediction_df.get("sleep_quality_label"),
    prediction_df.get("sleep_quality_proba"),
)
activity_proba = _coalesce_probability(
    prediction_df.get("activity_quality_label"),
    prediction_df.get("activity_quality_proba"),
)

sleep_label = pd.to_numeric(prediction_df.get("sleep_quality_label"), errors="coerce")
activity_label = pd.to_numeric(
    prediction_df.get("activity_quality_label"), errors="coerce"
)

sleep_label = sleep_label.fillna((sleep_proba >= 0.5).astype(int))
activity_label = activity_label.fillna((activity_proba >= 0.5).astype(int))
prediction_df["sleep_proba"] = sleep_proba
prediction_df["activity_proba"] = activity_proba
prediction_df["sleep_label"] = sleep_label
prediction_df["activity_label"] = activity_label

health_good_pct = ((sleep_label.eq(1) & activity_label.eq(1)).mean() * 100).round(1)
sleep_good_pct = (sleep_label.eq(1).mean() * 100).round(1)
activity_good_pct = (activity_label.eq(1).mean() * 100).round(1)

overall_score = pd.concat([sleep_proba, activity_proba], axis=1).mean(axis=1)
rolling_std = overall_score.rolling(7, min_periods=1).std()
consistency_score = (1 / (1 + rolling_std.mean())) * 100 if not rolling_std.empty else 0.0

kpi_cols = st.columns(5)
kpi_cols[0].metric("Health score", f"{health_good_pct:.0f}%")
kpi_cols[1].metric("Sleep quality", f"{sleep_good_pct:.0f}%")
kpi_cols[2].metric("Activity quality", f"{activity_good_pct:.0f}%")
kpi_cols[3].metric("Consistency", f"{consistency_score:.0f}%")
kpi_cols[4].metric("Days analyzed", f"{len(prediction_df)}")

trend_df = prediction_df[["date"]].copy()
trend_df["sleep_proba"] = prediction_df["sleep_proba"]
trend_df["activity_proba"] = prediction_df["activity_proba"]
trend_df["health_risk"] = 1 - overall_score
trend_df["sleep_rolling"] = trend_df["sleep_proba"].rolling(7, min_periods=1).mean()
trend_df["activity_rolling"] = trend_df["activity_proba"].rolling(7, min_periods=1).mean()
trend_df["health_rolling"] = trend_df["health_risk"].rolling(7, min_periods=1).mean()

if granularity != "Daily":
    freq = "W-MON" if granularity == "Weekly" else "MS"
    trend_df = (
        trend_df.set_index("date")
        .resample(freq)
        .mean(numeric_only=True)
        .reset_index()
    )

trend_cols = st.columns(3)
with trend_cols[0]:
    with st.container(border=True):
        st.subheader("Sleep quality probability")
        st.caption("How likely was high sleep quality?")
        sleep_chart = (
            alt.Chart(trend_df)
            .transform_fold(
                ["sleep_proba", "sleep_rolling"],
                as_=["series", "value"],
            )
            .mark_line()
            .encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("value:Q", title="Probability", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(
                        domain=["sleep_proba", "sleep_rolling"],
                        range=["#4C78A8", "#72B7B2"],
                    ),
                    legend=alt.Legend(title=None),
                ),
            )
            .properties(height=220)
        )
        st.altair_chart(sleep_chart, use_container_width=True)

with trend_cols[1]:
    with st.container(border=True):
        st.subheader("Activity quality probability")
        st.caption("How likely was strong daily activity?")
        activity_chart = (
            alt.Chart(trend_df)
            .transform_fold(
                ["activity_proba", "activity_rolling"],
                as_=["series", "value"],
            )
            .mark_line()
            .encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("value:Q", title="Probability", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(
                        domain=["activity_proba", "activity_rolling"],
                        range=["#F58518", "#E45756"],
                    ),
                    legend=alt.Legend(title=None),
                ),
            )
            .properties(height=220)
        )
        st.altair_chart(activity_chart, use_container_width=True)

with trend_cols[2]:
    with st.container(border=True):
        st.subheader("Overall health risk")
        st.caption("Higher means elevated risk signals.")
        health_chart = (
            alt.Chart(trend_df)
            .transform_fold(
                ["health_risk", "health_rolling"],
                as_=["series", "value"],
            )
            .mark_line()
            .encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("value:Q", title="Risk", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(
                        domain=["health_risk", "health_rolling"],
                        range=["#54A24B", "#9D755D"],
                    ),
                    legend=alt.Legend(title=None),
                ),
            )
            .properties(height=220)
        )
        st.altair_chart(health_chart, use_container_width=True)

st.subheader("Distribution snapshots")

label_counts = (
    pd.DataFrame(
        {
            "sleep": prediction_df["sleep_label"].value_counts().sort_index(),
            "activity": prediction_df["activity_label"].value_counts().sort_index(),
        }
    )
    .fillna(0)
    .reset_index()
    .rename(columns={"index": "label"})
)

count_cols = st.columns(3)
with count_cols[0]:
    with st.container(border=True):
        st.caption("Sleep label counts")
        sleep_bar = (
            alt.Chart(label_counts)
            .mark_bar(color="#4C78A8")
            .encode(
                x=alt.X("label:O", title="Label"),
                y=alt.Y("sleep:Q", title="Days"),
            )
            .properties(height=200)
        )
        st.altair_chart(sleep_bar, use_container_width=True)

with count_cols[1]:
    with st.container(border=True):
        st.caption("Activity label counts")
        activity_bar = (
            alt.Chart(label_counts)
            .mark_bar(color="#F58518")
            .encode(
                x=alt.X("label:O", title="Label"),
                y=alt.Y("activity:Q", title="Days"),
            )
            .properties(height=200)
        )
        st.altair_chart(activity_bar, use_container_width=True)

with count_cols[2]:
    with st.container(border=True):
        st.caption("Monthly good days (sleep vs activity)")
        monthly_df = prediction_df[["date"]].copy()
        monthly_df["month"] = monthly_df["date"].dt.to_period("M").dt.to_timestamp()
        monthly_df["sleep_good"] = prediction_df["sleep_label"].eq(1).astype(int)
        monthly_df["activity_good"] = prediction_df["activity_label"].eq(1).astype(int)
        monthly_summary = (
            monthly_df.groupby("month", as_index=False)
            .agg({"sleep_good": "sum", "activity_good": "sum"})
            .melt("month", var_name="metric", value_name="days")
        )
        monthly_chart = (
            alt.Chart(monthly_summary)
            .mark_bar()
            .encode(
                x=alt.X("month:T", title=None),
                y=alt.Y("days:Q", title="Good days"),
                color=alt.Color(
                    "metric:N",
                    scale=alt.Scale(range=["#72B7B2", "#E45756"]),
                    legend=alt.Legend(title=None),
                ),
            )
            .properties(height=200)
        )
        st.altair_chart(monthly_chart, use_container_width=True)

st.subheader("Behavioral insights")

insight_cols = st.columns(3)
with insight_cols[0]:
    with st.container(border=True):
        st.caption("Sleep quality by day of week")
        heatmap_df = prediction_df[["date"]].copy()
        heatmap_df["day_of_week"] = prediction_df["date"].dt.day_name()
        heatmap_df["sleep_label"] = prediction_df["sleep_label"]
        heatmap_df["sleep_proba"] = prediction_df["sleep_proba"]
        day_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        heatmap_summary = (
            heatmap_df.dropna(subset=["sleep_label", "sleep_proba"])
            .groupby(["day_of_week", "sleep_label"], as_index=False)
            .agg(avg_proba=("sleep_proba", "mean"))
        )
        heatmap = (
            alt.Chart(heatmap_summary)
            .mark_rect()
            .encode(
                x=alt.X("day_of_week:N", sort=day_order, title=None),
                y=alt.Y("sleep_label:O", title="Label"),
                color=alt.Color("avg_proba:Q", title="Avg. prob"),
            )
            .properties(height=220)
        )
        st.altair_chart(heatmap, use_container_width=True)

with insight_cols[1]:
    with st.container(border=True):
        st.caption("Steps vs activity probability")
        scatter_df = prediction_df.merge(
            features_df,
            on=["user_id", "source", "date"],
            how="left",
        )
        scatter_df["steps"] = pd.to_numeric(scatter_df.get("steps"), errors="coerce")
        scatter_df = scatter_df.dropna(subset=["steps", "activity_proba"])
        if scatter_df.empty:
            st.info("No step data available for this selection.")
        else:
            scatter = (
                alt.Chart(scatter_df)
                .mark_circle(size=70, color="#F58518")
                .encode(
                    x=alt.X("steps:Q", title="Steps"),
                    y=alt.Y(
                        "activity_proba:Q",
                        title="Activity probability",
                        scale=alt.Scale(domain=[0, 1]),
                    ),
                    tooltip=["date:T", "steps:Q", "activity_proba:Q"],
                )
                .properties(height=220)
            )
            st.altair_chart(scatter, use_container_width=True)

with insight_cols[2]:
    with st.container(border=True):
        st.caption("Sleep minutes vs sleep label")
        box_df = prediction_df.merge(
            features_df,
            on=["user_id", "source", "date"],
            how="left",
        )
        box_df["sleep_minutes"] = pd.to_numeric(
            box_df.get("sleep_minutes"), errors="coerce"
        )
        box_df = box_df.dropna(subset=["sleep_minutes", "sleep_label"])
        if box_df.empty:
            st.info("No sleep minute data available for this selection.")
        else:
            boxplot = (
                alt.Chart(box_df)
                .mark_boxplot(size=30, color="#4C78A8")
                .encode(
                    x=alt.X("sleep_label:O", title="Sleep label"),
                    y=alt.Y("sleep_minutes:Q", title="Sleep minutes"),
                )
                .properties(height=220)
            )
            st.altair_chart(boxplot, use_container_width=True)

with st.expander("Prediction details", expanded=False):
    display_columns = [
        "date",
        "sleep_quality_label",
        "sleep_quality_proba",
        "activity_quality_label",
        "activity_quality_proba",
        "created_at",
    ]
    display_df = prediction_df.copy()
    for col in display_columns:
        if col not in display_df.columns:
            display_df[col] = None
    display_df = display_df[display_columns].reset_index(drop=True)
    st.dataframe(display_df, use_container_width=True)

    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="daily_predictions.csv",
        mime="text/csv",
    )
