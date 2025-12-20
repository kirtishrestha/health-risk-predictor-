"""Production analytics dashboard for daily predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.app.ui_helpers import add_rolling, compute_risk_bucket, safe_has_cols
from src.app.ui_predictions import (
    clear_features_cache,
    clear_monthly_metrics_cache,
    clear_prediction_cache,
    clear_prediction_options_cache,
    env_ok,
    load_daily_features,
    load_daily_predictions,
    load_monthly_metrics,
    load_prediction_options,
)


st.set_page_config(page_title="Analytics Dashboard", page_icon="📊", layout="wide")

st.title("Analytics Dashboard")
st.caption("Personal health analytics from Supabase-powered daily insights.")

env_ready, _ = env_ok()
if not env_ready:
    st.info(
        "Analytics data isn't available yet. Configure Supabase credentials to view this dashboard."
    )
    st.stop()

options = load_prediction_options()
if not options["user_ids"] or not options["sources"]:
    st.info("No analytics data found yet. Run the pipeline to populate predictions.")
    st.stop()

user_options = options["user_ids"]
source_options = options["sources"]


st.sidebar.header("Risk bucket thresholds")
low_threshold = st.sidebar.slider(
    "Low risk minimum score",
    min_value=0.6,
    max_value=1.0,
    value=0.75,
    step=0.05,
)
high_threshold = st.sidebar.slider(
    "High risk maximum score",
    min_value=0.0,
    max_value=0.6,
    value=0.45,
    step=0.05,
)
if high_threshold >= low_threshold:
    st.sidebar.warning("High risk threshold should be lower than low risk threshold.")


def normalize_date_range(value) -> tuple[pd.Timestamp, pd.Timestamp]:
    if isinstance(value, tuple) and len(value) == 2:
        return value[0], value[1]
    return value, value


st.header("Filters")
st.subheader("Refine your dashboard view.")
filter_cols = st.columns([2, 2, 3, 2.2, 1])
with filter_cols[0]:
    selected_user = st.selectbox("User", options=user_options)
with filter_cols[1]:
    selected_source = st.selectbox("Source", options=source_options)

prediction_df = load_daily_predictions(selected_user, selected_source)
if prediction_df.empty:
    st.info("No predictions available yet for the selected user and source.")
    st.stop()

prediction_df = prediction_df.copy()
prediction_df["date"] = pd.to_datetime(prediction_df["date"], errors="coerce")
prediction_df = prediction_df.dropna(subset=["date"])
prediction_df["date_only"] = prediction_df["date"].dt.date
if prediction_df.empty:
    st.info("No valid prediction dates available yet for the selected user and source.")
    st.stop()

min_date = prediction_df["date_only"].min()
max_date = prediction_df["date_only"].max()

with filter_cols[2]:
    selected_dates = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
with filter_cols[3]:
    granularity = st.radio(
        "Granularity",
        options=["Daily", "Weekly", "Monthly"],
        horizontal=True,
    )
with filter_cols[4]:
    if st.button("Refresh", use_container_width=True):
        clear_prediction_cache()
        clear_features_cache()
        clear_prediction_options_cache()
        clear_monthly_metrics_cache()
        st.rerun()

st.markdown("---")

start_date, end_date = normalize_date_range(selected_dates)

mask = (prediction_df["date_only"] >= start_date) & (
    prediction_df["date_only"] <= end_date
)
prediction_df = prediction_df.loc[mask].copy()

features_df = load_daily_features(selected_user, selected_source)
if not features_df.empty:
    features_df = features_df.copy()
    features_df["date"] = pd.to_datetime(features_df["date"], errors="coerce")
    features_df = features_df.dropna(subset=["date"])
    features_df["date_only"] = features_df["date"].dt.date
    feature_mask = (features_df["date_only"] >= start_date) & (
        features_df["date_only"] <= end_date
    )
    features_df = features_df.loc[feature_mask].copy()

monthly_df = load_monthly_metrics(selected_user, selected_source)
if not monthly_df.empty:
    monthly_df = monthly_df.copy()
    monthly_df["month"] = pd.to_datetime(monthly_df["month"], errors="coerce")
    monthly_df = monthly_df.dropna(subset=["month"])
    monthly_df["month_only"] = monthly_df["month"].dt.date
    monthly_mask = (monthly_df["month_only"] >= start_date) & (
        monthly_df["month_only"] <= end_date
    )
    monthly_df = monthly_df.loc[monthly_mask].copy()

if prediction_df.empty:
    st.warning("No predictions found for the selected date range.")
    st.stop()


sleep_proba = pd.to_numeric(
    prediction_df.get("sleep_quality_proba"), errors="coerce"
).fillna(pd.to_numeric(prediction_df.get("sleep_quality_label"), errors="coerce"))
activity_proba = pd.to_numeric(
    prediction_df.get("activity_quality_proba"), errors="coerce"
).fillna(pd.to_numeric(prediction_df.get("activity_quality_label"), errors="coerce"))

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
prediction_df["weekday"] = prediction_df["date"].dt.day_name()
prediction_df["month"] = prediction_df["date"].dt.month_name()

risk_columns = [
    "predicted_health_risk_level",
    "predicted_cardiovascular_strain_risk",
    "predicted_sleep_quality_label",
    "health_risk_level",
    "risk_level",
]
risk_source_column = next(
    (col for col in risk_columns if col in prediction_df.columns), None
)

prediction_df = compute_risk_bucket(
    prediction_df,
    low_threshold=low_threshold,
    high_threshold=high_threshold,
    score_columns=["sleep_proba", "activity_proba"],
    existing_column=risk_source_column,
)


st.header("Predictions at a glance")
st.subheader("Model outputs front-and-center for the selected range.")

avg_sleep = sleep_proba.mean()
avg_activity = activity_proba.mean()
sleep_good_pct = sleep_label.eq(1).mean()
activity_good_pct = activity_label.eq(1).mean()

risk_bucket = prediction_df["risk_bucket"]
if risk_bucket.notna().any():
    high_risk_pct = risk_bucket.str.lower().eq("high").mean()
    most_common_risk = risk_bucket.mode().iloc[0] if not risk_bucket.mode().empty else "—"
else:
    high_risk_pct = np.nan
    most_common_risk = "—"

coverage_count = len(prediction_df)

kpi_row_one = st.columns(3)
kpi_row_one[0].metric("Avg sleep probability", f"{avg_sleep:.0%}" if pd.notna(avg_sleep) else "—")
kpi_row_one[1].metric(
    "Avg activity probability", f"{avg_activity:.0%}" if pd.notna(avg_activity) else "—"
)
kpi_row_one[2].metric("Model output coverage", f"{coverage_count:,}")

kpi_row_two = st.columns(3)
kpi_row_two[0].metric(
    "% days sleep classified as good",
    f"{sleep_good_pct:.0%}" if pd.notna(sleep_good_pct) else "—",
)
kpi_row_two[1].metric(
    "% days activity classified as good",
    f"{activity_good_pct:.0%}" if pd.notna(activity_good_pct) else "—",
)
if pd.notna(high_risk_pct):
    kpi_row_two[2].metric("% High risk days", f"{high_risk_pct:.0%}")
else:
    kpi_row_two[2].metric("Most common risk", str(most_common_risk))

st.markdown("---")
st.header("Prediction trends over time")
st.subheader("Daily probabilities with rolling context.")

trend_df = prediction_df[["date", "sleep_proba", "activity_proba"]].copy()
trend_df = trend_df.sort_values("date")

if granularity == "Monthly":
    trend_df = (
        trend_df.set_index("date").resample("MS").mean(numeric_only=True).reset_index()
    )
elif granularity == "Weekly":
    trend_df = (
        trend_df.set_index("date").resample("W-MON").mean(numeric_only=True).reset_index()
    )

rolling_window = 7 if granularity == "Daily" else 3
trend_df["sleep_rolling"] = add_rolling(trend_df["sleep_proba"], rolling_window)
trend_df["activity_rolling"] = add_rolling(trend_df["activity_proba"], rolling_window)

trend_cols = st.columns(2)
with trend_cols[0]:
    st.subheader("Sleep probability")
    if trend_df["sleep_proba"].notna().any():
        sleep_plot_df = trend_df.melt(
            id_vars=["date"],
            value_vars=["sleep_proba", "sleep_rolling"],
            var_name="series",
            value_name="value",
        )
        sleep_plot_df["series"] = sleep_plot_df["series"].replace(
            {
                "sleep_proba": "Sleep probability",
                "sleep_rolling": f"{rolling_window}-period rolling",
            }
        )
        sleep_fig = px.line(
            sleep_plot_df,
            x="date",
            y="value",
            color="series",
            markers=True,
            labels={"value": "Probability", "date": ""},
        )
        sleep_fig.update_yaxes(range=[0, 1])
        st.plotly_chart(sleep_fig, use_container_width=True)
    else:
        st.info("Sleep probability data isn't available for this range.")

with trend_cols[1]:
    st.subheader("Activity probability")
    if trend_df["activity_proba"].notna().any():
        activity_plot_df = trend_df.melt(
            id_vars=["date"],
            value_vars=["activity_proba", "activity_rolling"],
            var_name="series",
            value_name="value",
        )
        activity_plot_df["series"] = activity_plot_df["series"].replace(
            {
                "activity_proba": "Activity probability",
                "activity_rolling": f"{rolling_window}-period rolling",
            }
        )
        activity_fig = px.line(
            activity_plot_df,
            x="date",
            y="value",
            color="series",
            markers=True,
            labels={"value": "Probability", "date": ""},
        )
        activity_fig.update_yaxes(range=[0, 1])
        st.plotly_chart(activity_fig, use_container_width=True)
    else:
        st.info("Activity probability data isn't available for this range.")

st.subheader("Rolling share of good vs bad sleep labels")
label_share_df = prediction_df[["date", "sleep_label"]].dropna().copy()
if label_share_df.empty:
    st.info("Label share data isn't available for this range.")
else:
    label_share_df = label_share_df.sort_values("date")
    if granularity == "Monthly":
        label_share_df = (
            label_share_df.set_index("date")
            .resample("MS")
            .mean(numeric_only=True)
            .reset_index()
        )
    elif granularity == "Weekly":
        label_share_df = (
            label_share_df.set_index("date")
            .resample("W-MON")
            .mean(numeric_only=True)
            .reset_index()
        )
    label_share_df["good_share"] = add_rolling(
        label_share_df["sleep_label"], rolling_window
    )
    label_share_df["bad_share"] = 1 - label_share_df["good_share"]
    share_plot_df = label_share_df.melt(
        id_vars=["date"],
        value_vars=["good_share", "bad_share"],
        var_name="series",
        value_name="value",
    )
    share_plot_df["series"] = share_plot_df["series"].replace(
        {"good_share": "Good", "bad_share": "Bad"}
    )
    area_fig = px.area(
        share_plot_df,
        x="date",
        y="value",
        color="series",
        labels={"value": "Share", "date": ""},
    )
    area_fig.update_yaxes(range=[0, 1])
    st.plotly_chart(area_fig, use_container_width=True)

st.markdown("---")
st.header("Classifications & risk distribution")
st.subheader("Label counts and risk buckets.")

label_cols = st.columns(2)
with label_cols[0]:
    st.subheader("Sleep label counts")
    if prediction_df["sleep_label"].notna().any():
        sleep_counts = (
            prediction_df["sleep_label"]
            .map({1: "Good", 0: "Bad"})
            .fillna("Unknown")
            .value_counts()
            .reset_index()
        )
        sleep_counts.columns = ["label", "count"]
        sleep_bar = px.bar(
            sleep_counts,
            x="label",
            y="count",
            color="label",
            labels={"count": "Days"},
        )
        st.plotly_chart(sleep_bar, use_container_width=True)
    else:
        st.info("Sleep label data isn't available for this range.")

with label_cols[1]:
    st.subheader("Activity label counts")
    if prediction_df["activity_label"].notna().any():
        activity_counts = (
            prediction_df["activity_label"]
            .map({1: "Good", 0: "Bad"})
            .fillna("Unknown")
            .value_counts()
            .reset_index()
        )
        activity_counts.columns = ["label", "count"]
        activity_bar = px.bar(
            activity_counts,
            x="label",
            y="count",
            color="label",
            labels={"count": "Days"},
        )
        st.plotly_chart(activity_bar, use_container_width=True)
    else:
        st.info("Activity label data isn't available for this range.")

st.subheader("Risk bucket distribution")
if prediction_df["risk_bucket"].notna().any():
    risk_counts = (
        prediction_df["risk_bucket"].fillna("Unknown").value_counts().reset_index()
    )
    risk_counts.columns = ["risk_bucket", "count"]
    risk_pie = px.pie(
        risk_counts,
        names="risk_bucket",
        values="count",
        hole=0.5,
    )
    risk_pie.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(risk_pie, use_container_width=True)
else:
    st.info("Risk bucket data isn't available for this range.")

st.markdown("---")
st.header("Scores vs behavior")
st.subheader("Relationships between predictions and daily activity.")

merged_df = prediction_df.copy()
if not features_df.empty:
    merged_df = merged_df.merge(
        features_df,
        on=["user_id", "source", "date"],
        how="left",
        suffixes=("", "_feature"),
    )

steps_column = next(
    (col for col in ["steps", "total_steps"] if col in merged_df.columns), None
)
sleep_minutes_column = next(
    (col for col in ["sleep_minutes", "total_minutes_asleep"] if col in merged_df.columns),
    None,
)
size_column = next(
    (col for col in ["active_minutes", "calories"] if col in merged_df.columns), None
)

scatter_cols = st.columns(2)
with scatter_cols[0]:
    st.subheader("Steps vs sleep probability")
    if (
        steps_column
        and merged_df[steps_column].notna().any()
        and merged_df["sleep_proba"].notna().any()
    ):
        scatter_df = merged_df.dropna(subset=[steps_column, "sleep_proba"]).copy()
        scatter_df[steps_column] = pd.to_numeric(scatter_df[steps_column], errors="coerce")
        scatter_fig = px.scatter(
            scatter_df,
            x=steps_column,
            y="sleep_proba",
            hover_data=["date", steps_column, "sleep_proba"],
            labels={"sleep_proba": "Sleep probability", steps_column: "Steps"},
            trendline="lowess",
        )
        scatter_fig.update_yaxes(range=[0, 1])
        st.plotly_chart(scatter_fig, use_container_width=True)
    else:
        st.info("Steps data isn't available for this range.")

with scatter_cols[1]:
    st.subheader("Sleep minutes vs sleep probability")
    if (
        sleep_minutes_column
        and merged_df[sleep_minutes_column].notna().any()
        and merged_df["sleep_proba"].notna().any()
    ):
        sleep_scatter_df = merged_df.dropna(
            subset=[sleep_minutes_column, "sleep_proba"]
        ).copy()
        sleep_scatter_df[sleep_minutes_column] = pd.to_numeric(
            sleep_scatter_df[sleep_minutes_column], errors="coerce"
        )
        sleep_scatter = px.scatter(
            sleep_scatter_df,
            x=sleep_minutes_column,
            y="sleep_proba",
            hover_data=["date", sleep_minutes_column, "sleep_proba"],
            labels={
                "sleep_proba": "Sleep probability",
                sleep_minutes_column: "Sleep minutes",
            },
        )
        sleep_scatter.update_yaxes(range=[0, 1])
        st.plotly_chart(sleep_scatter, use_container_width=True)
    else:
        st.info("Sleep minutes data isn't available for this range.")

st.subheader("Steps vs sleep minutes (bubble)")
if steps_column and sleep_minutes_column and size_column:
    bubble_df = merged_df.dropna(
        subset=[steps_column, sleep_minutes_column, size_column]
    ).copy()
    bubble_df[steps_column] = pd.to_numeric(bubble_df[steps_column], errors="coerce")
    bubble_df[sleep_minutes_column] = pd.to_numeric(
        bubble_df[sleep_minutes_column], errors="coerce"
    )
    bubble_df[size_column] = pd.to_numeric(bubble_df[size_column], errors="coerce")
    if bubble_df.empty:
        st.info("No overlapping data available for the bubble chart.")
    else:
        bubble_fig = px.scatter(
            bubble_df,
            x=steps_column,
            y=sleep_minutes_column,
            size=size_column,
            color="sleep_proba",
            hover_data=["date", steps_column, sleep_minutes_column, size_column],
            labels={
                steps_column: "Steps",
                sleep_minutes_column: "Sleep minutes",
                "sleep_proba": "Sleep probability",
            },
        )
        st.plotly_chart(bubble_fig, use_container_width=True)
else:
    st.info("Bubble chart requires steps, sleep minutes, and an activity size metric.")

st.markdown("---")
st.header("Distributions")
st.subheader("Probability spread across the selected range.")

hist_cols = st.columns(2)
with hist_cols[0]:
    st.subheader("Sleep probability distribution")
    if prediction_df["sleep_proba"].notna().any():
        sleep_hist = px.histogram(
            prediction_df,
            x="sleep_proba",
            nbins=20,
            labels={"sleep_proba": "Sleep probability"},
        )
        sleep_hist.update_xaxes(range=[0, 1])
        st.plotly_chart(sleep_hist, use_container_width=True)
    else:
        st.info("Sleep probability data isn't available for this range.")

with hist_cols[1]:
    st.subheader("Activity probability distribution")
    if prediction_df["activity_proba"].notna().any():
        activity_hist = px.histogram(
            prediction_df,
            x="activity_proba",
            nbins=20,
            labels={"activity_proba": "Activity probability"},
        )
        activity_hist.update_xaxes(range=[0, 1])
        st.plotly_chart(activity_hist, use_container_width=True)
    else:
        st.info("Activity probability data isn't available for this range.")

st.subheader("Sleep probability by weekday")
weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
if prediction_df["sleep_proba"].notna().any():
    box_fig = px.box(
        prediction_df,
        x="weekday",
        y="sleep_proba",
        category_orders={"weekday": weekday_order},
        labels={"sleep_proba": "Sleep probability", "weekday": ""},
    )
    box_fig.update_yaxes(range=[0, 1])
    st.plotly_chart(box_fig, use_container_width=True)
else:
    st.info("Sleep probability data isn't available for this range.")

st.markdown("---")
st.header("Heatmaps")
st.subheader("Weekday vs month view of sleep probability.")

if prediction_df["sleep_proba"].notna().any():
    heatmap_data = (
        prediction_df.dropna(subset=["sleep_proba"])
        .groupby(["month", "weekday"], as_index=False)
        .agg(avg_sleep_proba=("sleep_proba", "mean"))
    )
    if heatmap_data.empty:
        st.info("Heatmap data isn't available for this range.")
    else:
        heatmap_pivot = heatmap_data.pivot(
            index="month", columns="weekday", values="avg_sleep_proba"
        ).reindex(columns=weekday_order)
        heatmap_fig = px.imshow(
            heatmap_pivot,
            labels={"color": "Avg sleep probability"},
            aspect="auto",
            text_auto=".2f",
        )
        st.plotly_chart(heatmap_fig, use_container_width=True)
else:
    st.info("Sleep probability data isn't available for this range.")

st.markdown("---")
st.header("Timelines / event markers")
st.subheader("High-risk days highlighted across time.")

if prediction_df["risk_bucket"].notna().any():
    high_risk_df = prediction_df[prediction_df["risk_bucket"].str.lower() == "high"]
    if high_risk_df.empty:
        st.info("No high-risk days were detected in this range.")
    else:
        if high_risk_df["risk_score"].notna().any():
            y_metric = "risk_score"
        elif high_risk_df["sleep_proba"].notna().any():
            y_metric = "sleep_proba"
        elif high_risk_df["activity_proba"].notna().any():
            y_metric = "activity_proba"
        else:
            y_metric = None
        if y_metric is None:
            st.info("High-risk days found, but no score values are available to plot.")
        else:
            timeline_fig = px.scatter(
                high_risk_df,
                x="date",
                y=y_metric,
                hover_data=["date", "risk_bucket", y_metric],
                labels={y_metric: "Score", "date": ""},
            )
            st.plotly_chart(timeline_fig, use_container_width=True)
else:
    st.info("Risk bucket data isn't available for this range.")

st.markdown("---")
st.header("Treemap")
st.subheader("Categorical risk columns breakdown.")

categorical_cols = [
    col
    for col in [
        "predicted_health_risk_level",
        "predicted_cardiovascular_strain_risk",
        "predicted_sleep_quality_label",
    ]
    if col in prediction_df.columns
]

if categorical_cols:
    treemap_df = (
        prediction_df[categorical_cols]
        .melt(value_name="category")
        .dropna(subset=["category"])
    )
    treemap_counts = treemap_df.value_counts("category").reset_index()
    treemap_counts.columns = ["category", "count"]
    treemap_fig = px.treemap(
        treemap_counts,
        path=["category"],
        values="count",
    )
    st.plotly_chart(treemap_fig, use_container_width=True)
else:
    st.info("No categorical risk columns available for a treemap.")

st.markdown("---")
st.header("Other advanced views")
st.subheader("Maps, Sankey, or text visuals when data exists.")

location_columns = [
    col
    for col in ["lat", "lon", "latitude", "longitude", "city", "country"]
    if col in prediction_df.columns
]
if location_columns:
    st.info("Location data detected, but map rendering is not configured in this view.")
else:
    st.info("No location data available.")

sankey_possible = safe_has_cols(prediction_df, ["sleep_label"])
if sankey_possible and prediction_df["sleep_label"].notna().any():
    st.info("Sankey view is available for future iterations based on label transitions.")
else:
    st.info("No transition data available for a Sankey chart.")

text_columns = [col for col in ["notes", "tags", "mood"] if col in prediction_df.columns]
if text_columns:
    st.info("Text data detected, but no word cloud is configured in this view.")
else:
    st.info("No textual categories available for word cloud visuals.")

st.markdown("---")
st.header("Steps & sleep minutes")
st.subheader("General KPIs below prediction insights.")

steps_column = None
sleep_minutes_column = None
feature_candidates = features_df.columns if not features_df.empty else []
for col in ("steps", "total_steps"):
    if col in feature_candidates:
        steps_column = col
        break
for col in ("sleep_minutes", "total_minutes_asleep"):
    if col in feature_candidates:
        sleep_minutes_column = col
        break

if features_df.empty or not steps_column or not sleep_minutes_column:
    st.info("Steps or sleep minutes data isn't available for the selected range.")
else:
    trend_metrics = features_df[["date", steps_column, sleep_minutes_column]].copy()
    trend_metrics = trend_metrics.sort_values("date")
    if granularity == "Monthly":
        trend_metrics = (
            trend_metrics.set_index("date")
            .resample("MS")
            .mean(numeric_only=True)
            .reset_index()
        )
    elif granularity == "Weekly":
        trend_metrics = (
            trend_metrics.set_index("date")
            .resample("W-MON")
            .mean(numeric_only=True)
            .reset_index()
        )
    trend_metrics = trend_metrics.rename(
        columns={
            steps_column: "Steps",
            sleep_minutes_column: "Sleep minutes",
        }
    )
    trend_long = trend_metrics.melt(
        id_vars=["date"],
        value_vars=["Steps", "Sleep minutes"],
        var_name="metric",
        value_name="value",
    )
    st.subheader("Steps & sleep minutes over time")
    trend_fig = px.line(
        trend_long,
        x="date",
        y="value",
        color="metric",
        markers=True,
    )
    st.plotly_chart(trend_fig, use_container_width=True)

with st.expander("View data table", expanded=False):
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
    if "date" in display_df.columns:
        display_df["date"] = pd.to_datetime(
            display_df["date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
    if "created_at" in display_df.columns:
        display_df["created_at"] = pd.to_datetime(
            display_df["created_at"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
    st.dataframe(display_df, use_container_width=True)

    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="daily_predictions.csv",
        mime="text/csv",
    )
