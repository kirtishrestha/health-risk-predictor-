"""Production analytics dashboard for daily predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.app.ui import card, empty_state, section_header, set_page_theme
from src.app.ui_helpers import add_rolling, compute_risk_bucket
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
from src.app.viz_guards import has_rows, require_columns


st.set_page_config(page_title="Analytics Dashboard", page_icon="📊", layout="wide")
set_page_theme()

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
low_max = st.sidebar.slider(
    "Low risk max score",
    min_value=0.0,
    max_value=1.0,
    value=0.35,
    step=0.05,
)
high_min = st.sidebar.slider(
    "High risk min score",
    min_value=0.0,
    max_value=1.0,
    value=0.65,
    step=0.05,
)


def normalize_thresholds(low_value: float, high_value: float) -> tuple[float, float]:
    if low_value >= high_value:
        st.sidebar.warning(
            "Low risk max must be lower than high risk min. Values were adjusted."
        )
        low_value, high_value = high_value, low_value
        if low_value >= high_value:
            high_value = min(1.0, low_value + 0.05)
            low_value = max(0.0, high_value - 0.05)
    return low_value, high_value


low_max, high_min = normalize_thresholds(low_max, high_min)

with st.sidebar.expander("Developer toggles", expanded=False):
    show_experimental = st.checkbox(
        "Show experimental charts (maps/sankey/treemap/wordcloud)", value=False
    )
    show_placeholders = st.checkbox("Show placeholder empty-states", value=False)
    show_debug = st.checkbox("Show debug info (df shape, columns)", value=False)

st.session_state["show_debug_info"] = show_debug


def normalize_date_range(value) -> tuple[pd.Timestamp, pd.Timestamp]:
    if isinstance(value, tuple) and len(value) == 2:
        return value[0], value[1]
    return value, value


def render_card_or_placeholder(
    title: str,
    subtitle: str | None,
    enabled: bool,
    placeholder_message: str,
    render_fn,
) -> bool:
    if enabled:
        with card(title, subtitle):
            render_fn()
        return True
    if show_placeholders:
        with card(title, subtitle):
            empty_state(placeholder_message, "ℹ️")
    return False


section_header("Filters", "Refine your dashboard view.")
with card():
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

st.divider()

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

if show_debug:
    st.sidebar.write(f"Predictions: {prediction_df.shape}")
    st.sidebar.write(sorted(prediction_df.columns))
    if not features_df.empty:
        st.sidebar.write(f"Features: {features_df.shape}")
        st.sidebar.write(sorted(features_df.columns))
    if not monthly_df.empty:
        st.sidebar.write(f"Monthly metrics: {monthly_df.shape}")
        st.sidebar.write(sorted(monthly_df.columns))

sleep_proba = pd.Series(np.nan, index=prediction_df.index)
for col in [
    "sleep_quality_proba",
    "predicted_sleep_probability",
    "predicted_sleep_proba",
]:
    if col in prediction_df.columns:
        sleep_proba = sleep_proba.fillna(pd.to_numeric(prediction_df[col], errors="coerce"))

activity_proba = pd.Series(np.nan, index=prediction_df.index)
for col in [
    "activity_quality_proba",
    "predicted_activity_probability",
    "predicted_activity_proba",
]:
    if col in prediction_df.columns:
        activity_proba = activity_proba.fillna(
            pd.to_numeric(prediction_df[col], errors="coerce")
        )

sleep_label = pd.Series(np.nan, index=prediction_df.index)
for col in ["sleep_quality_label", "predicted_sleep_label"]:
    if col in prediction_df.columns:
        sleep_label = sleep_label.fillna(pd.to_numeric(prediction_df[col], errors="coerce"))

activity_label = pd.Series(np.nan, index=prediction_df.index)
for col in ["activity_quality_label", "predicted_activity_label"]:
    if col in prediction_df.columns:
        activity_label = activity_label.fillna(
            pd.to_numeric(prediction_df[col], errors="coerce")
        )

sleep_label = sleep_label.fillna((sleep_proba >= 0.5).astype(int))
activity_label = activity_label.fillna((activity_proba >= 0.5).astype(int))

prediction_df["sleep_proba"] = sleep_proba
prediction_df["activity_proba"] = activity_proba
prediction_df["sleep_label"] = sleep_label
prediction_df["activity_label"] = activity_label
prediction_df["weekday"] = prediction_df["date"].dt.day_name()
prediction_df["month"] = prediction_df["date"].dt.month_name()

risk_label_candidates = [
    "predicted_health_risk_level",
    "predicted_cardiovascular_strain_risk",
    "predicted_sleep_quality_label",
    "health_risk_level",
    "risk_level",
]
risk_label_column = next(
    (col for col in risk_label_candidates if col in prediction_df.columns), None
)

risk_score_candidates = [
    "predicted_risk_score",
    "predicted_health_risk_score",
]
risk_score_columns = [col for col in risk_score_candidates if col in prediction_df.columns]
invert_scores = False
score_columns = risk_score_columns
if not score_columns:
    score_columns = [
        col for col in ["sleep_proba", "activity_proba"] if col in prediction_df.columns
    ]
    invert_scores = True

prediction_df = compute_risk_bucket(
    prediction_df,
    low_threshold=low_max,
    high_threshold=high_min,
    score_columns=score_columns,
    existing_column=risk_label_column,
    invert_scores=invert_scores,
)


section_header("Predictions at a glance", "Model outputs front-and-center.")
with card():
    avg_sleep = sleep_proba.mean()
    avg_activity = activity_proba.mean()
    sleep_good_pct = sleep_label.eq(1).mean()
    activity_good_pct = activity_label.eq(1).mean()

    risk_bucket = prediction_df["risk_bucket"]
    if risk_bucket.notna().any():
        high_risk_pct = risk_bucket.str.lower().eq("high").mean()
        most_common_risk = (
            risk_bucket.mode().iloc[0] if not risk_bucket.mode().empty else "—"
        )
    else:
        high_risk_pct = np.nan
        most_common_risk = "—"

    coverage_count = len(prediction_df)

    kpi_row_one = st.columns(3)
    kpi_row_one[0].metric(
        "Avg sleep probability", f"{avg_sleep:.0%}" if pd.notna(avg_sleep) else "—"
    )
    kpi_row_one[1].metric(
        "Avg activity probability",
        f"{avg_activity:.0%}" if pd.notna(avg_activity) else "—",
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

st.divider()

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

sleep_trend_available = trend_df["sleep_proba"].notna().any()
activity_trend_available = trend_df["activity_proba"].notna().any()

label_share_df = prediction_df[["date", "sleep_label"]].dropna().copy()
if not label_share_df.empty:
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
    label_share_available = label_share_df["good_share"].notna().any()
else:
    label_share_available = False

if sleep_trend_available or activity_trend_available or label_share_available or show_placeholders:
    section_header("Prediction trends over time", "Daily probabilities with rolling context.")
    with card():
        trend_cols = st.columns(2)
        with trend_cols[0]:
            if sleep_trend_available:
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
            elif show_placeholders:
                empty_state("Sleep probability data isn't available for this range.", "ℹ️")

        with trend_cols[1]:
            if activity_trend_available:
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
            elif show_placeholders:
                empty_state(
                    "Activity probability data isn't available for this range.", "ℹ️"
                )

        if label_share_available:
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
        elif show_placeholders:
            empty_state("Label share data isn't available for this range.", "ℹ️")

    st.divider()

sleep_label_available = prediction_df["sleep_label"].notna().any()
activity_label_available = prediction_df["activity_label"].notna().any()
risk_bucket_available = prediction_df["risk_bucket"].notna().any()

if (
    sleep_label_available
    or activity_label_available
    or risk_bucket_available
    or show_placeholders
):
    section_header("Classifications & risk distribution", "Label counts and risk buckets.")
    with card():
        label_cols = st.columns(2)
        with label_cols[0]:
            if sleep_label_available:
                sleep_counts = (
                    prediction_df["sleep_label"]
                    .replace({1: "Good", 0: "Bad"})
                    .fillna("Unknown")
                    .astype(str)
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
            elif show_placeholders:
                empty_state("Sleep label data isn't available for this range.", "ℹ️")

        with label_cols[1]:
            if activity_label_available:
                activity_counts = (
                    prediction_df["activity_label"]
                    .replace({1: "Good", 0: "Bad"})
                    .fillna("Unknown")
                    .astype(str)
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
            elif show_placeholders:
                empty_state("Activity label data isn't available for this range.", "ℹ️")

        if risk_bucket_available:
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
        elif show_placeholders:
            empty_state("Risk bucket data isn't available for this range.", "ℹ️")

    st.divider()

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
    (
        col
        for col in ["sleep_minutes", "total_minutes_asleep"]
        if col in merged_df.columns
    ),
    None,
)
size_column = next(
    (col for col in ["active_minutes", "calories"] if col in merged_df.columns), None
)

steps_vs_sleep_available = (
    steps_column
    and merged_df[steps_column].notna().any()
    and merged_df["sleep_proba"].notna().any()
)
sleep_minutes_available = (
    sleep_minutes_column
    and merged_df[sleep_minutes_column].notna().any()
    and merged_df["sleep_proba"].notna().any()
)
bubble_available = (
    steps_column
    and sleep_minutes_column
    and size_column
    and merged_df[[steps_column, sleep_minutes_column, size_column]].notna().any().any()
)

if (
    steps_vs_sleep_available
    or sleep_minutes_available
    or bubble_available
    or show_placeholders
):
    section_header("Scores vs behavior", "Relationships between predictions and activity.")
    with card():
        scatter_cols = st.columns(2)
        with scatter_cols[0]:
            if steps_vs_sleep_available:
                scatter_df = merged_df.dropna(subset=[steps_column, "sleep_proba"]).copy()
                scatter_df[steps_column] = pd.to_numeric(
                    scatter_df[steps_column], errors="coerce"
                )
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
            elif show_placeholders:
                empty_state("Steps data isn't available for this range.", "ℹ️")

        with scatter_cols[1]:
            if sleep_minutes_available:
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
            elif show_placeholders:
                empty_state("Sleep minutes data isn't available for this range.", "ℹ️")

        if bubble_available:
            bubble_df = merged_df.dropna(
                subset=[steps_column, sleep_minutes_column, size_column]
            ).copy()
            bubble_df[steps_column] = pd.to_numeric(
                bubble_df[steps_column], errors="coerce"
            )
            bubble_df[sleep_minutes_column] = pd.to_numeric(
                bubble_df[sleep_minutes_column], errors="coerce"
            )
            bubble_df[size_column] = pd.to_numeric(bubble_df[size_column], errors="coerce")
            if not bubble_df.empty:
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
            elif show_placeholders:
                empty_state("No overlapping data available for the bubble chart.", "ℹ️")
        elif show_placeholders:
            empty_state(
                "Bubble chart requires steps, sleep minutes, and an activity size metric.",
                "ℹ️",
            )

    st.divider()

sleep_dist_available = prediction_df["sleep_proba"].notna().any()
activity_dist_available = prediction_df["activity_proba"].notna().any()

if sleep_dist_available or activity_dist_available or show_placeholders:
    section_header("Distributions", "Probability spread across the selected range.")
    with card():
        hist_cols = st.columns(2)
        with hist_cols[0]:
            if sleep_dist_available:
                sleep_hist = px.histogram(
                    prediction_df,
                    x="sleep_proba",
                    nbins=20,
                    labels={"sleep_proba": "Sleep probability"},
                )
                sleep_hist.update_xaxes(range=[0, 1])
                st.plotly_chart(sleep_hist, use_container_width=True)
            elif show_placeholders:
                empty_state("Sleep probability data isn't available for this range.", "ℹ️")

        with hist_cols[1]:
            if activity_dist_available:
                activity_hist = px.histogram(
                    prediction_df,
                    x="activity_proba",
                    nbins=20,
                    labels={"activity_proba": "Activity probability"},
                )
                activity_hist.update_xaxes(range=[0, 1])
                st.plotly_chart(activity_hist, use_container_width=True)
            elif show_placeholders:
                empty_state(
                    "Activity probability data isn't available for this range.", "ℹ️"
                )

        if sleep_dist_available:
            weekday_order = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
            box_fig = px.box(
                prediction_df,
                x="weekday",
                y="sleep_proba",
                category_orders={"weekday": weekday_order},
                labels={"sleep_proba": "Sleep probability", "weekday": ""},
            )
            box_fig.update_yaxes(range=[0, 1])
            st.plotly_chart(box_fig, use_container_width=True)
        elif show_placeholders:
            empty_state("Sleep probability data isn't available for this range.", "ℹ️")

    st.divider()

heatmap_available = prediction_df["sleep_proba"].notna().any()
if heatmap_available or show_placeholders:
    section_header("Heatmaps", "Weekday vs month view of sleep probability.")
    if heatmap_available:
        heatmap_data = (
            prediction_df.dropna(subset=["sleep_proba"])
            .groupby(["month", "weekday"], as_index=False)
            .agg(avg_sleep_proba=("sleep_proba", "mean"))
        )
        heatmap_available = not heatmap_data.empty

    if heatmap_available:
        weekday_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        heatmap_pivot = heatmap_data.pivot(
            index="month", columns="weekday", values="avg_sleep_proba"
        ).reindex(columns=weekday_order)
        with card():
            heatmap_fig = px.imshow(
                heatmap_pivot,
                labels={"color": "Avg sleep probability"},
                aspect="auto",
                text_auto=".2f",
            )
            st.plotly_chart(heatmap_fig, use_container_width=True)
    elif show_placeholders:
        with card():
            empty_state("Heatmap data isn't available for this range.", "ℹ️")

    st.divider()

timeline_available = prediction_df["risk_bucket"].notna().any()
if timeline_available or show_placeholders:
    section_header("Timelines / event markers", "High-risk days highlighted over time.")
    if timeline_available:
        high_risk_df = prediction_df[prediction_df["risk_bucket"].str.lower() == "high"]
        if not high_risk_df.empty:
            if high_risk_df["risk_score"].notna().any():
                y_metric = "risk_score"
            elif high_risk_df["sleep_proba"].notna().any():
                y_metric = "sleep_proba"
            elif high_risk_df["activity_proba"].notna().any():
                y_metric = "activity_proba"
            else:
                y_metric = None

            if y_metric is not None:
                with card():
                    timeline_fig = px.scatter(
                        high_risk_df,
                        x="date",
                        y=y_metric,
                        hover_data=["date", "risk_bucket", y_metric],
                        labels={y_metric: "Score", "date": ""},
                    )
                    st.plotly_chart(timeline_fig, use_container_width=True)
            elif show_placeholders:
                with card():
                    empty_state(
                        "High-risk days found, but no score values are available to plot.",
                        "ℹ️",
                    )
        elif show_placeholders:
            with card():
                empty_state("No high-risk days were detected in this range.", "ℹ️")
    elif show_placeholders:
        with card():
            empty_state("Risk bucket data isn't available for this range.", "ℹ️")

    st.divider()

treemap_columns = [
    col
    for col in [
        "predicted_health_risk_level",
        "predicted_cardiovascular_strain_risk",
        "predicted_sleep_quality_label",
    ]
    if col in prediction_df.columns
]
treemap_available = bool(treemap_columns) and prediction_df[treemap_columns].notna().any().any()


def render_treemap() -> None:
    treemap_df = (
        prediction_df[treemap_columns]
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


location_columns = [
    col
    for col in ["lat", "lon", "latitude", "longitude", "city", "country"]
    if col in prediction_df.columns
]

def render_location_placeholder() -> None:
    empty_state("Location data detected, but map rendering is not configured.", "🗺️")


sankey_possible = require_columns(prediction_df, ["sleep_label"])
def render_sankey_placeholder() -> None:
    empty_state(
        "Sankey view is available for future iterations based on label transitions.",
        "🔀",
    )


text_columns = [col for col in ["notes", "tags", "mood"] if col in prediction_df.columns]
def render_wordcloud_placeholder() -> None:
    empty_state("Text data detected, but no word cloud is configured.", "☁️")


if show_experimental:
    section_header(
        "Advanced/experimental views", "Optional visuals enabled via developer toggles."
    )
    render_card_or_placeholder(
        "Treemap",
        "Categorical risk columns breakdown.",
        treemap_available,
        "No categorical risk columns available for a treemap.",
        render_treemap,
    )

    render_card_or_placeholder(
        "Location map",
        "Geospatial context (placeholder).",
        bool(location_columns),
        "No location data available.",
        render_location_placeholder,
    )

    render_card_or_placeholder(
        "Sankey transitions",
        "Label transitions (placeholder).",
        sankey_possible,
        "No transition data available for a Sankey chart.",
        render_sankey_placeholder,
    )

    render_card_or_placeholder(
        "Word cloud",
        "Textual sentiment or tags (placeholder).",
        bool(text_columns),
        "No textual categories available for word cloud visuals.",
        render_wordcloud_placeholder,
    )

    st.divider()

section_header("General KPIs", "Steps and sleep minutes over time.")

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

steps_kpi_available = (
    not features_df.empty and steps_column and sleep_minutes_column and has_rows(features_df)
)

if steps_kpi_available or show_placeholders:
    if steps_kpi_available:
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
        with card():
            trend_fig = px.line(
                trend_long,
                x="date",
                y="value",
                color="metric",
                markers=True,
            )
            st.plotly_chart(trend_fig, use_container_width=True)
    elif show_placeholders:
        with card():
            empty_state("Steps or sleep minutes data isn't available.", "ℹ️")


section_header("Data export", "Detailed predictions for the selected range.")
with card():
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
