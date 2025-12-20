"""Production analytics dashboard for daily predictions."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

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
from src.app.ui_style import card, inject_global_css


st.set_page_config(page_title="Analytics Dashboard", page_icon="📊", layout="wide")

inject_global_css()

st.markdown('<div class="hrp-title">Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hrp-subtitle">Personal health analytics from Supabase-powered daily insights.</div>',
    unsafe_allow_html=True,
)

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


with card("Filters", "Refine your dashboard view", class_name="filter-card"):
    filter_cols = st.columns([2, 2, 3, 2.2, 1])
    with filter_cols[0]:
        selected_user = st.selectbox("User", options=user_options)
    with filter_cols[1]:
        selected_source = st.selectbox("Source", options=source_options)

    prediction_df = load_daily_predictions(selected_user, selected_source)
    if prediction_df.empty:
        st.info("No predictions available yet for the selected user and source.")
        st.stop()

    min_date = prediction_df["date"].min().date()
    max_date = prediction_df["date"].max().date()

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
        if st.button("Refresh"):
            clear_prediction_cache()
            clear_features_cache()
            clear_prediction_options_cache()
            clear_monthly_metrics_cache()
            st.rerun()

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
else:
    start_date = end_date = selected_dates

prediction_df = prediction_df.copy()
mask = (prediction_df["date"].dt.date >= start_date) & (
    prediction_df["date"].dt.date <= end_date
)
prediction_df = prediction_df.loc[mask].copy()

features_df = load_daily_features(selected_user, selected_source)
if not features_df.empty:
    feature_mask = (features_df["date"].dt.date >= start_date) & (
        features_df["date"].dt.date <= end_date
    )
    features_df = features_df.loc[feature_mask].copy()

monthly_df = load_monthly_metrics(selected_user, selected_source)
if not monthly_df.empty:
    monthly_mask = (monthly_df["month"].dt.date >= start_date) & (
        monthly_df["month"].dt.date <= end_date
    )
    monthly_df = monthly_df.loc[monthly_mask].copy()

if prediction_df.empty:
    st.info("No predictions found for the selected date range.")
    st.stop()


def _coalesce_probability(label_series: pd.Series, proba_series: pd.Series) -> pd.Series:
    label_numeric = pd.to_numeric(label_series, errors="coerce")
    proba_numeric = pd.to_numeric(proba_series, errors="coerce")
    return proba_numeric.fillna(label_numeric)


def _format_percent(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.0f}%"


def _format_number(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:,.0f}"


def _render_kpi(label: str, value: str, context: str | None = None) -> None:
    with card(class_name="kpi-card"):
        st.markdown(f'<div class="kpi-value">{value}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-label">{label}</div>', unsafe_allow_html=True)
        if context:
            st.markdown(f'<div class="kpi-context">{context}</div>', unsafe_allow_html=True)


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

sleep_good_pct = sleep_label.eq(1).mean() * 100 if sleep_label.notna().any() else None
activity_good_pct = (
    activity_label.eq(1).mean() * 100 if activity_label.notna().any() else None
)

overall_score = pd.concat([sleep_proba, activity_proba], axis=1).mean(axis=1)
rolling_std = overall_score.rolling(7, min_periods=1).std()
consistency_score = (
    (1 / (1 + rolling_std.mean())) * 100 if not rolling_std.empty else None
)

sleep_trend_value = "—"
if len(prediction_df) >= 14:
    sorted_df = prediction_df.sort_values("date")
    recent = sorted_df.tail(7)
    prior = sorted_df.iloc[-14:-7]
    trend_delta = recent["sleep_proba"].mean() - prior["sleep_proba"].mean()
    sleep_trend_value = f"{trend_delta * 100:+.1f} pts"

st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Key performance indicators for the selected view.</div>',
    unsafe_allow_html=True,
)

kpi_cols = st.columns(4)
with kpi_cols[0]:
    _render_kpi("Days analyzed", _format_number(len(prediction_df)))
with kpi_cols[1]:
    date_label = f"{start_date} → {end_date}" if start_date and end_date else "—"
    _render_kpi("Date range", date_label)
with kpi_cols[2]:
    _render_kpi("Sleep good %", _format_percent(sleep_good_pct))
with kpi_cols[3]:
    _render_kpi("Activity good %", _format_percent(activity_good_pct))

secondary_cols = st.columns(2)
with secondary_cols[0]:
    _render_kpi("Consistency score", _format_percent(consistency_score))
with secondary_cols[1]:
    _render_kpi("Sleep trend", sleep_trend_value, "vs previous 7 days")

st.markdown('<div class="section-title">Trends</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Sleep and activity probabilities over time.</div>',
    unsafe_allow_html=True,
)

trend_df = prediction_df[["date", "sleep_proba", "activity_proba"]].copy()
trend_df = trend_df.sort_values("date")

if granularity == "Monthly":
    monthly_index = monthly_df.rename(columns={"month": "date"})[["date"]]
    monthly_rollup = (
        trend_df.set_index("date").resample("MS").mean(numeric_only=True).reset_index()
    )
    if not monthly_index.empty:
        trend_df = monthly_index.merge(monthly_rollup, on="date", how="left")
    else:
        trend_df = monthly_rollup
elif granularity == "Weekly":
    trend_df = (
        trend_df.set_index("date").resample("W-MON").mean(numeric_only=True).reset_index()
    )

rolling_window = 7 if granularity == "Daily" else 3
if len(trend_df) >= rolling_window:
    trend_df["sleep_rolling"] = trend_df["sleep_proba"].rolling(
        rolling_window, min_periods=1
    ).mean()
    trend_df["activity_rolling"] = trend_df["activity_proba"].rolling(
        rolling_window, min_periods=1
    ).mean()


def _trend_chart(
    data: pd.DataFrame,
    base_col: str,
    rolling_col: str,
    title: str,
    subtitle: str,
    colors: list[str],
) -> alt.Chart:
    series_cols = [base_col]
    if rolling_col in data.columns:
        series_cols.append(rolling_col)

    chart = (
        alt.Chart(data)
        .transform_fold(series_cols, as_=["series", "value"])
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("value:Q", title="Probability", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=series_cols, range=colors),
                legend=alt.Legend(title=None),
            ),
            tooltip=["date:T", alt.Tooltip("value:Q", format=".2f")],
        )
        .properties(height=280)
    )

    return chart.properties(title={"text": title, "subtitle": subtitle})


if trend_df.empty or (
    trend_df["sleep_proba"].isna().all() and trend_df["activity_proba"].isna().all()
):
    with card("Trend insights"):
        st.info("Trend data isn't available for the selected range yet.")
else:
    trend_cols = st.columns(2)
    with trend_cols[0]:
        with card(
            "Sleep probability trend",
            "Rolling averages smooth recent changes.",
        ):
            if trend_df["sleep_proba"].notna().any():
                sleep_chart = _trend_chart(
                    trend_df,
                    "sleep_proba",
                    "sleep_rolling",
                    "Sleep probability",
                    f"{granularity} trend",
                    ["#4C78A8", "#72B7B2"],
                )
                st.altair_chart(sleep_chart, use_container_width=True)
            else:
                st.info("Sleep probability data isn't available for this range.")

    with trend_cols[1]:
        with card(
            "Activity probability trend",
            "Rolling averages smooth recent changes.",
        ):
            if trend_df["activity_proba"].notna().any():
                activity_chart = _trend_chart(
                    trend_df,
                    "activity_proba",
                    "activity_rolling",
                    "Activity probability",
                    f"{granularity} trend",
                    ["#F58518", "#E45756"],
                )
                st.altair_chart(activity_chart, use_container_width=True)
            else:
                st.info("Activity probability data isn't available for this range.")

st.markdown('<div class="section-title">Insights</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Behavior signals connected to quality predictions.</div>',
    unsafe_allow_html=True,
)

if features_df.empty:
    with card("Insights"):
        st.info(
            "Daily feature metrics aren't available yet. Sync wearable data to unlock deeper insights."
        )
else:
    insights_cols = st.columns(3)
    with insights_cols[0]:
        with card("Day-of-week heatmap", "Average sleep probability"):
            heatmap_df = prediction_df[["date", "sleep_proba"]].copy()
            heatmap_df["day_of_week"] = heatmap_df["date"].dt.day_name()
            heatmap_df["week"] = heatmap_df["date"].dt.to_period("W-MON").dt.start_time
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
                heatmap_df.dropna(subset=["sleep_proba"])
                .groupby(["week", "day_of_week"], as_index=False)
                .agg(avg_proba=("sleep_proba", "mean"))
            )
            if heatmap_summary.empty:
                st.info("Not enough sleep probability data to build the heatmap.")
            else:
                heatmap = (
                    alt.Chart(heatmap_summary)
                    .mark_rect()
                    .encode(
                        x=alt.X("day_of_week:N", sort=day_order, title=None),
                        y=alt.Y("week:T", title="Week of"),
                        color=alt.Color("avg_proba:Q", title="Avg. prob"),
                        tooltip=["week:T", "day_of_week:N", "avg_proba:Q"],
                    )
                    .properties(height=260)
                )
                st.altair_chart(heatmap, use_container_width=True)

    with insights_cols[1]:
        with card("Steps vs activity probability"):
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
                    .properties(height=260)
                )
                st.altair_chart(scatter, use_container_width=True)

    with insights_cols[2]:
        with card("Label distribution", "Sleep vs activity labels"):
            label_counts = (
                pd.DataFrame(
                    {
                        "sleep": prediction_df["sleep_label"]
                        .value_counts()
                        .sort_index(),
                        "activity": prediction_df["activity_label"]
                        .value_counts()
                        .sort_index(),
                    }
                )
                .fillna(0)
                .reset_index()
                .rename(columns={"index": "label"})
            )
            label_long = label_counts.melt(
                id_vars=["label"],
                value_vars=["sleep", "activity"],
                var_name="type",
                value_name="count",
            )
            distribution = (
                alt.Chart(label_long)
                .mark_bar()
                .encode(
                    x=alt.X("label:O", title="Label"),
                    y=alt.Y("count:Q", title="Days"),
                    color=alt.Color(
                        "type:N",
                        scale=alt.Scale(range=["#4C78A8", "#F58518"]),
                        legend=alt.Legend(title=None),
                    ),
                    tooltip=["label:O", "type:N", "count:Q"],
                )
                .properties(height=260)
            )
            st.altair_chart(distribution, use_container_width=True)

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
    st.dataframe(display_df, use_container_width=True)

    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="daily_predictions.csv",
        mime="text/csv",
    )
