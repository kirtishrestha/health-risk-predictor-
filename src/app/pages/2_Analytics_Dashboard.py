"""Production analytics dashboard for daily predictions."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from src.app.ui_predictions import (
    clear_features_cache,
    clear_daily_sleep_cache,
    clear_monthly_metrics_cache,
    clear_prediction_cache,
    clear_prediction_options_cache,
    env_ok,
    load_daily_features,
    load_daily_sleep,
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

with card("Filters", "Refine the dashboard view"):
    filter_cols = st.columns([2, 2, 3, 2, 1])
    with filter_cols[0]:
        selected_user = st.selectbox("User", options=user_options)
    with filter_cols[1]:
        selected_source = st.selectbox("Source", options=source_options)

    prediction_df = load_daily_predictions(selected_user, selected_source)
    min_date = (
        prediction_df["date"].min().date()
        if not prediction_df.empty
        else pd.Timestamp.today().date()
    )
    max_date = (
        prediction_df["date"].max().date()
        if not prediction_df.empty
        else pd.Timestamp.today().date()
    )

    with filter_cols[2]:
        selected_dates = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
    with filter_cols[3]:
        granularity = st.selectbox("Granularity", options=["Daily", "Weekly", "Monthly"])
    with filter_cols[4]:
        if st.button("Refresh"):
            clear_prediction_cache()
            clear_features_cache()
            clear_daily_sleep_cache()
            clear_prediction_options_cache()
            clear_monthly_metrics_cache()
            st.rerun()

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
else:
    start_date = end_date = selected_dates

if prediction_df.empty:
    st.info("No predictions available yet for the selected user and source.")
    st.stop()

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

sleep_df = load_daily_sleep(selected_user, selected_source)
if not sleep_df.empty:
    sleep_mask = (sleep_df["date"].dt.date >= start_date) & (
        sleep_df["date"].dt.date <= end_date
    )
    sleep_df = sleep_df.loc[sleep_mask].copy()

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


def _trend_sentence(current: float, previous: float, label: str) -> str:
    delta = current - previous
    if delta > 0.03:
        direction = "improved"
    elif delta < -0.03:
        direction = "declined"
    else:
        direction = "is stable"
    return f"{label} {direction}"


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

sleep_good_pct = (sleep_label.eq(1).mean() * 100).round(1)
activity_good_pct = (activity_label.eq(1).mean() * 100).round(1)

overall_score = pd.concat([sleep_proba, activity_proba], axis=1).mean(axis=1)
rolling_std = overall_score.rolling(7, min_periods=1).std()
consistency_score = (
    (1 / (1 + rolling_std.mean())) * 100 if not rolling_std.empty else 0.0
)

summary_text = "Add more data to unlock trend summaries."
if len(prediction_df) >= 14:
    sorted_df = prediction_df.sort_values("date")
    recent = sorted_df.tail(7)
    prior = sorted_df.iloc[-14:-7]
    summary_text = (
        f"{_trend_sentence(recent['sleep_proba'].mean(), prior['sleep_proba'].mean(), 'Sleep quality')} "
        f"over the last 7 days; {_trend_sentence(recent['activity_proba'].mean(), prior['activity_proba'].mean(), 'activity')}"
        "."
    )

st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Key performance indicators and a quick readout.</div>',
    unsafe_allow_html=True,
)

with card("Summary"):
    st.markdown(summary_text)

kpi_cols = st.columns(6)
with kpi_cols[0]:
    with card("Days analyzed"):
        st.metric("Days analyzed", f"{len(prediction_df)}")
with kpi_cols[1]:
    with card("Date range"):
        st.metric("Date range", f"{start_date} → {end_date}")
with kpi_cols[2]:
    with card("Sleep good %"):
        st.metric("Sleep good %", f"{sleep_good_pct:.0f}%")
with kpi_cols[3]:
    with card("Activity good %"):
        st.metric("Activity good %", f"{activity_good_pct:.0f}%")
with kpi_cols[4]:
    with card("Consistency score"):
        st.metric("Consistency", f"{consistency_score:.0f}%")
with kpi_cols[5]:
    trend_delta = recent = prior = None
    trend_label = "n/a"
    if len(prediction_df) >= 14:
        recent = prediction_df.sort_values("date").tail(7)
        prior = prediction_df.sort_values("date").iloc[-14:-7]
        trend_delta = recent["sleep_proba"].mean() - prior["sleep_proba"].mean()
        trend_label = f"{trend_delta * 100:+.1f} pts"
    with card("Sleep trend"):
        st.metric("Sleep trend", trend_label)

st.markdown('<div class="section-title">Trends</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Daily performance with 7-day rolling averages.</div>',
    unsafe_allow_html=True,
)

trend_df = prediction_df[["date", "sleep_proba", "activity_proba"]].copy()
trend_df = trend_df.sort_values("date")
trend_df["sleep_rolling"] = trend_df["sleep_proba"].rolling(7, min_periods=1).mean()
trend_df["activity_rolling"] = trend_df["activity_proba"].rolling(7, min_periods=1).mean()

if granularity != "Daily":
    freq = "W-MON" if granularity == "Weekly" else "MS"
    trend_df = (
        trend_df.set_index("date")
        .resample(freq)
        .mean(numeric_only=True)
        .reset_index()
    )

if not trend_df.empty and (
    trend_df["sleep_proba"].notna().any() or trend_df["activity_proba"].notna().any()
):
    trend_cols = st.columns(2)
    with trend_cols[0]:
        with card("Sleep probability trend", "Daily values with 7-day rolling average"):
            if trend_df["sleep_proba"].notna().any():
                sleep_chart = (
                    alt.Chart(trend_df)
                    .transform_fold(
                        ["sleep_proba", "sleep_rolling"],
                        as_=["series", "value"],
                    )
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("date:T", title=None),
                        y=alt.Y(
                            "value:Q",
                            title="Probability",
                            scale=alt.Scale(domain=[0, 1]),
                        ),
                        color=alt.Color(
                            "series:N",
                            scale=alt.Scale(
                                domain=["sleep_proba", "sleep_rolling"],
                                range=["#4C78A8", "#72B7B2"],
                            ),
                            legend=alt.Legend(title=None),
                        ),
                        tooltip=["date:T", "value:Q"],
                    )
                    .properties(height=280)
                )
                st.altair_chart(sleep_chart, use_container_width=True)
            else:
                st.info("No sleep probability data for the selected range.")

    with trend_cols[1]:
        with card(
            "Activity probability trend",
            "Daily values with 7-day rolling average",
        ):
            if trend_df["activity_proba"].notna().any():
                activity_chart = (
                    alt.Chart(trend_df)
                    .transform_fold(
                        ["activity_proba", "activity_rolling"],
                        as_=["series", "value"],
                    )
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("date:T", title=None),
                        y=alt.Y(
                            "value:Q",
                            title="Probability",
                            scale=alt.Scale(domain=[0, 1]),
                        ),
                        color=alt.Color(
                            "series:N",
                            scale=alt.Scale(
                                domain=["activity_proba", "activity_rolling"],
                                range=["#F58518", "#E45756"],
                            ),
                            legend=alt.Legend(title=None),
                        ),
                        tooltip=["date:T", "value:Q"],
                    )
                    .properties(height=280)
                )
                st.altair_chart(activity_chart, use_container_width=True)
            else:
                st.info("No activity probability data for the selected range.")
else:
    with card("Daily trend data"):
        st.info(
            "No probability trends available for the selected range. "
            "Run inference to generate daily predictions."
        )

sleep_anchor = prediction_df[["user_id", "source", "date"]].drop_duplicates()
if sleep_df.empty:
    sleep_sleep_df = pd.DataFrame(
        columns=["user_id", "source", "date", "sleep_minutes_daily_sleep"]
    )
else:
    sleep_sleep_df = sleep_df.rename(
        columns={"sleep_minutes": "sleep_minutes_daily_sleep"}
    )

if features_df.empty:
    sleep_feature_df = pd.DataFrame(
        columns=["user_id", "source", "date", "sleep_minutes_daily_features"]
    )
else:
    sleep_feature_df = features_df[
        ["user_id", "source", "date", "sleep_minutes"]
    ].rename(columns={"sleep_minutes": "sleep_minutes_daily_features"})

sleep_merged = (
    sleep_anchor.merge(sleep_sleep_df, on=["user_id", "source", "date"], how="left")
    .merge(sleep_feature_df, on=["user_id", "source", "date"], how="left")
    .sort_values("date")
)
sleep_merged["sleep_minutes"] = pd.to_numeric(
    sleep_merged["sleep_minutes_daily_sleep"], errors="coerce"
).fillna(
    pd.to_numeric(sleep_merged["sleep_minutes_daily_features"], errors="coerce")
)
sleep_chart_df = sleep_merged.dropna(subset=["sleep_minutes"]).copy()
if granularity != "Daily" and not sleep_chart_df.empty:
    freq = "W-MON" if granularity == "Weekly" else "MS"
    sleep_chart_df = (
        sleep_chart_df.set_index("date")
        .resample(freq)
        .mean(numeric_only=True)
        .reset_index()
    )

with card("Sleep duration", "Daily minutes asleep from Supabase sleep tables"):
    if sleep_chart_df.empty:
        st.info(
            "No sleep minutes data available for this selection. "
            "Run inference or sync sleep data to populate this chart."
        )
    else:
        sleep_minutes_chart = (
            alt.Chart(sleep_chart_df)
            .mark_line(point=True, color="#4C78A8")
            .encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("sleep_minutes:Q", title="Minutes asleep"),
                tooltip=["date:T", "sleep_minutes:Q"],
            )
            .properties(height=280)
        )
        st.altair_chart(sleep_minutes_chart, use_container_width=True)

if granularity == "Monthly":
    with card("Monthly metrics", "Aggregated from monthly_metrics"):
        if monthly_df.empty:
            st.info("No monthly metrics available for this selection.")
        else:
            monthly_long = monthly_df.melt(
                id_vars=["month"],
                value_vars=["sleep_days_count", "activity_days_count"],
                var_name="metric",
                value_name="days",
            )
            monthly_chart = (
                alt.Chart(monthly_long)
                .mark_bar()
                .encode(
                    x=alt.X("month:T", title=None),
                    y=alt.Y("days:Q", title="Days tracked"),
                    color=alt.Color(
                        "metric:N",
                        scale=alt.Scale(range=["#72B7B2", "#F58518"]),
                        legend=alt.Legend(title=None),
                    ),
                    tooltip=["month:T", "metric:N", "days:Q"],
                )
                .properties(height=280)
            )
            st.altair_chart(monthly_chart, use_container_width=True)

st.markdown('<div class="section-title">Distributions</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">How sleep and activity labels break down over time.</div>',
    unsafe_allow_html=True,
)

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

label_cols = st.columns(2)
with label_cols[0]:
    with card("Sleep label counts"):
        sleep_bar = (
            alt.Chart(label_counts)
            .mark_bar(color="#4C78A8")
            .encode(
                x=alt.X("label:O", title="Label"),
                y=alt.Y("sleep:Q", title="Days"),
                tooltip=["label:O", "sleep:Q"],
            )
            .properties(height=240)
        )
        st.altair_chart(sleep_bar, use_container_width=True)

with label_cols[1]:
    with card("Activity label counts"):
        activity_bar = (
            alt.Chart(label_counts)
            .mark_bar(color="#F58518")
            .encode(
                x=alt.X("label:O", title="Label"),
                y=alt.Y("activity:Q", title="Days"),
                tooltip=["label:O", "activity:Q"],
            )
            .properties(height=240)
        )
        st.altair_chart(activity_bar, use_container_width=True)

st.markdown('<div class="section-title">Behavior Insights</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Daily features connected to quality predictions.</div>',
    unsafe_allow_html=True,
)

behavior_cols = st.columns(3)
with behavior_cols[0]:
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

with behavior_cols[1]:
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

with behavior_cols[2]:
    with card("Sleep minutes vs sleep label"):
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
                    tooltip=["sleep_label:O", "sleep_minutes:Q"],
                )
                .properties(height=260)
            )
            st.altair_chart(boxplot, use_container_width=True)

with st.expander("View data", expanded=False):
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
