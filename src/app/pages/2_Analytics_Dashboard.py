"""Production analytics dashboard for daily predictions."""

from __future__ import annotations

import altair as alt
import pandas as pd
import plotly.express as px
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


def normalize_date_range(value) -> tuple[pd.Timestamp, pd.Timestamp]:
    if isinstance(value, tuple) and len(value) == 2:
        return value[0], value[1]
    return value, value


def kpi_card(title: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="kpi-context">{sub}</div>' if sub else ""
    return f"""
        <div class="hrp-card kpi-card">
            <div class="kpi-label">{title}</div>
            <div class="kpi-value">{value}</div>
            {sub_html}
        </div>
    """


def section_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="hrp-card">
            <div class="card-title">{title}</div>
            <div class="card-subtitle">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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

start_date, end_date = normalize_date_range(selected_dates)

date_series = prediction_df["date_only"]
mask = (date_series >= start_date) & (date_series <= end_date)
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
sleep_avg = sleep_proba.mean() * 100 if sleep_proba.notna().any() else None
activity_avg = activity_proba.mean() * 100 if activity_proba.notna().any() else None

st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Key performance indicators for the selected view.</div>',
    unsafe_allow_html=True,
)

date_label = (
    f"{start_date:%Y-%m-%d} → {end_date:%Y-%m-%d}" if start_date and end_date else "—"
)
kpi_html = "".join(
    [
        kpi_card("Days analyzed", _format_number(len(prediction_df))),
        kpi_card("Date range", date_label),
        kpi_card("Sleep good %", _format_percent(sleep_good_pct)),
        kpi_card("Activity good %", _format_percent(activity_good_pct)),
        kpi_card("Avg sleep proba", _format_percent(sleep_avg)),
        kpi_card("Avg activity proba", _format_percent(activity_avg)),
    ]
)
st.markdown(f'<div class="kpi-grid">{kpi_html}</div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Trends</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Sleep and activity probabilities with rolling averages.</div>',
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


if trend_df.empty or (
    trend_df["sleep_proba"].isna().all() and trend_df["activity_proba"].isna().all()
):
    section_card("Trend insights", "Trend data isn't available for the selected range.")
else:
    trend_cols = st.columns(2)
    with trend_cols[0]:
        with card("Sleep pattern", "Probability with rolling average"):
            if trend_df["sleep_proba"].notna().any():
                sleep_plot_df = trend_df.copy()
                if "sleep_rolling" not in sleep_plot_df.columns:
                    sleep_plot_df["sleep_rolling"] = sleep_plot_df["sleep_proba"]
                sleep_plot_df = sleep_plot_df.melt(
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
        with card("Activity pattern", "Probability with rolling average"):
            if trend_df["activity_proba"].notna().any():
                activity_plot_df = trend_df.copy()
                if "activity_rolling" not in activity_plot_df.columns:
                    activity_plot_df["activity_rolling"] = activity_plot_df[
                        "activity_proba"
                    ]
                activity_plot_df = activity_plot_df.melt(
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

st.markdown('<div class="section-title">Risk distribution</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Breakdown of predicted risk levels.</div>',
    unsafe_allow_html=True,
)

risk_columns = [
    "predicted_health_risk_level",
    "health_risk_level",
    "risk_level",
]
risk_column = next((col for col in risk_columns if col in prediction_df.columns), None)
if risk_column:
    risk_counts = (
        prediction_df[risk_column].fillna("Unknown").value_counts().reset_index()
    )
    risk_counts.columns = ["risk_level", "count"]
    risk_cols = st.columns(2)
    with risk_cols[0]:
        with card("Risk level share", "Donut view of risk categories"):
            pie_fig = px.pie(
                risk_counts,
                names="risk_level",
                values="count",
                hole=0.5,
            )
            pie_fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(pie_fig, use_container_width=True)
    with risk_cols[1]:
        with card("Risk level counts", "Daily predictions by category"):
            bar_fig = px.bar(
                risk_counts,
                x="risk_level",
                y="count",
                color="risk_level",
            )
            st.plotly_chart(bar_fig, use_container_width=True)
else:
    section_card("Risk distribution", "No risk level column found in predictions.")

st.markdown('<div class="section-title">KPI trends</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Steps and sleep minutes over time.</div>',
    unsafe_allow_html=True,
)

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
    section_card(
        "KPI trends",
        "Steps or sleep minutes data isn't available for the selected range.",
    )
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
    with card("Steps & sleep minutes", f"{granularity} averages"):
        trend_fig = px.line(
            trend_long,
            x="date",
            y="value",
            color="metric",
            markers=True,
        )
        st.plotly_chart(trend_fig, use_container_width=True)

st.markdown('<div class="section-title">Insights</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Behavior signals connected to quality predictions.</div>',
    unsafe_allow_html=True,
)

heatmap_summary = pd.DataFrame()
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

insights_cols = st.columns(2)
with insights_cols[0]:
    with card("Steps vs activity probability"):
        if features_df.empty:
            st.info(
                "Daily feature metrics aren't available yet. Sync wearable data to unlock deeper insights."
            )
        else:
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
                    .properties(height=320)
                )
                st.altair_chart(scatter, use_container_width=True)

with insights_cols[1]:
    with card("Label distribution", "Sleep vs activity labels"):
        label_counts = (
            pd.DataFrame(
                {
                    "sleep": prediction_df["sleep_label"].value_counts().sort_index(),
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
            .properties(height=320)
        )
        st.altair_chart(distribution, use_container_width=True)

if not heatmap_summary.empty:
    with card("Day-of-week heatmap", "Average sleep probability"):
        heatmap = (
            alt.Chart(heatmap_summary)
            .mark_rect()
            .encode(
                x=alt.X("day_of_week:N", sort=day_order, title=None),
                y=alt.Y("week:T", title="Week of"),
                color=alt.Color("avg_proba:Q", title="Avg. prob"),
                tooltip=["week:T", "day_of_week:N", "avg_proba:Q"],
            )
            .properties(height=320)
        )
        st.altair_chart(heatmap, use_container_width=True)

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
