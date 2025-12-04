CREATE TABLE IF NOT EXISTS raw_fitbit_daily_activity (
    id BIGINT,
    activity_date DATE,
    total_steps INTEGER,
    total_distance DOUBLE PRECISION,
    very_active_minutes INTEGER,
    fairly_active_minutes INTEGER,
    lightly_active_minutes INTEGER,
    sedentary_minutes INTEGER,
    calories INTEGER,
    source TEXT NOT NULL DEFAULT 'fitbit_bella_b'
);

CREATE TABLE IF NOT EXISTS raw_fitbit_sleep_day (
    id BIGINT,
    sleep_date DATE,
    total_sleep_records INTEGER,
    total_minutes_asleep INTEGER,
    total_time_in_bed INTEGER,
    source TEXT NOT NULL DEFAULT 'fitbit_bella_b'
);

CREATE TABLE IF NOT EXISTS raw_fitbit_hr_seconds (
    id BIGINT,
    ts TIMESTAMP,
    heart_rate INTEGER,
    source TEXT NOT NULL DEFAULT 'fitbit_bella_b'
);

CREATE TABLE IF NOT EXISTS daily_metrics (
    id BIGINT,
    date DATE,
    total_steps INTEGER,
    total_distance DOUBLE PRECISION,
    very_active_minutes INTEGER,
    fairly_active_minutes INTEGER,
    lightly_active_minutes INTEGER,
    sedentary_minutes INTEGER,
    calories INTEGER,
    total_minutes_asleep INTEGER,
    total_time_in_bed INTEGER,
    sleep_efficiency DOUBLE PRECISION,
    avg_hr DOUBLE PRECISION,
    max_hr INTEGER,
    min_hr INTEGER,
    active_minutes INTEGER,
    source TEXT NOT NULL DEFAULT 'fitbit_bella_b',
    health_risk_level TEXT,
    cardiovascular_strain_risk TEXT,
    sleep_quality_risk TEXT,
    stress_risk TEXT,
    PRIMARY KEY (id, date)
);
