CREATE TABLE IF NOT EXISTS daily_sleep (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    raw_user_id TEXT NULL,
    date DATE NOT NULL,
    source TEXT NOT NULL DEFAULT 'fitbit',
    sleep_minutes INTEGER NOT NULL,
    time_in_bed_minutes INTEGER,
    awakenings_count INTEGER,
    sleep_efficiency DOUBLE PRECISION,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    UNIQUE (user_id, date, source)
);

CREATE TABLE IF NOT EXISTS daily_activity (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    raw_user_id TEXT NULL,
    date DATE NOT NULL,
    source TEXT NOT NULL DEFAULT 'fitbit',
    steps INTEGER NOT NULL,
    distance_km DOUBLE PRECISION NOT NULL,
    active_minutes INTEGER,
    calories DOUBLE PRECISION,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    UNIQUE (user_id, date, source)
);

CREATE TABLE IF NOT EXISTS daily_features (
    user_id TEXT NOT NULL,
    raw_user_id TEXT NULL,
    date DATE NOT NULL,
    source TEXT NOT NULL DEFAULT 'fitbit',
    sleep_minutes INTEGER,
    steps INTEGER,
    distance_km DOUBLE PRECISION,
    active_minutes INTEGER,
    calories DOUBLE PRECISION,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    UNIQUE (user_id, date, source)
);

CREATE TABLE IF NOT EXISTS monthly_metrics (
    user_id TEXT NOT NULL,
    month DATE NOT NULL,
    source TEXT NOT NULL DEFAULT 'fitbit',
    avg_sleep_minutes DOUBLE PRECISION,
    avg_steps DOUBLE PRECISION,
    avg_distance_km DOUBLE PRECISION,
    avg_active_minutes DOUBLE PRECISION,
    total_steps DOUBLE PRECISION,
    total_distance_km DOUBLE PRECISION,
    total_active_minutes DOUBLE PRECISION,
    sleep_days_count INTEGER,
    activity_days_count INTEGER,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    UNIQUE (user_id, month, source)
);

-- Daily predictions for model inference
CREATE TABLE IF NOT EXISTS public.daily_predictions (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    date DATE NOT NULL,
    source TEXT NOT NULL,
    sleep_quality_label INT,
    sleep_quality_proba DOUBLE PRECISION,
    activity_quality_label INT,
    activity_quality_proba DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, date, source)
);

CREATE INDEX IF NOT EXISTS daily_predictions_user_date_idx
    ON public.daily_predictions (user_id, date);
