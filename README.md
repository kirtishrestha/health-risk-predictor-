# Health Risk Predictor

This repository sets up an end-to-end health risk predictor pipeline using the Kaggle **Fit Bit Raw Datasets** (archive.zip) focused on the **bella_b** participant data. The project is structured for clean ingestion, transformation, modeling, and presentation.

## Data Source
- Single data source: Kaggle Fit Bit Raw Datasets → `bella_b` CSV exports (`dailyActivity_merged`, `sleepDay_merged`, `heartrate_seconds_merged`).

## High-Level Workflow
1. **Ingestion to Supabase**: Raw CSVs from `data/raw/bella_b` will be loaded into Supabase.
2. **PySpark ETL**: Transformations will build `daily_metrics` datasets stored in `data/processed` and Supabase.
3. **Risk Label Generation**: Derived labels will prepare data for modeling.
4. **Model Training**: Train sklearn models and save serialized `.pkl` files in `models/`.
5. **Streamlit Dashboard**: Serve predictions and monitoring via a Streamlit app.

## Getting started
1. Create and activate a virtual environment:
   - `python -m venv .venv`
   - On macOS/Linux: `source .venv/bin/activate`
   - On Windows: `.venv\\Scripts\\activate`
2. Install dependencies: `pip install -r requirements.txt`.
3. Run `sql/schema.sql` in Supabase (via the SQL editor) to create raw and daily metric tables.
4. Set `SUPABASE_DB_URL` and (optionally) `SUPABASE_SCHEMA` environment variables.
5. Load the raw Fitbit data into Supabase: `python -m src.ingestion.load_raw_fitbit`.

This PR establishes the project scaffolding; detailed logic will be added in future iterations.
