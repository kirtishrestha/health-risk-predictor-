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

This PR establishes the project scaffolding; detailed logic will be added in future iterations.
