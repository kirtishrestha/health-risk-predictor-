# Fitbit Health Risk Predictor

An end-to-end data engineering and machine learning project that ingests Fitbit wearable data, generates daily health risk predictions, and visualizes insights through an interactive Streamlit dashboard backed by Supabase.

---

## Overview

This project demonstrates how raw smartwatch data can be transformed into meaningful health insights using a complete pipeline:

- Fitbit ZIP data ingestion
- Data cleaning and daily aggregation (ETL)
- Machine learning model training and inference
- Centralized storage using Supabase
- Interactive analytics dashboard using Streamlit

The system focuses on **probability-based predictions** and **visual explainability**, not medical diagnosis.

---

## Tech Stack

- **Python** – ETL, feature engineering, ML
- **Supabase (PostgreSQL)** – data storage
- **Scikit-learn** – classification models
- **Streamlit** – interactive dashboard
- **Pickle** – model persistence

---

## How It Works

1. Upload Fitbit ZIP data in **Pipeline Runner**
2. Run ETL to standardize and store daily metrics
3. Train ML models on labeled daily data
4. Run inference to generate probabilities and labels
5. Explore predictions and trends in **Analytics Dashboard**

---

## Models & Predictions

- Sleep quality classifier
- Activity quality classifier
- Probability scores (0–1) with derived labels
- Rolling averages and risk bucket thresholds

Models are evaluated based on **stability, interpretability, and probability outputs**, not raw accuracy alone.

---

## Dashboard Features

- Daily / weekly / monthly views
- Prediction KPIs and trends
- Risk bucket distributions
- Behavior vs prediction analysis
- Histograms, box plots, heatmaps

Advanced visualizations (Sankey, maps, treemap) are **commented out** when data is unavailable and can be re-enabled later.

---

## Limitations

- Fitbit data only
- Single-user focused
- No clinical ground-truth labels
- Not a medical or diagnostic system

---

## Future Improvements

- Multi-user support
- Additional wearable data sources
- Time-series models
- Model comparison and explainability
- Personalized risk insights

---

## Run the App

```bash
pip install -r requirements.txt
streamlit run src/app/streamlit_app.py


## Disclaimer

This project is for educational and research purposes only and does not provide medical advice.

---
