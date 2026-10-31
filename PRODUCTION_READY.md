# Operational Guide

## Current state

The repository contains a local demonstration application, the cleaned data it reads, and the model artifacts it loads. The saved model is a Random Forest with a recorded holdout accuracy of 76.47%. See PROJECT_REPORT.md for architecture, data, evaluation, and limitations.

## Run locally

    python -m venv .venv
    .venv\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    python start_app.py

Open http://localhost:5000. The analytics page is at http://localhost:5000/analytics.

Required runtime files in the project directory:

- cleaned_toolmatcher_dataset.csv
- best_model.pkl
- scaler.pkl
- label_encoders.pkl
- model_performance_results.csv

These files are included so a fresh clone can start without running training.

## Rebuild the model

    python data_cleaning_pipeline.py
    python quick_model_training.py

The advanced_model_training.py script runs a slower model comparison and tuning workflow. Any training route overwrites artifacts in the project directory.

## Deployment note

The included start script is intended for local demonstration. It starts Flask on port 5000 and binds to all network interfaces. For deployment, use a production WSGI server, configure network access deliberately, and review the limitations in PROJECT_REPORT.md.

## Data and model notes

The cleaned CSV has 339 rows and 65 tool labels. The included classifier exposes 63 classes because the unstratified holdout split left two labels without training examples. The final data still contains missing values, mainly context-window values. The data snapshot is not refreshed from vendors automatically.
