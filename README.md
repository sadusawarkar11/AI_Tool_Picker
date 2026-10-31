# ToolMatcher AI

A Flask and machine-learning project that helps people shortlist AI tools by task category, expertise level, and budget. It presents free-plan tools and paid tools within budget, with an analytics dashboard for the included dataset.

## Current project snapshot

These figures describe the files included in this repository:

- Raw dataset: 3,425 rows, 24 columns, 95 tool names, 34 task categories, and 12 expertise-level spellings.
- Cleaned dataset: 339 rows, 24 columns, 65 standardized tool names, 26 task categories, and 3 expertise levels.
- Saved classifier: Random Forest, 11 input features, 63 classes represented in the fitted model.
- Recorded held-out accuracy: 76.47% on the quick-training script's 68-row test split.
- Web application: Flask routes for recommendations and analytics, with HTML, CSS, JavaScript, and Chart.js templates.

The project also contains older training notes and a notebook from an earlier experiment. Their 97.02% result is not the metric for the model artifact currently included. See PROJECT_REPORT.md and ML_TRAINING_REPORT.md for the current figures and limitations.

## Run the application

Use a Python interpreter compatible with the pinned dependencies in requirements.txt.

    python -m venv .venv
    .venv\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    python start_app.py

Open http://localhost:5000 for recommendations or http://localhost:5000/analytics for the dashboard. The application expects to be started from the project directory. The included cleaned dataset and model files let you run the app without retraining.

On Windows, START_SERVER.bat starts the Flask application. RUN_NOTEBOOK.bat opens the saved Jupyter notebook. PRESENTATION_SETUP.bat starts the app and notebook in separate windows.

## Train the included model again

The quick training path uses the cleaned CSV and writes best_model.pkl, scaler.pkl, label_encoders.pkl, and model_performance_results.csv.

    python data_cleaning_pipeline.py
    python quick_model_training.py
    python start_app.py

The advanced training script compares additional estimators and performs hyperparameter search; it can take longer and replaces the same model and performance files.

    python advanced_model_training.py

## What the application does

1. The home page collects a task category, expertise level, budget, platform, and a free-text requirement.
2. The server filters the cleaned records by task category and expertise level, then falls back to a broader candidate set if necessary.
3. For each candidate record, the saved classifier estimates a tool class probability. The app combines that score with efficiency, quality, ease-of-use, and a budget-fit factor.
4. The response includes up to ten tools with a free plan and up to ten paid tools whose listed monthly price is within budget.
5. The analytics page summarizes category and tool counts, free-plan availability, average ratings, and saved model accuracy results.

The free-text requirement is currently accepted by the API but does not affect filtering or scoring. Platform is used as a feature for a candidate record, but the request's platform value does not filter the candidate list.

## Project contents

- app.py and start_app.py: Flask routes, model loading, and server entry point.
- templates/: recommendation and analytics pages.
- data_cleaning_pipeline.py: data normalization, duplicate removal, numeric conversion, and report generation.
- quick_model_training.py: fast training path used for the current saved classifier.
- advanced_model_training.py: alternate model comparison and tuning workflow.
- toolmatcher_ai_advanced.ipynb: saved exploratory training notebook. Its outputs include an earlier experiment.
- toolmatcher_ai_raw_dataset_v2.csv: source dataset.
- cleaned_toolmatcher_dataset.csv: current application dataset.
- best_model.pkl, scaler.pkl, label_encoders.pkl: model and preprocessing artifacts required by the app.
- model_performance_results.csv: recorded model accuracy results.
- PROJECT_REPORT.md: full project and methodology report.
- ML_TRAINING_REPORT.md: current training details and evaluation caveats.
- ALGORITHMS_EXPLAINED.md and QUICK_ALGORITHM_REFERENCE.md: learning and presentation notes.

## API routes

| Route | Method | Purpose |
| --- | --- | --- |
| / | GET | Recommendation form |
| /analytics | GET | Analytics dashboard |
| /get_categories | GET | Categories, expertise levels, and platforms |
| /predict | POST | Ranked free-plan and paid recommendations |
| /get_analytics_data | GET | Dashboard data and model results |

The prediction endpoint accepts JSON fields named user_requirement, expertise_level, task_category, monthly_budget, and platform.

## Important limitations

- The test split is small and not stratified. The saved Random Forest has 63 output classes while the encoder and cleaned dataset contain 65 tool labels. Tools not represented among the fitted classes cannot receive learned probabilities.
- The reported 76.47% is a single held-out accuracy score, not a confidence interval or independent benchmark. The included result file contains accuracy only.
- Recommendations depend on dataset fields and a hand-written ranking formula. The application does not fetch live tool features or prices.
- Some fields remain missing in the cleaned dataset, especially context-window values. The application uses defaults for several missing numeric values.
- The Flask development server is configured for local demonstrations. Use a production WSGI server and review network exposure before deploying.
- The repository does not include a LICENSE file. Check rights and source terms before redistributing the dataset or project.

## Documentation

- Comprehensive project report: PROJECT_REPORT.md
- Current ML training report: ML_TRAINING_REPORT.md
- Operational guide: PRODUCTION_READY.md
- Notebook instructions: HOW_TO_RUN_NOTEBOOK.md

## Author

Sayli
