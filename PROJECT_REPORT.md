# ToolMatcher AI — Comprehensive Project Report

**Report date:** 31 October 2026  
**Project type:** Educational machine-learning recommendation web application  
**Implementation:** Python, scikit-learn, Flask, pandas, NumPy, HTML/CSS/JavaScript

## 1. Executive summary

ToolMatcher AI is a web application for narrowing down a collection of AI tools using a selected task category, expertise level, and monthly budget. It combines a trained multi-class classifier with a hand-written ranking score, returns separate lists for tools with free plans and paid tools within budget, and provides a dashboard of dataset summaries.

The repository includes the source datasets, a cleaned dataset, training scripts, model and preprocessing artifacts, a Flask application, presentation materials, and a saved notebook. The current cleaned dataset contains 339 records representing 65 standardized tool labels. The committed classifier is a Random Forest fitted on 11 features. The current result file records 76.47% accuracy on a 68-row held-out split.

Some older documents and notebook outputs describe earlier experiments and quote a 97.02% result. That figure does not match the committed model, which is a Random Forest, nor the current results CSV. This report treats the checked-in model and current CSV as the current snapshot.

## 2. Problem and intended users

The project addresses the difficulty of comparing many AI products when a user has a particular task, experience level, and spending limit. It is designed as a demonstration and learning project for users who want an initial shortlist rather than a definitive purchasing decision.

The interface presents tool details such as monthly price, available free plan, free quota, trial length, quality, efficiency, ease of use, underlying model, and platform where the dataset supplies those values.

## 3. User experience and recommendation flow

The home page collects five fields: free-text requirement, task category, expertise level, monthly budget, and platform.

The server filters dataset records by the selected task category and expertise level. If there are no exact matches, it broadens the candidate set to the category and then to all cleaned records. For each candidate it prepares the model features, obtains a probability for that candidate tool when that class exists in the fitted model, and calculates a combined ranking score.

The score currently used in app.py is:

    (confidence × 40
     + efficiency_rate_pct × 0.3
     + output_quality_1_10 × 2
     + ease_of_use_1_5 × 2)
    × budget_match

Budget match is 1.0 when the listed monthly price is within the entered budget and 0.5 otherwise. The free list includes candidates with a free plan; the paid list includes candidates with a positive monthly price no greater than the budget. Each list is capped at ten results and ordered by the combined score.

The analytics page summarizes the cleaned dataset: leading categories and tools, free-plan availability, average quality/efficiency/ease/price values, platform distribution, and model accuracy results.

## 4. System architecture

### Data preparation

- data_cleaning_pipeline.py loads the raw CSV, standardizes expertise and category labels, normalizes tool names, removes repeated tool/category/expertise combinations, converts numeric and boolean fields, handles selected missing values, removes out-of-range records, and writes the cleaned CSV and a text report.
- quick_model_training.py encodes categorical columns, selects 11 features, splits the cleaned records into train and test sets, scales the features, compares a Random Forest with a small stacking classifier, and saves the stronger model and preprocessing objects.
- advanced_model_training.py provides a separate, slower comparison and tuning route.
- toolmatcher_ai_advanced.ipynb records an earlier exploratory training run. Its embedded outputs should be read as historical experiment results, not as results for the current model artifact.

### Application

- app.py loads the cleaned dataset, serialized classifier, scaler, label encoders, and recorded performance file. It serves the API and recommendation pages.
- start_app.py loads the same resources and starts Flask without debug mode.
- templates/index.html contains the recommendation interface.
- templates/analytics.html renders the analytics dashboard and model accuracy comparison.

The application is a single-process Flask demonstration. The main launcher binds to all network interfaces on port 5000; it is suitable for a controlled demo and needs an appropriate deployment setup before public hosting.

## 5. Data inventory and cleaning

The raw file toolmatcher_ai_raw_dataset_v2.csv has 3,425 rows and 24 columns. It contains 95 distinct tool names, 34 category spellings, and 12 expertise spellings. It also has 3,473 missing cells, with context_window_tokens accounting for most of them.

The checked-in cleaned file cleaned_toolmatcher_dataset.csv has 339 rows and 24 columns, with 65 standardized tool names, 26 task categories, and the three expertise levels Beginner, Intermediate, and Advanced. There are no exact duplicate rows or repeated tool/category/expertise combinations in that file.

The recorded cleaning process removed 3,081 duplicate or repeated combinations and five out-of-range records. It standardized variants of expertise levels, task categories, and tool names. It reports 68 filled values, but the final cleaned CSV still contains 310 missing cells: 231 in context_window_tokens, with the remainder in monthly budget, free trial, yearly price, standard-plan price, and rank fields. Therefore “cleaned” means normalized and deduplicated for this workflow; it does not mean every cell is complete.

The dataset includes price_source_status and last_verified fields, so price freshness can vary. The app uses the supplied snapshot and has no live data refresh.

## 6. Machine-learning method and current evaluation

### Feature set

The current training workflow uses these 11 features, in order:

1. Encoded expertise level
2. Encoded task category
3. Encoded platform
4. Encoded underlying model
5. Monthly user budget
6. Monthly tool price
7. Ease-of-use rating
8. Output-quality rating
9. Efficiency percentage
10. Free-plan flag
11. Annual-billing flag

The target is the encoded tool name. A StandardScaler is fit on the training portion and saved for inference. Label encoders for input categories and tool names are also saved.

### Current committed artifact

The included best_model.pkl loads as a scikit-learn RandomForestClassifier. The quick-training script uses 100 trees, maximum depth 20, square-root feature selection, minimum split size 5, and random seed 42. It creates an 80/20 random split of 339 records: 271 training rows and 68 test rows. The recorded held-out accuracy is 0.7647058824, or 76.47%.

The label encoder contains 65 tool labels, while the fitted model exposes 63 classes. This follows from the unstratified random split and sparse classes. Probability lookup now uses the model's actual class IDs; a label absent from model.classes_ receives zero confidence instead of an invented fallback value.

Only accuracy is available in the committed model_performance_results.csv. Precision, recall, per-class metrics, confidence intervals, and an independent validation result are not included, so this report does not claim them.

### Historical results

ML_TRAINING_REPORT.md and the notebook contain earlier metrics, including 97.02%. They describe a different training run and data split. They are not supported by the current serialized model or result CSV and should not be used as the headline performance claim for this snapshot.

## 7. Software stack and repository artifacts

- Backend: Python and Flask
- Data work: pandas and NumPy
- Modeling and preprocessing: scikit-learn
- Serialization: joblib/pickle
- UI: HTML, CSS, JavaScript, Chart.js
- Experimentation: Jupyter Notebook

The runtime artifacts are included intentionally: best_model.pkl, scaler.pkl, label_encoders.pkl, and cleaned_toolmatcher_dataset.csv are required by the application. They are small enough for ordinary GitHub storage. The raw dataset, notebook, source scripts, templates, reports, and Windows launchers are also included. Python caches and Jupyter checkpoint folders are excluded.

## 8. Local setup and operation

Install a Python environment compatible with requirements.txt, then from the project root run:

    python -m venv .venv
    .venv\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    python start_app.py

Visit http://localhost:5000 for the main app and http://localhost:5000/analytics for the dashboard.

To rebuild the current data and model:

    python data_cleaning_pipeline.py
    python quick_model_training.py

The advanced script offers additional tuning but is slower. Training writes artifacts into the project root and replaces the committed versions.

## 9. API surface

| Endpoint | Method | Function |
| --- | --- | --- |
| / | GET | Serves the recommendation page |
| /analytics | GET | Serves the dashboard |
| /get_categories | GET | Returns available categories, expertise levels, and platforms |
| /predict | POST | Returns ranked free-plan and paid tool lists |
| /get_analytics_data | GET | Returns summary metrics and model performance rows |

The prediction request accepts user_requirement, expertise_level, task_category, monthly_budget, and platform as JSON fields. The response includes the two recommendation lists, counts, records processed, whether deduplication was applied, and the recorded accuracy value.

## 10. Limitations and risks to interpretation

- The free-text requirement is currently logged but not used in candidate filtering or model features.
- The request's platform does not filter the candidate records; the candidate row's platform is used as a model feature.
- The training dataset is small relative to 65 tool labels, and the random split is not stratified. Two tool labels are absent from the fitted model classes.
- A single 68-row test split can give a noisy accuracy estimate. Accuracy alone does not describe performance for each class or whether probabilities are calibrated.
- The ranking score is a heuristic whose weights have not been validated against user satisfaction or a business objective.
- Dataset prices and product characteristics may be stale, and several fields remain missing.
- The local development server is not a production deployment. Use a production WSGI server, configure the host and environment appropriately, and review exposure before deployment.
- The repository has no LICENSE file. Source and dataset redistribution rights should be checked before reuse.

## 11. Suggested next work

1. Add stratified evaluation or repeated cross-validation where class counts allow, and publish class-wise metrics.
2. Gather more labeled examples for rare tools and include a stable train/test evaluation protocol.
3. Either use the free-text request in a documented matching method or remove it from the interface until implemented.
4. Apply the requested platform and budget consistently in filtering and scoring.
5. Validate the ranking formula with user feedback, then tune its weights from measurable outcomes.
6. Add automated tests, input validation, and a production WSGI/deployment configuration.
7. Record data source and refresh information for every price field, and define the project license.

## 12. Summary

ToolMatcher AI demonstrates an end-to-end path from raw data cleaning and model training to Flask-based recommendations and dashboard reporting. The committed snapshot is reproducible from the included artifacts and reports a 76.47% held-out accuracy for its Random Forest model. Its strongest next step is improving evaluation and aligning the free-text/platform inputs with the behavior users expect from the interface.
