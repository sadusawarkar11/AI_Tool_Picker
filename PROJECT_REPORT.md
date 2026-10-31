# ToolMatcher AI — Comprehensive Project Report

**Report date:** 31 October 2026  
**Project type:** Educational machine-learning recommendation web application  
**Author:** Sayli  
**Technology:** Python, scikit-learn, Flask, pandas, NumPy, HTML, CSS, JavaScript, Chart.js

## Executive summary

ToolMatcher AI is a decision-support web application that helps users compare AI tools for a task, experience level, and budget. It combines a multi-class machine-learning model with a rule-based ranking score, then returns two shortlists: tools that offer a free plan and paid tools whose listed monthly price is within the entered budget. An analytics page summarizes the dataset and the model result.

The project demonstrates an end-to-end machine-learning workflow: clean a raw dataset, encode its categorical values, train and save a classifier, load the model in a Flask service, and present recommendations in a browser. The repository includes the code, raw and cleaned data, model artifacts, notebook, launch scripts, and supporting documentation.

This report describes the current files in the repository. The cleaned data contains 339 records and 65 standardized tool labels. The included classifier is a Random Forest trained with 11 input features. The saved result file records 76.47% accuracy on a 68-record holdout. Older notebook outputs and earlier documents describe a separate experiment and include a 97.02% result; that result is not representative of the model currently included with the web application.

## 1. Project description

### 1.1 The problem

AI products cover many overlapping categories, including writing, coding, research, image creation, transcription, productivity, and data analysis. A user who needs a tool must often compare products across different pricing structures and feature descriptions. The best choice also depends on personal context: a beginner may value ease of use, while an advanced user may value quality or control; a strict budget may rule out tools with strong paid features.

Manual comparison takes time and can produce a poor match when a product is too expensive, too complex, or not suited to the task. ToolMatcher AI organizes a supplied dataset of tools and turns selected user criteria into a ranked starting list.

### 1.2 Why a solution is needed

The need for the project comes from four practical issues:

1. **Choice overload:** A large and changing tool market makes it difficult to know which products are relevant.
2. **Different user needs:** A recommendation should consider both the task and the user's level of experience.
3. **Pricing complexity:** Free plans, trials, and paid tiers make a simple “best tool” list inadequate.
4. **Scattered comparison data:** Important attributes such as price, quality, platform, and free quotas are not always presented in a consistent format.

A structured recommendation system can reduce the first round of search by filtering and ranking options against these criteria. The aim is to help users form a shortlist more quickly, not to replace a product trial or independent price check.

### 1.3 Project objectives

- Standardize inconsistent tool names, task categories, and expertise labels.
- Prepare a reusable dataset for model training and application use.
- Train a classifier to estimate which tool label is associated with a candidate feature profile.
- Combine the model's score with practical quality, efficiency, ease-of-use, and budget factors.
- Present free-plan and eligible paid choices in a simple web interface.
- Provide a small analytics dashboard to inspect the included data and recorded model accuracy.

### 1.4 Intended users and use cases

The application is intended as an educational demonstration and a first-pass discovery aid for people choosing AI software. Example use cases include:

- A beginner looking for a coding assistant with a free plan.
- A user comparing image-generation tools under a monthly budget.
- A user looking for research or transcription software for a particular task.
- A presenter demonstrating data cleaning, model training, and Flask integration.

The dataset is a snapshot, so users should verify current prices, limits, and product details with the vendors before making a purchase decision.

## 2. What the system does

### 2.1 User inputs

The recommendation page collects:

- A free-text requirement.
- Task category.
- Expertise level: Beginner, Intermediate, or Advanced.
- Monthly budget.
- Platform preference.

The page displays tool information available in the dataset, such as monthly price, yearly price, free-plan availability, free quota, trial length, quality rating, efficiency rating, ease-of-use rating, platform, and underlying model.

### 2.2 Recommendation flow

1. The Flask server loads the cleaned CSV, classifier, scaler, and label encoders.
2. The app filters records by the selected task category and expertise level.
3. If no exact rows exist, it broadens the candidates to the chosen category, then to the full cleaned dataset.
4. For each candidate row, the app builds the 11-feature vector using the user's budget and the candidate's recorded attributes.
5. It scales that vector, obtains the classifier's class probabilities, and looks up the probability associated with the candidate tool's encoded label.
6. It combines this probability with non-ML ratings and a budget-fit multiplier.
7. It retains the highest-scoring row for each tool name to avoid duplicate tool entries.
8. It returns up to ten free-plan recommendations and up to ten paid tools whose listed monthly price is positive and within the budget.

The free-text requirement currently does not affect filtering, features, or scoring. The selected platform also does not filter the candidates; the platform recorded on each candidate row is used in its feature vector. These are current implementation limits.

### 2.3 Ranking formula

The app currently calculates a combined score as follows:

    score =
        (tool_probability × 40
         + efficiency_rate_pct × 0.3
         + output_quality_1_10 × 2
         + ease_of_use_1_5 × 2)
        × budget_match

where budget_match is 1.0 when the candidate's monthly price is no greater than the user's budget, and 0.5 otherwise.

The free-plan list is sorted by this score and then limited to ten results. The paid list is restricted to positive monthly prices within budget and sorted by the same score. This is a hand-written ranking heuristic layered on top of the classifier; its weights have not been tuned from user feedback.

## 3. System architecture

### 3.1 Data and training components

- **toolmatcher_ai_raw_dataset_v2.csv:** Original 3,425-row, 24-column dataset.
- **data_cleaning_pipeline.py:** Standardizes labels, removes repeated rows/combinations, converts numeric and boolean columns, fills selected missing values, checks ranges, and writes the cleaned data and report.
- **cleaned_toolmatcher_dataset.csv:** The 339-row data file used by the application and quick-training script.
- **quick_model_training.py:** Trains a Random Forest and a small stacking alternative, evaluates both on a holdout, saves the stronger candidate and writes model accuracy results.
- **advanced_model_training.py:** A separate, slower workflow that compares and tunes additional algorithms.
- **toolmatcher_ai_advanced.ipynb:** A saved exploratory notebook from an earlier experiment; its embedded outputs do not describe the current model artifact.

### 3.2 Web application components

- **app.py:** Flask routes, model loading, recommendation processing, and analytics data.
- **start_app.py:** Loads the runtime artifacts and starts the application without Flask debug mode.
- **templates/index.html:** Recommendation form and results interface.
- **templates/analytics.html:** Charts and model-accuracy display.
- **best_model.pkl:** Serialized current classifier.
- **scaler.pkl:** Feature scaler fitted during training.
- **label_encoders.pkl:** Encoders used to transform categories and tool names.
- **model_performance_results.csv:** Current saved accuracy record.

The app is a single-process Flask demonstration. The main launcher binds to all network interfaces on port 5000. Use an appropriate WSGI server and configure network exposure before deploying beyond a controlled local demonstration.

## 4. Dataset and data preparation

### 4.1 Raw data

The raw file has 3,425 rows and 24 columns. It includes 95 distinct tool names, 34 task-category spellings, and 12 expertise-level spellings. It also contains 3,473 missing cells, with context-window values accounting for most of the missing data.

Each row represents a candidate tool record associated with a requirement and includes fields such as task category, expertise level, budget, tool name, price, free-plan availability, ratings, platform, and source-verification metadata.

### 4.2 Cleaning steps

The cleaning script carries out these operations:

1. Loads the raw CSV.
2. Standardizes expertise variations to Beginner, Intermediate, and Advanced.
3. Normalizes tool-name spelling and known aliases.
4. Standardizes task-category variants.
5. Removes exact duplicates.
6. Removes repeated tool/category/expertise combinations, keeping the row with the strongest efficiency value.
7. Converts price and rating fields into numeric types and normalizes boolean flags.
8. Fills selected missing fields. For example, missing monthly price is set to zero by the current script.
9. Removes records outside expected ranges for efficiency, quality, and ease-of-use scores.
10. Saves the cleaned CSV and a cleaning summary.

### 4.3 Cleaned data snapshot

The included cleaned dataset has:

- 339 rows and 24 columns.
- 65 distinct standardized tool labels.
- 26 task categories.
- 3 expertise levels.
- No exact duplicate rows or repeated tool/category/expertise combinations.
- 310 remaining missing cells, of which 231 are context-window values.

The cleaning pipeline reports 3,081 duplicate or repeated combinations removed, five out-of-range records removed, and 68 values filled. It is important to distinguish normalization and deduplication from complete imputation: the final CSV still has missing values. Treating an unknown monthly price as zero can also make “unknown” look like “free,” so that cleaning rule deserves review.

The dataset includes price_source_status and last_verified fields, but the app does not refresh its data automatically. Product details and prices can become stale.

## 5. Detailed machine-learning methodology

### 5.1 Learning task

This is a **multi-class classification** task. The model receives a numeric feature vector describing a user budget and a candidate AI tool record. It is trained to predict the encoded tool-name class associated with that row.

The classifier is not a natural-language model. It does not interpret the free-text requirement. It learns associations in the structured training rows between their encoded category/experience/platform/model fields, prices, and ratings and the tool-name label.

### 5.2 Input features

The training script uses the following 11 features in this exact order:

| # | Feature | Type | Meaning |
| ---: | --- | --- | --- |
| 1 | expertise_level_Encoded | Encoded category | User expertise level |
| 2 | task_category_Encoded | Encoded category | Task the tool is listed for |
| 3 | platform_Encoded | Encoded category | Candidate's recorded platform |
| 4 | best_model_Encoded | Encoded category | Underlying model listed for the tool |
| 5 | monthly_budget_usd | Numeric | User's monthly budget |
| 6 | monthly_price_usd | Numeric | Candidate tool's listed monthly price |
| 7 | ease_of_use_1_5 | Numeric | Ease-of-use rating |
| 8 | output_quality_1_10 | Numeric | Output-quality rating |
| 9 | efficiency_rate_pct | Numeric | Efficiency rating |
| 10 | free_plan_available | Binary | Whether a free plan is recorded |
| 11 | annual_billing_available | Binary | Whether annual billing is recorded |

The target is tool_name_Encoded, a numeric label representing the tool name. The quick-training script encodes expertise, category, platform, underlying model, and tool name with scikit-learn LabelEncoder objects, which are saved for application use.

### 5.3 Preprocessing and data split

The quick-training workflow performs these steps:

1. Reads the cleaned dataset.
2. Encodes the categorical input columns and target labels.
3. Normalizes yes/no and numeric/string representations of the two billing flags.
4. Selects the 11 features and encoded tool-name target.
5. Creates an 80/20 random train/test split with random_state=42.
6. Fits StandardScaler on the training features only and transforms both splits.
7. Trains model candidates on the training split.
8. Calculates accuracy on the held-out test split.
9. Saves the selected classifier, scaler, label encoders, and accuracy results.

For 339 rows, the split contains 271 training records and 68 test records. The split is not stratified. Because the data has many tool labels and some labels occur rarely, not every tool is guaranteed to appear in the training portion.

Label encoding assigns integer IDs to category values. This is convenient for the current scripts, but those integers can imply an ordering that the categories do not truly have. One-hot encoding or a model pipeline that handles categorical variables directly would be a useful future comparison.

StandardScaler transforms each feature using the training mean and standard deviation:

    scaled_value = (value - training_mean) / training_standard_deviation

Scaling makes numeric ranges more comparable and is useful for several candidate algorithms. A Random Forest does not generally require standardized features, but the scaler is kept in the pipeline so training and inference use the same feature-processing steps.

### 5.4 Current model: Random Forest

The saved best_model.pkl loads as a scikit-learn RandomForestClassifier. The quick-training script configures its Random Forest candidate with:

| Setting | Value | Purpose |
| --- | ---: | --- |
| Number of trees | 100 | Average decisions from multiple trees |
| Maximum tree depth | 20 | Limit tree complexity |
| Features considered per split | sqrt | Randomly consider a subset of features at each split |
| Minimum samples to split | 5 | Avoid splitting very small nodes |
| Random seed | 42 | Make the random process repeatable |
| Parallel jobs | All available cores | Speed up fitting |

A Random Forest combines many decision trees. Each tree learns a series of feature-based splits from a bootstrap sample of the training data. At each split, only a random subset of features is considered. For classification, tree outputs are aggregated to select a class; the model's predict_proba method averages probability estimates across trees.

The ensemble is useful for this demonstration because it can model non-linear interactions between attributes, handles numeric features without assuming a straight-line relationship, and offers a practical baseline for a small structured dataset. It is still constrained by the dataset's size and label imbalance.

### 5.5 Alternative model considered

The quick-training script also evaluates a small StackingClassifier. In this version, the base layer contains a Random Forest and a Logistic Regression meta-estimator combines its out-of-fold outputs using two-fold internal cross-validation. The script compares its held-out accuracy against the standalone Random Forest and saves whichever scores higher.

The currently included artifact is the standalone Random Forest, so the stacking alternative was not selected for this repository snapshot. The current performance CSV records the selected Random Forest result only.

The advanced training script and historical notebook contain additional experiments. Their model outputs are from separate workflows and should not be mixed with the performance result for the committed artifact.

### 5.6 Model evaluation

The current recorded score is:

| Metric | Current value |
| --- | ---: |
| Held-out accuracy | 76.47% |
| Test records | 68 |
| Saved model | Random Forest |

Accuracy is the fraction of held-out examples for which the predicted tool class matches the recorded target. With 68 test rows, a score near 76.47% corresponds to roughly 52 correct predictions out of 68.

The result is a single holdout score, not an independent benchmark or a guarantee of user satisfaction. The current performance file does not provide precision, recall, F1-score, per-class support, a confidence interval, or probability-calibration results.

The encoder contains 65 tool labels, while the fitted Random Forest exposes 63 classes. This is consistent with rare labels being absent from the unstratified training split. The application maps probabilities by the classifier's actual class IDs; a tool label missing from model.classes_ receives zero model confidence. Its non-ML ratings can still contribute to the ranking score.

## 6. From model output to displayed recommendation

The classifier probability is only one part of the displayed ranking. The application combines the probability associated with a candidate tool and its recorded ratings, then applies a budget multiplier. It deduplicates candidates by tool name, keeping the highest-scoring candidate row, and produces the two result lists.

This hybrid design is intended to balance the learned signal with criteria that are directly understandable to users. However, the score weights are hand-selected and have not been validated against clicks, purchases, satisfaction, or expert judgments. The probability should not be read as a calibrated probability that a user will like or choose the tool.

## 7. Web interface and API

### 7.1 API endpoints

| Endpoint | Method | Purpose |
| --- | --- | --- |
| / | GET | Serves the recommendation page |
| /analytics | GET | Serves the analytics dashboard |
| /get_categories | GET | Returns task categories, expertise levels, and platforms |
| /predict | POST | Returns ranked free-plan and paid recommendation lists |
| /get_analytics_data | GET | Returns dataset summaries and recorded model accuracy |

The /predict endpoint accepts JSON fields named user_requirement, expertise_level, task_category, monthly_budget, and platform. The response contains recommendation lists, counts, number of records processed, a deduplication flag, and the recorded accuracy value.

### 7.2 Analytics

The dashboard summarizes the included data with category counts, popular tool labels, free-plan availability, average efficiency/quality/ease/price, platform distribution, and model accuracy. The current performance CSV records accuracy only, so the dashboard does not claim unrecorded precision, recall, or F1 metrics.

## 8. Local setup and operation

Use a Python environment compatible with the pinned packages in requirements.txt. From the project directory:

    python -m venv .venv
    .venv\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    python start_app.py

Open http://localhost:5000 for recommendations and http://localhost:5000/analytics for the dashboard.

To rebuild the current data and model:

    python data_cleaning_pipeline.py
    python quick_model_training.py

The advanced script offers a slower comparison and tuning route. Training scripts write artifacts into the project directory and can replace the model, encoders, scaler, and metrics used by the app.

## 9. Limitations and interpretation

- The free-text requirement does not currently affect candidate selection or model features.
- The requested platform does not filter results; the candidate record's platform is used as an input.
- The cleaned dataset contains only 339 rows for 65 tool labels, and the fitted model includes 63 of those labels.
- The random split is unstratified. The 68-row holdout is small, so accuracy may vary with another split.
- Accuracy does not show which tools are being confused, whether rare classes work well, or whether probability estimates are calibrated.
- Label-encoded nominal categories can introduce artificial numeric ordering.
- The hybrid score is heuristic and is not validated against observed user outcomes.
- Some price and tool data may be outdated; several fields remain missing.
- Missing monthly prices are converted to zero by the current cleaner, which can confuse unknown prices with free tools.
- Flask's built-in development server is for local demonstration. Use production hosting practices before exposing the service publicly.
- The repository does not include a LICENSE file. Check rights and data-source terms before redistributing the project or dataset.

## 10. Recommended next steps

1. Collect more labeled examples, especially for infrequent tool labels.
2. Compare label encoding with one-hot encoding or categorical-aware pipelines.
3. Use stratified or repeated evaluation where class counts permit; report per-class precision, recall, F1, and a confusion matrix.
4. Keep a fixed final test set and avoid choosing model settings based on that same test set.
5. Check probability calibration before describing classifier probabilities as confidence.
6. Use the free-text requirement through a documented text-matching or embedding method, or remove it from the active recommendation form until implemented.
7. Apply the user's platform preference in candidate filtering and test budget handling with edge cases.
8. Validate the ranking weights with user feedback and measurable outcomes.
9. Add input validation, automated tests, a production WSGI configuration, and a defined project license.
10. Record data sources and refresh dates and avoid treating missing prices as confirmed free plans.

## Conclusion

ToolMatcher AI addresses a real discovery problem: choosing among many AI tools while balancing task fit, expertise, and cost. It provides a structured shortlist and demonstrates a complete ML-enabled web workflow from data preparation through model serving.

The current implementation is best understood as a prototype and educational decision-support system. Its included Random Forest records 76.47% accuracy on a small 68-row holdout. The most valuable improvements are to gather more balanced training examples, evaluate the classifier more thoroughly, make the free-text and platform inputs affect recommendations, and validate the ranking formula with actual user feedback.
