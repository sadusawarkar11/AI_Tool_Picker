# ML Training Report — Current Repository Snapshot

## Summary

The application currently ships a Random Forest model trained from the 339-row cleaned dataset. The recorded test accuracy is 76.47% (0.7647058824) on a 68-row holdout. This value comes from model_performance_results.csv and the quick-training workflow.

This report replaces earlier claims that the deployed model achieves 97.02%. The saved model artifact is a RandomForestClassifier, and the current performance file does not contain that earlier result.

## Data and target

- Raw dataset: 3,425 rows and 24 columns.
- Cleaned dataset: 339 rows and 24 columns.
- Standardized tool labels in the cleaned data: 65.
- Task categories: 26.
- Expertise levels: Beginner, Intermediate, Advanced.
- Target: encoded tool_name.

The cleaning pipeline normalizes labels and removes duplicate tool/category/expertise combinations. The final cleaned dataset still has 310 missing cells, mainly context-window values.

## Training workflow

quick_model_training.py encodes expertise_level, task_category, platform, best_model, and tool_name. It uses 11 model features: four encoded categorical features, budget, price, ease, quality, efficiency, free-plan availability, and annual-billing availability.

The workflow creates an unstratified random 80/20 split with random_state 42: 271 training records and 68 test records. StandardScaler is fit on the training data and applied to the holdout.

The Random Forest candidate uses 100 estimators, max_depth 20, max_features sqrt, min_samples_split 5, and random_state 42. A small stacking classifier is also evaluated. The better holdout model is saved to best_model.pkl; the currently included artifact is RandomForestClassifier.

## Recorded result

| Model | Accuracy |
| --- | ---: |
| Random Forest | 76.47% |

The result file records accuracy only. It does not include precision, recall, F1-score, per-class support, a confidence interval, or results from an independent dataset.

The encoder has 65 tool labels while the trained classifier has 63 output classes. Sparse labels can be absent from the training fold because the split is not stratified. The application maps model probabilities using the classifier's actual class IDs and assigns zero confidence to a tool class not represented by the fitted model.

## Reproduce training

From the project directory, run:

    python data_cleaning_pipeline.py
    python quick_model_training.py

This rewrites the cleaned CSV, model artifacts, encoders, scaler, and accuracy CSV. advanced_model_training.py is an optional slower comparison and tuning path; it also writes model artifacts into the project directory.

## Historical notebook results

The notebook and previous versions of this report include results from earlier experiments, including a 97.02% stacking-ensemble figure. Those are historical outputs from a different data/model run and do not describe the artifact included in this repository snapshot. Use the current CSV and serialized model as the reference for the shipped application.

## Interpretation

A single 68-record holdout is small and may not represent all 65 labels. The accuracy number is useful as a project snapshot, not as a broad guarantee of recommendation quality. Further evaluation should use a stratified or repeated protocol where supported by class counts, class-wise metrics, calibration checks, and additional labeled records.
