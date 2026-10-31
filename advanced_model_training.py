"""
Advanced ML Pipeline for ToolMatcher AI
Professional implementation with hyperparameter tuning, cross-validation, and ensemble methods
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TOOLMATCHER AI - ADVANCED MODEL TRAINING")
print("="*80)

# Load dataset
print("\n[1/9] Loading dataset...")
df = pd.read_csv('toolmatcher_ai_raw_dataset_v2.csv')
print(f"✓ Loaded {len(df)} records")
print(f"✓ Columns: {list(df.columns)}")

# Data cleaning
print("\n[2/9] Cleaning data...")
initial_count = len(df)

# Remove duplicates
df = df.drop_duplicates()

# Strip whitespace
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.strip()

# Remove invalid entries (tool_name and task_category are critical)
df = df[df['tool_name'].notna() & (df['tool_name'].str.len() > 0)]
df = df[df['task_category'].notna() & (df['task_category'].str.len() > 0)]

# Handle missing values in features we'll use
# For boolean columns
if 'free_plan_available' in df.columns:
    df['free_plan_available'] = df['free_plan_available'].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})
    df['free_plan_available'] = df['free_plan_available'].fillna(0)

if 'annual_billing_available' in df.columns:
    df['annual_billing_available'] = df['annual_billing_available'].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})
    df['annual_billing_available'] = df['annual_billing_available'].fillna(0)

# For numerical columns, fill with median
numeric_cols = ['monthly_budget_usd', 'monthly_price_usd', 'ease_of_use_1_5', 
                'output_quality_1_10', 'efficiency_rate_pct']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

# For categorical columns, fill with 'Unknown'
categorical_cols = ['expertise_level', 'platform', 'best_model']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')

cleaned_count = len(df)
print(f"✓ Cleaned: {initial_count} → {cleaned_count} records ({initial_count - cleaned_count} removed)")

# Save cleaned dataset
df.to_csv('cleaned_toolmatcher_dataset.csv', index=False)
print(f"✓ Saved: cleaned_toolmatcher_dataset.csv")

# Feature engineering
print("\n[3/9] Engineering features...")

# Encode categorical variables
label_encoders = {}
categorical_features = []

# Add available categorical features
if 'expertise_level' in df.columns:
    categorical_features.append('expertise_level')
if 'task_category' in df.columns:
    categorical_features.append('task_category')
if 'platform' in df.columns:
    categorical_features.append('platform')
if 'best_model' in df.columns:
    categorical_features.append('best_model')

for feature in categorical_features:
    le = LabelEncoder()
    df[feature + '_Encoded'] = le.fit_transform(df[feature])
    label_encoders[feature] = le

# Numerical features
numerical_features = []
if 'monthly_budget_usd' in df.columns:
    numerical_features.append('monthly_budget_usd')
if 'monthly_price_usd' in df.columns:
    numerical_features.append('monthly_price_usd')
if 'ease_of_use_1_5' in df.columns:
    numerical_features.append('ease_of_use_1_5')
if 'output_quality_1_10' in df.columns:
    numerical_features.append('output_quality_1_10')
if 'efficiency_rate_pct' in df.columns:
    numerical_features.append('efficiency_rate_pct')
if 'free_plan_available' in df.columns:
    numerical_features.append('free_plan_available')
if 'annual_billing_available' in df.columns:
    numerical_features.append('annual_billing_available')

# Create feature matrix
feature_columns = [f + '_Encoded' for f in categorical_features] + numerical_features
X = df[feature_columns].values
y = LabelEncoder().fit_transform(df['tool_name'])  # Predict tool_name
label_encoders['tool_name'] = LabelEncoder().fit(df['tool_name'])

print(f"✓ Features: {len(feature_columns)} columns")
print(f"✓ Target classes: {len(np.unique(y))}")

# Check class distribution
unique, counts = np.unique(y, return_counts=True)
min_samples = counts.min()
print(f"✓ Min samples per class: {min_samples}")

# Split data 
print("\n[4/9] Splitting data...")
test_size = 0.2
# Use stratified split only if we have at least 2 samples per class
if min_samples >= 2:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"✓ Stratified split: {len(X_train)} train, {len(X_test)} test")
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    print(f"✓ Random split: {len(X_train)} train, {len(X_test)} test")

# Feature scaling
print("\n[5/9] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✓ StandardScaler fitted")

# Determine CV folds based on min samples
cv_folds = max(2, min(5, min_samples))  # At least 2, at most 5
print(f"✓ Using {cv_folds} CV folds")

# Model training with hyperparameter tuning
print("\n[6/9] Training models with hyperparameter tuning...")
print("This may take several minutes...")

results = []

# 1. Random Forest with GridSearch
print("\n  → Random Forest (with GridSearch)...")
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 'log2']
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_params,
    cv=cv_folds,
    n_jobs=1,  # Set to 1 since RandomForest already uses parallel
    verbose=0,
    scoring='accuracy'
)
rf_grid.fit(X_train_scaled, y_train)
rf_best = rf_grid.best_estimator_
rf_pred = rf_best.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, rf_pred)
results.append({
    'Model': 'Random Forest (Tuned)',
    'Accuracy': rf_acc,
    'Best_Params': str(rf_grid.best_params_)
})
print(f"    Best params: {rf_grid.best_params_}")
print(f"    Accuracy: {rf_acc:.4f}")

# 2. Gradient Boosting with GridSearch
print("\n  → Gradient Boosting (with GridSearch)...")
gb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5],
    'min_samples_split': [2, 5]
}
gb_grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    gb_params,
    cv=cv_folds,
    n_jobs=-1,
    verbose=0,
    scoring='accuracy'
)
gb_grid.fit(X_train_scaled, y_train)
gb_best = gb_grid.best_estimator_
gb_pred = gb_best.predict(X_test_scaled)
gb_acc = accuracy_score(y_test, gb_pred)
results.append({
    'Model': 'Gradient Boosting (Tuned)',
    'Accuracy': gb_acc,
    'Best_Params': str(gb_grid.best_params_)
})
print(f"    Best params: {gb_grid.best_params_}")
print(f"    Accuracy: {gb_acc:.4f}")

# 3. Logistic Regression
print("\n  → Logistic Regression...")
lr_params = {
    'C': [0.1, 1, 10],
    'max_iter': [500, 1000],
    'solver': ['lbfgs', 'saga']
}
lr_grid = GridSearchCV(
    LogisticRegression(random_state=42),
    lr_params,
    cv=cv_folds,
    n_jobs=-1,
    verbose=0,
    scoring='accuracy'
)
lr_grid.fit(X_train_scaled, y_train)
lr_best = lr_grid.best_estimator_
lr_pred = lr_best.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, lr_pred)
results.append({
    'Model': 'Logistic Regression (Tuned)',
    'Accuracy': lr_acc,
    'Best_Params': str(lr_grid.best_params_)
})
print(f"    Best params: {lr_grid.best_params_}")
print(f"    Accuracy: {lr_acc:.4f}")

# 4. K-Nearest Neighbors
print("\n  → K-Nearest Neighbors...")
knn_params = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
knn_grid = GridSearchCV(
    KNeighborsClassifier(),
    knn_params,
    cv=cv_folds,
    n_jobs=-1,
    verbose=0,
    scoring='accuracy'
)
knn_grid.fit(X_train_scaled, y_train)
knn_best = knn_grid.best_estimator_
knn_pred = knn_best.predict(X_test_scaled)
knn_acc = accuracy_score(y_test, knn_pred)
results.append({
    'Model': 'KNN (Tuned)',
    'Accuracy': knn_acc,
    'Best_Params': str(knn_grid.best_params_)
})
print(f"    Best params: {knn_grid.best_params_}")
print(f"    Accuracy: {knn_acc:.4f}")

# 5. Voting Classifier (Ensemble)
print("\n  → Voting Classifier (Ensemble)...")
voting_clf = VotingClassifier(
    estimators=[
        ('rf', rf_best),
        ('gb', gb_best),
        ('lr', lr_best)
    ],
    voting='hard',
    n_jobs=-1
)
voting_clf.fit(X_train_scaled, y_train)
voting_pred = voting_clf.predict(X_test_scaled)
voting_acc = accuracy_score(y_test, voting_pred)
results.append({
    'Model': 'Voting Ensemble',
    'Accuracy': voting_acc,
    'Best_Params': 'RF + GB + LR'
})
print(f"    Accuracy: {voting_acc:.4f}")

# 6. Stacking Classifier (Advanced Ensemble)
print("\n  → Stacking Classifier (Advanced Ensemble)...")
stacking_clf = StackingClassifier(
    estimators=[
        ('rf', rf_best),
        ('gb', gb_best),
        ('knn', knn_best)
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=cv_folds,
    n_jobs=-1
)
stacking_clf.fit(X_train_scaled, y_train)
stacking_pred = stacking_clf.predict(X_test_scaled)
stacking_acc = accuracy_score(y_test, stacking_pred)
results.append({
    'Model': 'Stacking Ensemble',
    'Accuracy': stacking_acc,
    'Best_Params': 'RF + GB + KNN → LR'
})
print(f"    Accuracy: {stacking_acc:.4f}")

# Find best model
print("\n[7/9] Comparing models...")
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False)
results_df.to_csv('model_performance_results.csv', index=False)

print("\n" + "="*80)
print("MODEL PERFORMANCE COMPARISON")
print("="*80)
for idx, row in results_df.iterrows():
    print(f"{row['Model']:35} | Accuracy: {row['Accuracy']:.4f} ({row['Accuracy']*100:.2f}%)")

best_model_name = results_df.iloc[0]['Model']
best_accuracy = results_df.iloc[0]['Accuracy']

print("\n" + "="*80)
print(f"🏆 BEST MODEL: {best_model_name}")
print(f"🎯 ACCURACY: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print("="*80)

# Select best model for saving
model_mapping = {
    'Random Forest (Tuned)': rf_best,
    'Gradient Boosting (Tuned)': gb_best,
    'Logistic Regression (Tuned)': lr_best,
    'KNN (Tuned)': knn_best,
    'Voting Ensemble': voting_clf,
    'Stacking Ensemble': stacking_clf
}
best_model = model_mapping[best_model_name]

# Save best model
print("\n[8/9] Saving best model...")
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
print("✓ Saved: best_model.pkl")
print("✓ Saved: scaler.pkl")
print("✓ Saved: label_encoders.pkl")

# Detailed evaluation
print("\n[9/9] Detailed evaluation of best model...")
print("\nTop 10 Most Common Tools in Test Set:")
print("-" * 80)
y_test_tools = label_encoders['tool_name'].inverse_transform(y_test)
test_tool_counts = pd.Series(y_test_tools).value_counts().head(10)
for tool, count in test_tool_counts.items():
    print(f"  {tool:40} : {count} occurrences")

print("\nClassification Report (Top 10 Tools):")
print("-" * 80)
# Get predictions for best model
best_pred = best_model.predict(X_test_scaled)
# Convert to tool names
pred_tools = label_encoders['tool_name'].inverse_transform(best_pred)
# Calculate accuracy for top 10 tools
top_10_tools = test_tool_counts.index.tolist()
mask = pd.Series(y_test_tools).isin(top_10_tools).values
if mask.sum() > 0:
    filtered_y_test = [y for y, m in zip(y_test_tools, mask) if m]
    filtered_y_pred = [p for p, m in zip(pred_tools, mask) if m]
    print(classification_report(filtered_y_test, filtered_y_pred, 
                              labels=top_10_tools,
                              zero_division=0))

# Feature importance (if available)
if hasattr(best_model, 'feature_importances_'):
    print("\nTop 10 Most Important Features:")
    print("-" * 80)
    feature_names = [f.replace('_Encoded', '') for f in feature_columns]
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    for i, idx in enumerate(indices, 1):
        print(f"{i:2}. {feature_names[idx]:30} | Importance: {importances[idx]:.4f}")

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
print(f"\nDataset: {cleaned_count} records")
print(f"Features: {len(feature_columns)}")
print(f"Tool Classes: {len(np.unique(y))}")
print(f"Best Model: {best_model_name}")
print(f"Test Accuracy: {best_accuracy*100:.2f}%")
print("\nFiles generated:")
print("  - cleaned_toolmatcher_dataset.csv")
print("  - model_performance_results.csv")
print("  - best_model.pkl")
print("  - scaler.pkl")
print("  - label_encoders.pkl")
print("\n" + "="*80)
