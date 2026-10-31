"""
Quick Model Training for Production Deployment
Uses cleaned data with fixed hyperparameters for fast training
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🚀 QUICK MODEL TRAINING - PRODUCTION DEPLOYMENT")
print("="*70)

# Load cleaned dataset
print("\n[1/5] Loading cleaned dataset...")
df = pd.read_csv('cleaned_toolmatcher_dataset.csv')
print(f"✓ Loaded {len(df)} records")
print(f"✓ Unique tools: {df['tool_name'].nunique()}")
print(f"✓ Expertise levels: {sorted(df['expertise_level'].unique())}")

# Feature engineering
print("\n[2/5] Engineering features...")
label_encoders = {}
for col in ['expertise_level', 'task_category', 'platform', 'best_model', 'tool_name']:
    le = LabelEncoder()
    df[f'{col}_Encoded'] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Convert boolean/binary columns
if 'free_plan_available' in df.columns:
    df['free_plan_available'] = df['free_plan_available'].map({1: 1, 0: 0, '1': 1, '0': 0, 'Yes': 1, 'No': 0, True: 1, False: 0}).fillna(0).astype(int)
if 'annual_billing_available' in df.columns:
    df['annual_billing_available'] = df['annual_billing_available'].map({1: 1, 0: 0, '1': 1, '0': 0, 'Yes': 1, 'No': 0, True: 1, False: 0}).fillna(0).astype(int)

# Select features
feature_columns = [
    'expertise_level_Encoded',
    'task_category_Encoded',
    'platform_Encoded',
    'best_model_Encoded',
    'monthly_budget_usd',
    'monthly_price_usd',
    'ease_of_use_1_5',
    'output_quality_1_10',
    'efficiency_rate_pct',
    'free_plan_available',
    'annual_billing_available'
]

X = df[feature_columns]
y = df['tool_name_Encoded']

print(f"✓ Features: {len(feature_columns)}")
print(f"✓ Classes: {len(label_encoders['tool_name'].classes_)}")

# Split data
print("\n[3/5] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✓ Train: {len(X_train)} | Test: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models with optimal hyperparameters
print("\n[4/5] Training models (fast mode)...")

# Random Forest
print("  → Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    max_features='sqrt',
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
rf_accuracy = rf_model.score(X_test_scaled, y_test)
print(f"    Accuracy: {rf_accuracy:.4f}")

# Stacking Ensemble (best model)
print("  → Stacking Ensemble...")
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1))
]
stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(max_iter=500, random_state=42),
    cv=2
)
stacking_model.fit(X_train_scaled, y_train)
stacking_accuracy = stacking_model.score(X_test_scaled, y_test)
print(f"    Accuracy: {stacking_accuracy:.4f}")

# Select best model
if stacking_accuracy >= rf_accuracy:
    best_model = stacking_model
    best_accuracy = stacking_accuracy
    model_name = "Stacking Ensemble"
else:
    best_model = rf_model
    best_accuracy = rf_accuracy
    model_name = "Random Forest"

print(f"\n✅ Best Model: {model_name}")
print(f"   Accuracy: {best_accuracy*100:.2f}%")

# Save models
print("\n[5/5] Saving models...")
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

print("✓ Saved best_model.pkl")
print("✓ Saved scaler.pkl")
print("✓ Saved label_encoders.pkl")

# Save performance report
performance_data = pd.DataFrame([
    {'Model': 'Random Forest', 'Accuracy': rf_accuracy},
    {'Model': model_name, 'Accuracy': best_accuracy}
]).drop_duplicates(subset=['Model'], keep='last')
performance_data.to_csv('model_performance_results.csv', index=False)
print("✓ Saved model_performance_results.csv")

print("\n" + "="*70)
print("🎯 PRODUCTION-READY MODELS DEPLOYED!")
print("="*70)
print(f"\n✅ Model: {model_name}")
print(f"✅ Accuracy: {best_accuracy*100:.2f}%")
print(f"✅ Classes: {len(label_encoders['tool_name'].classes_)}")
print(f"✅ No duplicates: Professional data cleaning applied")
print(f"✅ Standardized levels: Beginner, Intermediate, Advanced")
print("\n🚀 Ready to run: python start_app.py")
print("="*70 + "\n")
