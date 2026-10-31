from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Global variables for models and data
df_clean = None
best_model = None
scaler = None
label_encoders = None
feature_columns = None
model_name = None
model_accuracy = None

def load_models():
    """Load trained ML models and preprocessed data"""
    global df_clean, best_model, scaler, label_encoders, feature_columns
    global model_name, model_accuracy
    
    try:
        print("Loading dataset...")
        df_clean = pd.read_csv('cleaned_toolmatcher_dataset.csv')
        print(f"✓ Dataset loaded: {len(df_clean)} records")
        
        print("Loading trained model...")
        best_model = joblib.load('best_model.pkl')
        print(f"✓ Model loaded: {type(best_model).__name__}")
        
        print("Loading scaler...")
        scaler = joblib.load('scaler.pkl')
        print("✓ Scaler loaded")
        
        print("Loading label encoders...")
        label_encoders = joblib.load('label_encoders.pkl')
        print(f"✓ Label encoders loaded: {list(label_encoders.keys())}")

        model_name = type(best_model).__name__.replace('Classifier', '')
        model_accuracy = None
        try:
            performance = pd.read_csv('model_performance_results.csv')
            performance = performance.dropna(subset=['Model', 'Accuracy'])
            if not performance.empty:
                best_result = performance.sort_values('Accuracy', ascending=False).iloc[0]
                model_name = str(best_result['Model'])
                model_accuracy = float(best_result['Accuracy']) * 100
        except (OSError, ValueError, KeyError):
            pass
        
        # Define feature columns in the same order as training
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
        
        print("\n" + "="*60)
        print("✅ ALL MODELS LOADED SUCCESSFULLY!")
        print("="*60)
        accuracy_text = f"{model_accuracy:.2f}%" if model_accuracy is not None else "not recorded"
        print(f"Model: {model_name} | Recorded test accuracy: {accuracy_text}")
        print(f"Tool Classes: {len(label_encoders['tool_name'].classes_)}")
        print("="*60 + "\n")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_categories', methods=['GET'])
def get_categories():
    """Get available task categories and expertise levels"""
    if df_clean is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    # Get unique values from dataset
    categories = sorted(df_clean['task_category'].unique().tolist())
    expertise_levels = sorted(df_clean['expertise_level'].unique().tolist())
    platforms = sorted(df_clean['platform'].unique().tolist())
    
    return jsonify({
        'categories': categories,
        'expertise_levels': expertise_levels,
        'platforms': platforms,
        'total_tools': len(df_clean['tool_name'].unique())
    })

@app.route('/predict', methods=['POST'])
def predict():
    """PRODUCTION-GRADE prediction endpoint with deduplication and ranking"""
    try:
        data = request.json
        
        # Get user inputs
        user_requirement = data.get('user_requirement', '')
        expertise = data.get('expertise_level', 'Beginner')
        category = data.get('task_category', 'General Assistant')
        budget = float(data.get('monthly_budget', 50))
        platform = data.get('platform', 'Web')
        
        print(f"\n{'='*60}")
        print(f"🔍 PREDICTION REQUEST")
        print(f"{'='*60}")
        print(f"Expertise: {expertise}")
        print(f"Category: {category}")
        print(f"Budget: ${budget}")
        print(f"Platform: {platform}")
        
        # Filter dataset based on user requirements (exact match first)
        filtered_df = df_clean[
            (df_clean['task_category'] == category) & 
            (df_clean['expertise_level'] == expertise)
        ].copy()
        
        # Fallback strategy if no exact matches
        if len(filtered_df) == 0:
            print(f"⚠️  No exact match, expanding to category: {category}")
            filtered_df = df_clean[df_clean['task_category'] == category].copy()
        
        if len(filtered_df) == 0:
            print(f"⚠️  No category match, using all tools")
            filtered_df = df_clean.copy()
        
        print(f"   Analyzing {len(filtered_df)} candidate records...")
        
        # CRITICAL: Use dictionary to deduplicate by tool_name
        # Key = tool_name, Value = best prediction for that tool
        tool_predictions = {}
        
        for idx, row in filtered_df.iterrows():
            try:
                tool_name = str(row['tool_name'])
                
                # Prepare features (same order as training)
                features = np.array([[
                    label_encoders['expertise_level'].transform([expertise])[0],
                    label_encoders['task_category'].transform([category])[0],
                    label_encoders['platform'].transform([row['platform']])[0] if pd.notna(row['platform']) else 0,
                    label_encoders['best_model'].transform([row['best_model']])[0] if pd.notna(row['best_model']) else 0,
                    budget,  # monthly_budget_usd
                    float(row['monthly_price_usd']) if pd.notna(row['monthly_price_usd']) else 0,
                    float(row['ease_of_use_1_5']) if pd.notna(row['ease_of_use_1_5']) else 4,
                    float(row['output_quality_1_10']) if pd.notna(row['output_quality_1_10']) else 8,
                    float(row['efficiency_rate_pct']) if pd.notna(row['efficiency_rate_pct']) else 85,
                    int(row['free_plan_available']) if pd.notna(row['free_plan_available']) else 0,
                    int(row['annual_billing_available']) if pd.notna(row['annual_billing_available']) else 0
                ]])
                
                # Scale features
                features_scaled = scaler.transform(features)
                
                # Map probabilities by the model's actual class IDs. Some labels
                # may be absent from the training split.
                proba = best_model.predict_proba(features_scaled)[0]
                tool_name_idx = label_encoders['tool_name'].transform([tool_name])[0]
                class_probabilities = dict(zip(best_model.classes_, proba))
                confidence = float(class_probabilities.get(tool_name_idx, 0.0))
                
                # Calculate combined score with professional weighting
                budget_match = 1.0 if row['monthly_price_usd'] <= budget else 0.5
                
                combined_score = (
                    confidence * 40 +  # ML model confidence (40%)
                    row['efficiency_rate_pct'] * 0.3 +  # Efficiency (30%)
                    row['output_quality_1_10'] * 2 +  # Quality (20%)
                    row['ease_of_use_1_5'] * 2  # Ease of use (10%)
                ) * budget_match
                
                prediction = {
                    'tool_name': tool_name,
                    'best_model': str(row['best_model']) if pd.notna(row['best_model']) else 'N/A',
                    'confidence': float(confidence),
                    'combined_score': float(combined_score),
                    'efficiency_rate': float(row['efficiency_rate_pct']),
                    'output_quality': float(row['output_quality_1_10']),
                    'ease_of_use': float(row['ease_of_use_1_5']),
                    'monthly_price': float(row['monthly_price_usd']) if pd.notna(row['monthly_price_usd']) else 0,
                    'yearly_price': float(row['yearly_price_usd']) if pd.notna(row['yearly_price_usd']) else 0,
                    'free_plan_available': int(row['free_plan_available']),
                    'free_quota': str(row['free_quota_details']) if pd.notna(row['free_quota_details']) else 'N/A',
                    'free_trial_days': int(row['free_trial_days']) if pd.notna(row['free_trial_days']) else 0,
                    'plan_name': str(row['cheapest_paid_plan']) if pd.notna(row['cheapest_paid_plan']) else 'Pro',
                    'context_window': str(row['context_window_tokens']) if pd.notna(row['context_window_tokens']) else 'N/A',
                    'platform': str(row['platform']) if pd.notna(row['platform']) else 'Web',
                    'budget_match': budget_match == 1.0
                }
                
                # DEDUPLICATION LOGIC: Keep only the best prediction per tool
                if tool_name not in tool_predictions:
                    tool_predictions[tool_name] = prediction
                else:
                    # If tool exists, keep the one with higher combined_score
                    if prediction['combined_score'] > tool_predictions[tool_name]['combined_score']:
                        tool_predictions[tool_name] = prediction
                
            except Exception as e:
                print(f"⚠️  Error predicting for {row['tool_name']}: {e}")
                continue
        
        # Convert dictionary back to list (now deduplicated!)
        predictions = list(tool_predictions.values())
        
        print(f"   Deduplicated to {len(predictions)} unique tools")
        
        # Sort by combined score
        predictions.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Split into free and paid with limits
        free_tools = [p for p in predictions if p['free_plan_available'] == 1][:10]
        paid_tools = [p for p in predictions if p['monthly_price'] > 0 and p['monthly_price'] <= budget][:10]
        
        print(f"\n✅ PREDICTIONS COMPLETE")
        print(f"   Total analyzed: {len(filtered_df)} records")
        print(f"   Unique tools: {len(predictions)}")
        print(f"   Free recommendations: {len(free_tools)}")
        print(f"   Paid recommendations: {len(paid_tools)}")
        if free_tools:
            print(f"   🥇 Top recommendation: {free_tools[0]['tool_name']} (confidence: {free_tools[0]['confidence']:.2%})")
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': True,
            'free_tools': free_tools,
            'paid_tools': paid_tools,
            'total_free': len(free_tools),
            'total_paid': len(paid_tools),
            'model_accuracy': model_accuracy,
            'total_analyzed': len(predictions),
            'records_processed': len(filtered_df),
            'deduplication_applied': True
        })
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@app.route('/get_analytics_data', methods=['GET'])
def get_analytics_data():
    """Get analytics data for dashboard"""
    try:
        if df_clean is None:
            return jsonify({'error': 'Data not loaded'}), 500
        
        # Category distribution
        category_counts = df_clean['task_category'].value_counts().head(10).to_dict()
        
        # Top tools
        tool_counts = df_clean['tool_name'].value_counts().head(10).to_dict()
        
        # Free vs Paid distribution
        free_plan_dist = df_clean['free_plan_available'].value_counts().to_dict()
        free_plan_dist = {
            'Free Plans': int(free_plan_dist.get(1, 0)),
            'Paid Only': int(free_plan_dist.get(0, 0))
        }
        
        # Average metrics
        avg_metrics = {
            'avg_efficiency': float(df_clean['efficiency_rate_pct'].mean()),
            'avg_quality': float(df_clean['output_quality_1_10'].mean()),
            'avg_ease': float(df_clean['ease_of_use_1_5'].mean()),
            'avg_monthly_price': float(df_clean[df_clean['monthly_price_usd'] > 0]['monthly_price_usd'].mean())
        }
        
        # Model performance
        try:
            model_performance = pd.read_csv('model_performance_results.csv')
            model_data = (
                model_performance
                .dropna(subset=['Model', 'Accuracy'])
                .drop_duplicates(subset=['Model'], keep='last')
                .sort_values('Accuracy', ascending=False)
                .to_dict('records')
            )
        except (OSError, ValueError, KeyError):
            accuracy_value = model_accuracy / 100 if model_accuracy is not None else None
            model_data = [{'Model': model_name, 'Accuracy': accuracy_value}]
        
        # Expertise level distribution
        expertise_dist = df_clean['expertise_level'].value_counts().to_dict()
        
        # Platform distribution
        platform_dist = df_clean['platform'].value_counts().head(5).to_dict()
        
        return jsonify({
            'category_counts': category_counts,
            'tool_counts': tool_counts,
            'free_plan_dist': free_plan_dist,
            'avg_metrics': avg_metrics,
            'model_performance': model_data,
            'expertise_dist': expertise_dist,
            'platform_dist': platform_dist,
            'total_tools': len(df_clean['tool_name'].unique()),
            'total_records': len(df_clean),
            'model_accuracy': model_accuracy
        })
        
    except Exception as e:
        print(f"Analytics error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 TOOLMATCHER AI - FLASK SERVER")
    print("="*60)
    print("\nLoading ML models and data...")
    
    if load_models():
        print("\n" + "="*60)
        print("✅ SERVER READY!")
        print("="*60)
        print(f"\n📍 Access the app at: http://localhost:5000")
        print(f"📊 Analytics at: http://localhost:5000/analytics")
        accuracy_text = f"{model_accuracy:.2f}%" if model_accuracy is not None else "not recorded"
        print(f"\n🎯 Model Performance: {accuracy_text}")
        print(f"🤖 Algorithm: {model_name}")
        print(f"📦 Total Tools: {len(df_clean['tool_name'].unique())}")
        print("\n" + "="*60)
        print("Press Ctrl+C to stop the server")
        print("="*60 + "\n")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n" + "="*60)
        print("❌ FAILED TO START SERVER")
        print("="*60)
        print("\n⚠️  Make sure these files exist:")
        print("   - cleaned_toolmatcher_dataset.csv")
        print("   - best_model.pkl")
        print("   - scaler.pkl")
        print("   - label_encoders.pkl")
        print("\n💡 Run 'python advanced_model_training.py' first!")
        print("="*60 + "\n")
