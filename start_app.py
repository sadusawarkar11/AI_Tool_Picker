"""
ToolMatcher AI - Flask Server Startup Script
Loads the included model artifacts and starts the ToolMatcher AI web interface
"""

from app import app, load_models

if __name__ == '__main__':
    print("\n" + "="*70)
    print(" " * 15 + "🚀 TOOLMATCHER AI - STARTING SERVER")
    print("="*70)
    print("\n📦 Loading ML models and data...")
    
    if load_models():
        # Import df_clean after models are loaded
        from app import df_clean, model_name, model_accuracy
        
        print("\n" + "="*70)
        print(" " * 25 + "✅ SERVER READY!")
        print("="*70)
        print(f"\n📍 Main App:      http://localhost:5000")
        print(f"📊 Analytics:     http://localhost:5000/analytics")
        accuracy_text = f"{model_accuracy:.2f}%" if model_accuracy is not None else "not recorded"
        print(f"\n🎯 Model:         {model_name}")
        print(f"📈 Test accuracy: {accuracy_text}")
        print(f"🤖 Total Tools:   {len(df_clean['tool_name'].unique())}")
        print(f"📂 Total Records: {len(df_clean):,}")
        print("\n" + "="*70)
        print("⌨️  Press Ctrl+C to stop the server")
        print("="*70 + "\n")
        
        # Start Flask server without debug mode to avoid reloading issues
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("\n" + "="*70)
        print(" " * 20 + "❌ FAILED TO START SERVER")
        print("="*70)
        print("\n⚠️  Required files missing. Please check:")
        print("   ✗ cleaned_toolmatcher_dataset.csv")
        print("   ✗ best_model.pkl")
        print("   ✗ scaler.pkl")
        print("   ✗ label_encoders.pkl")
        print("\n💡 Solution: Run 'python advanced_model_training.py' first!")
        print("="*70 + "\n")
