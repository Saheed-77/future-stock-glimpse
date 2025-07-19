#!/usr/bin/env python3
"""
Simple script to train ML models for stock prediction
"""

import sys
import os

# Add better error handling for imports
try:
    from ml_predictor_v2 import train_model_for_symbol
    print("✅ ML modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all dependencies are installed:")
    print("pip install scikit-learn pandas yfinance numpy")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error during import: {e}")
    sys.exit(1)

def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'AAPL'
    
    print(f"🚀 Starting training for {symbol}...")
    print("📊 This will fetch 10 years of historical data and train the model...")
    print("⏳ Please wait, this may take 1-2 minutes...")
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
        print("📁 Created models directory")
    
    try:
        # Train the model
        print(f"🔄 Training model for {symbol}...")
        predictor = train_model_for_symbol(symbol)
        
        if predictor:
            print(f"✅ Training completed successfully for {symbol}!")
            print(f"💾 Model saved to: models/stock_predictor_{symbol.lower()}.pkl")
            
            # Show some metrics
            try:
                metrics = predictor.calculate_metrics()
                print(f"📊 Model Performance Metrics:")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        print(f"  📈 {metric}: {value:.4f}")
                    else:
                        print(f"  📈 {metric}: {value}")
                        
                # Test a prediction
                print(f"\n🔮 Testing prediction capability...")
                test_predictions = predictor.predict_future_prices(days=5)
                if test_predictions and len(test_predictions) > 0:
                    print(f"  Next 5 days price predictions: {[f'${p:.2f}' for p in test_predictions[:5]]}")
                    
            except Exception as metric_error:
                print(f"⚠️  Could not calculate metrics: {metric_error}")
                    
        else:
            print("❌ Training failed - predictor is None")
            print("This might be due to:")
            print("  • Network connectivity issues")
            print("  • Invalid stock symbol")
            print("  • Insufficient historical data")
            
    except Exception as e:
        print(f"❌ Training failed with error: {str(e)}")
        print(f"🔍 Error type: {type(e).__name__}")
        import traceback
        print("📋 Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
