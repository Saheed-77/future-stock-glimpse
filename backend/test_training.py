#!/usr/bin/env python3
"""
Test the ML training system
"""

def test_training():
    print("🔬 Testing ML Training System")
    print("=" * 40)
    
    try:
        # Import the training function
        from ml_predictor_v2 import train_model_for_symbol, StockPredictor
        print("✅ Successfully imported ML modules")
        
        # Test with a simple symbol
        print("\n📈 Starting training for AAPL...")
        print("Note: This will take 1-2 minutes to complete")
        
        # Train the model
        predictor = train_model_for_symbol('AAPL')
        
        if predictor:
            print("✅ Training completed successfully!")
            
            # Get metrics
            metrics = predictor.calculate_metrics()
            print(f"\n📊 Model Performance:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
            
            # Test prediction
            print(f"\n🔮 Testing prediction...")
            predictions = predictor.predict_future_prices(days=5)
            if predictions:
                print(f"  Next 5 days predictions: {predictions[:5]}")
            
            print(f"\n💾 Model saved successfully!")
            return True
            
        else:
            print("❌ Training failed - no predictor returned")
            return False
            
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_training()
    if success:
        print("\n🎉 Training test completed successfully!")
    else:
        print("\n💥 Training test failed!")
