#!/usr/bin/env python3
"""
Training script with dependency checking and installation
"""

import sys
import subprocess
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_and_install_dependencies():
    """Check and install required dependencies"""
    required_packages = {
        'pandas': 'pandas>=1.5.0',
        'numpy': 'numpy>=1.21.0',
        'sklearn': 'scikit-learn>=1.3.0',
        'yfinance': 'yfinance>=0.2.0'
    }
    
    print("🔍 Checking dependencies...")
    
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✅ {module} is available")
        except ImportError:
            print(f"❌ {module} not found, installing {package}...")
            if install_package(package):
                print(f"✅ {package} installed successfully")
            else:
                print(f"❌ Failed to install {package}")
                return False
    
    return True

def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'AAPL'
    
    print("🚀 Stock ML Model Trainer")
    print("=" * 40)
    
    # Check dependencies first
    if not check_and_install_dependencies():
        print("❌ Failed to install required dependencies")
        sys.exit(1)
    
    # Now try to import our ML module
    try:
        from ml_predictor_v2 import train_model_for_symbol
        print("✅ ML predictor module loaded successfully")
    except ImportError as e:
        print(f"❌ Could not import ML predictor: {e}")
        sys.exit(1)
    
    print(f"\n🎯 Training model for {symbol}...")
    print("📊 Fetching 10 years of historical data...")
    print("⏳ This may take 1-2 minutes...")
    
    # Create models directory
    if not os.path.exists('models'):
        os.makedirs('models')
        print("📁 Created models directory")
    
    try:
        # Train the model
        predictor = train_model_for_symbol(symbol)
        
        if predictor:
            print(f"\n✅ Training completed successfully for {symbol}!")
            print(f"💾 Model saved to: models/stock_predictor_{symbol.lower()}.pkl")
            
            # Show metrics
            try:
                metrics = predictor.calculate_metrics()
                print(f"\n📊 Model Performance:")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        print(f"  📈 {metric}: {value:.4f}")
                    else:
                        print(f"  📈 {metric}: {value}")
                
                # Test prediction
                print(f"\n🔮 Testing predictions...")
                predictions = predictor.predict_future_prices(days=5)
                if predictions:
                    print(f"  Next 5 days: {[f'${p:.2f}' for p in predictions[:5]]}")
                
            except Exception as e:
                print(f"⚠️  Metrics calculation failed: {e}")
            
            print(f"\n🎉 Training complete! Model ready for predictions.")
            
        else:
            print("❌ Training failed - no model created")
            
    except Exception as e:
        print(f"❌ Training error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
