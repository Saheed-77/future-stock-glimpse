#!/usr/bin/env python3
"""
Test script for LSTM functionality
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test if all required packages can be imported"""
    try:
        import yfinance as yf
        print("✓ yfinance imported successfully")
        
        import numpy as np
        print("✓ numpy imported successfully")
        
        import pandas as pd
        print("✓ pandas imported successfully")
        
        from sklearn.preprocessing import MinMaxScaler
        print("✓ scikit-learn imported successfully")
        
        try:
            import tensorflow as tf
            from keras.models import Sequential
            from keras.layers import LSTM, Dense, Dropout
            print("✓ TensorFlow and Keras imported successfully")
            print(f"  TensorFlow version: {tf.__version__}")
            return True
        except ImportError as e:
            print(f"✗ TensorFlow/Keras import failed: {e}")
            return False
            
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_simple_prediction():
    """Test a simple stock data fetch and prediction setup"""
    try:
        import yfinance as yf
        import pandas as pd
        from datetime import datetime, timedelta
        
        print("\nTesting stock data fetch...")
        
        # Fetch some sample data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)
        
        data = yf.download('AAPL', start=start_date, end=end_date)
        
        if not data.empty:
            print(f"✓ Successfully fetched {len(data)} days of AAPL data")
            print(f"  Latest close price: ${data['Close'].iloc[-1]:.2f}")
            return True
        else:
            print("✗ Failed to fetch stock data")
            return False
            
    except Exception as e:
        print(f"✗ Stock data test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing LSTM Stock Prediction Setup")
    print("=" * 40)
    
    imports_ok = test_imports()
    data_ok = test_simple_prediction()
    
    if imports_ok and data_ok:
        print("\n✓ All tests passed! LSTM functionality should work.")
    else:
        print("\n✗ Some tests failed. Check the installation.")
        sys.exit(1)
