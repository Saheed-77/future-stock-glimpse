print("Starting diagnosis...")

# Test 1: Basic imports
print("Test 1: Basic Python imports")
try:
    import sys
    import os
    print("✅ sys, os imported")
except Exception as e:
    print(f"❌ Basic imports failed: {e}")

# Test 2: Data science imports
print("\nTest 2: Data science libraries")
try:
    import pandas as pd
    print("✅ pandas imported")
except Exception as e:
    print(f"❌ pandas failed: {e}")

try:
    import numpy as np
    print("✅ numpy imported")
except Exception as e:
    print(f"❌ numpy failed: {e}")

try:
    from sklearn.ensemble import RandomForestRegressor
    print("✅ sklearn imported")
except Exception as e:
    print(f"❌ sklearn failed: {e}")

try:
    import yfinance as yf
    print("✅ yfinance imported")
except Exception as e:
    print(f"❌ yfinance failed: {e}")

# Test 3: Our module
print("\nTest 3: Our ML module")
try:
    import ml_predictor_v2
    print("✅ ml_predictor_v2 imported")
except Exception as e:
    print(f"❌ ml_predictor_v2 failed: {e}")

try:
    from ml_predictor_v2 import train_model_for_symbol
    print("✅ train_model_for_symbol imported")
except Exception as e:
    print(f"❌ train_model_for_symbol failed: {e}")

print("\nDiagnosis complete!")
