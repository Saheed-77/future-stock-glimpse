#!/usr/bin/env python3
"""
Quick demo of how to train the ML models
"""

print("🚀 Stock Prediction Model Training Demo")
print("=" * 50)

# Method 1: Direct Training
print("\n📚 Method 1: Direct Training")
print("To train a model directly in Python:")
print("""
from ml_predictor_v2 import train_model_for_symbol

# Train for a specific symbol
predictor = train_model_for_symbol('AAPL')
print("Training completed!")
""")

# Method 2: API Training
print("\n🌐 Method 2: API Training")
print("To train via the API endpoint:")
print("""
POST /api/stock/AAPL/train
""")

# Method 3: Batch Training
print("\n📦 Method 3: Batch Training")
print("To train multiple symbols:")
print("""
symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
for symbol in symbols:
    predictor = train_model_for_symbol(symbol)
    print(f"Completed: {symbol}")
""")

print("\n✨ Training Process:")
print("1. Fetches 10 years of historical data from Yahoo Finance")
print("2. Calculates technical indicators (RSI, Moving Averages, Volatility)")
print("3. Engineers features (lag features, volume analysis)")
print("4. Trains ensemble model (Random Forest + Linear Regression)")
print("5. Saves model to models/stock_predictor_{symbol}.pkl")

print("\n📊 What happens during training:")
print("- Downloads ~2500 days of stock data")
print("- Creates 15+ technical features")
print("- Splits data 80/20 for training/testing")
print("- Trains 2 models and combines predictions")
print("- Evaluates with R² score, MAE, RMSE")

print("\n🎯 Ready to train? Run:")
print("python train_model.py AAPL")
