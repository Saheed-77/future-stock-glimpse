#!/usr/bin/env python3
"""
Advanced Stock Prediction without TensorFlow
Uses statistical methods and technical indicators
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf
from datetime import datetime, timedelta

def calculate_technical_indicators(data):
    """Calculate technical indicators for stock analysis"""
    df = data.copy()
    
    # Simple Moving Averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # Exponential Moving Average
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    
    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=10).mean()
    df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
    
    # Price momentum
    df['Price_momentum'] = df['Close'] / df['Close'].shift(5)
    
    # Volatility
    df['Volatility'] = df['Close'].pct_change().rolling(window=10).std()
    
    return df

def advanced_stock_prediction(symbol, days=30):
    """
    Advanced stock prediction using multiple ML models and technical analysis
    """
    try:
        print(f"[ADVANCED] Starting prediction for {symbol}")
        
        # Fetch 2 years of data for better model training
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        
        # Get stock data
        data = yf.download(symbol, start=start_date, end=end_date)
        
        if data.empty:
            print(f"[ADVANCED] No data available for {symbol}")
            return None
        
        print(f"[ADVANCED] Got {len(data)} days of data for {symbol}")
        
        # Calculate technical indicators
        data_with_indicators = calculate_technical_indicators(data)
        
        # Prepare features for prediction
        feature_columns = [
            'Open', 'High', 'Low', 'Volume',
            'SMA_5', 'SMA_10', 'SMA_20',
            'EMA_12', 'EMA_26', 'MACD', 'MACD_signal',
            'RSI', 'BB_middle', 'BB_upper', 'BB_lower',
            'Volume_ratio', 'Price_momentum', 'Volatility'
        ]
        
        # Remove NaN values
        clean_data = data_with_indicators.dropna()
        
        if len(clean_data) < 50:
            print(f"[ADVANCED] Insufficient clean data for {symbol}")
            return None
        
        # Prepare training data
        X = clean_data[feature_columns].values
        y = clean_data['Close'].values
        
        # Use last 80% for training, 20% for validation
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        # Train multiple models
        models = {
            'LinearRegression': LinearRegression(),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        model_predictions = {}
        model_scores = {}
        
        for name, model in models.items():
            print(f"[ADVANCED] Training {name} model...")
            model.fit(X_train_scaled, y_train_scaled)
            
            # Validate model
            val_pred_scaled = model.predict(X_val_scaled)
            val_pred = scaler_y.inverse_transform(val_pred_scaled.reshape(-1, 1)).flatten()
            
            # Calculate accuracy (inverse of mean absolute percentage error)
            mape = np.mean(np.abs((y_val - val_pred) / y_val)) * 100
            accuracy = max(0, 100 - mape) / 100
            model_scores[name] = accuracy
            
            print(f"[ADVANCED] {name} validation accuracy: {accuracy:.3f}")
            
            # Make future predictions
            last_features = X_val_scaled[-1].reshape(1, -1)
            future_pred_scaled = model.predict(last_features)
            future_pred = scaler_y.inverse_transform(future_pred_scaled.reshape(-1, 1))[0][0]
            
            model_predictions[name] = future_pred
        
        # Ensemble prediction (weighted average based on model performance)
        total_weight = sum(model_scores.values())
        if total_weight > 0:
            ensemble_prediction = sum(
                pred * (score / total_weight) 
                for pred, score in zip(model_predictions.values(), model_scores.values())
            )
        else:
            ensemble_prediction = np.mean(list(model_predictions.values()))
        
        # Current price for reference
        current_price = float(clean_data['Close'].iloc[-1])
        
        # Generate trend for multiple days
        base_change = (ensemble_prediction - current_price) / current_price
        
        predictions = []
        for i in range(days):
            future_date = datetime.now() + timedelta(days=i+1)
            
            # Apply diminishing trend over time
            trend_factor = 1 + (base_change * (0.9 ** i))  # Diminishing effect
            predicted_price = current_price * trend_factor
            
            # Add some realistic volatility based on historical data
            volatility = clean_data['Volatility'].iloc[-10:].mean()
            noise_factor = np.random.normal(0, volatility * 0.5)  # Reduced noise
            predicted_price *= (1 + noise_factor)
            
            # Calculate confidence based on model agreement and time horizon
            model_agreement = 1 - (np.std(list(model_predictions.values())) / np.mean(list(model_predictions.values())))
            time_decay = 0.95 ** i  # Confidence decreases over time
            confidence = max(0.6, min(0.95, model_agreement * time_decay))
            
            predictions.append({
                'date': future_date.strftime('%Y-%m-%d'),
                'price': round(predicted_price, 2),
                'confidence': round(confidence, 3)
            })
        
        result = {
            'predictions': predictions,
            'current_price': round(current_price, 2),
            'next_day_prediction': round(ensemble_prediction, 2),
            'model_scores': {k: round(v, 3) for k, v in model_scores.items()},
            'change': round(ensemble_prediction - current_price, 2),
            'change_percent': round(((ensemble_prediction - current_price) / current_price) * 100, 2),
            'method': 'Advanced ML (Linear Regression + Random Forest)',
            'features_used': len(feature_columns),
            'data_points': len(clean_data)
        }
        
        print(f"[ADVANCED] Prediction completed for {symbol}")
        print(f"[ADVANCED] Next day: ${ensemble_prediction:.2f} (change: {result['change_percent']:.2f}%)")
        
        return result
        
    except Exception as e:
        print(f"[ADVANCED] Error in advanced prediction for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_advanced_prediction():
    """Test the advanced prediction function"""
    print("Testing Advanced Stock Prediction (No TensorFlow)")
    print("=" * 50)
    
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    for symbol in symbols:
        print(f"\nTesting {symbol}...")
        result = advanced_stock_prediction(symbol, days=7)
        
        if result:
            print(f"✅ {symbol} prediction successful!")
            print(f"   Current: ${result['current_price']}")
            print(f"   Next day: ${result['next_day_prediction']} ({result['change_percent']:+.2f}%)")
            print(f"   Method: {result['method']}")
            print(f"   Features: {result['features_used']}")
            
            # Show first few predictions
            for i, pred in enumerate(result['predictions'][:3]):
                print(f"   Day {i+1}: ${pred['price']} (confidence: {pred['confidence']})")
        else:
            print(f"❌ {symbol} prediction failed")

if __name__ == "__main__":
    test_advanced_prediction()
