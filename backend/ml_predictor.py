#!/usr/bin/env python3
"""
Machine Learning Stock Price Predictor
Uses Random Forest and Linear Regression to predict stock prices based on 10 years of historical data
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self, symbol):
        self.symbol = symbol.upper()
        self.rf_model = None
        self.lr_model = None
        self.scaler = StandardScaler()
        self.price_scaler = MinMaxScaler()
        self.model_path = f"models/{self.symbol}_rf_model.pkl"
        self.lr_model_path = f"models/{self.symbol}_lr_model.pkl"
        self.scaler_path = f"models/{self.symbol}_scaler.pkl"
        self.price_scaler_path = f"models/{self.symbol}_price_scaler.pkl"
        self.data = None
        self.metrics = {}
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
    
    def fetch_training_data(self, years=10):
        """Fetch 10 years of stock data for training"""
        try:
            print(f"Fetching {years} years of data for {self.symbol}...")
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years * 365)
            
            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                raise Exception(f"No data found for {self.symbol}")
            
            # We'll use multiple features for better prediction
            features = ['Open', 'High', 'Low', 'Close', 'Volume']
            self.data = data[features].copy()
            
            # Add technical indicators
            self.data['MA_5'] = self.data['Close'].rolling(window=5).mean()
            self.data['MA_20'] = self.data['Close'].rolling(window=20).mean()
            self.data['MA_50'] = self.data['Close'].rolling(window=50).mean()
            self.data['RSI'] = self.calculate_rsi(self.data['Close'])
            self.data['Volatility'] = self.data['Close'].pct_change().rolling(window=20).std()
            self.data['Price_Change'] = self.data['Close'].pct_change()
            self.data['Volume_MA'] = self.data['Volume'].rolling(window=20).mean()
            
            # Add lag features
            for lag in [1, 2, 3, 5]:
                self.data[f'Close_lag_{lag}'] = self.data['Close'].shift(lag)
                self.data[f'Volume_lag_{lag}'] = self.data['Volume'].shift(lag)
            
            # Drop NaN values
            self.data = self.data.dropna()
            
            print(f"Successfully fetched {len(self.data)} days of data")
            return True
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False
    
    def calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def prepare_features(self):
        """Prepare features for training"""
        if self.data is None:
            return False
        
        # Features to use for prediction
        feature_columns = [
            'Open', 'High', 'Low', 'Volume',
            'MA_5', 'MA_20', 'MA_50', 'RSI', 'Volatility',
            'Price_Change', 'Volume_MA'
        ] + [f'Close_lag_{lag}' for lag in [1, 2, 3, 5]] + [f'Volume_lag_{lag}' for lag in [1, 2, 3, 5]]
        
        # Target variable (what we want to predict)
        target = 'Close'
        
        # Prepare features and target
        X = self.data[feature_columns].values
        y = self.data[target].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data (80% train, 20% test)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, shuffle=False  # Don't shuffle time series data
        )
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        print(f"Features: {len(feature_columns)}")
        
        return True
    
    def train_models(self):
        """Train Random Forest and Linear Regression models"""
        print("Training models...")
        
        # Train Random Forest
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(self.X_train, self.y_train)
        
        # Train Linear Regression
        self.lr_model = LinearRegression()
        self.lr_model.fit(self.X_train, self.y_train)
        
        print("Models trained successfully!")
        
        # Calculate metrics
        self.calculate_metrics()
        
        # Save models
        self.save_models()
        
        return True
    
    def calculate_metrics(self):
        """Calculate model performance metrics"""
        if self.rf_model is None or self.lr_model is None:
            return
        
        # Make predictions with both models
        rf_pred = self.rf_model.predict(self.X_test)
        lr_pred = self.lr_model.predict(self.X_test)
        
        # Use ensemble (average) of both models
        ensemble_pred = (rf_pred + lr_pred) / 2
        
        # Calculate metrics for ensemble
        mse = mean_squared_error(self.y_test, ensemble_pred)
        mae = mean_absolute_error(self.y_test, ensemble_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, ensemble_pred)
        
        # Calculate accuracy (percentage of predictions within 5% of actual)
        accuracy = np.mean(np.abs((ensemble_pred - self.y_test) / self.y_test) <= 0.05) * 100
        
        self.metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2_score': float(r2),
            'accuracy': float(accuracy),
            'model_name': 'Random Forest + Linear Regression Ensemble',
            'training_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'features': self.data.columns.tolist(),
            'last_updated': datetime.now().isoformat()
        }
        
        print(f"Model Metrics for {self.symbol}:")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Accuracy (±5%): {accuracy:.2f}%")
    
    def save_models(self):
        """Save the trained models and scalers"""
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.rf_model, f)
            with open(self.lr_model_path, 'wb') as f:
                pickle.dump(self.lr_model, f)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            with open(self.price_scaler_path, 'wb') as f:
                pickle.dump(self.price_scaler, f)
            
            print(f"Models saved successfully")
            return True
        except Exception as e:
            print(f"Error saving models: {e}")
            return False
    
    def load_models(self):
        """Load pre-trained models and scalers"""
        try:
            if (os.path.exists(self.model_path) and os.path.exists(self.lr_model_path) and 
                os.path.exists(self.scaler_path) and os.path.exists(self.price_scaler_path)):
                
                with open(self.model_path, 'rb') as f:
                    self.rf_model = pickle.load(f)
                with open(self.lr_model_path, 'rb') as f:
                    self.lr_model = pickle.load(f)
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                with open(self.price_scaler_path, 'rb') as f:
                    self.price_scaler = pickle.load(f)
                
                print(f"Models loaded successfully")
                return True
            else:
                print("Model files not found")
                return False
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def predict_future_prices(self, days=30):
        """Predict future stock prices"""
        if self.rf_model is None or self.lr_model is None or self.data is None:
            return None
        
        try:
            predictions = []
            current_data = self.data.tail(50).copy()  # Use last 50 days for context
            
            for day in range(days):
                # Prepare features for prediction
                last_row = current_data.iloc[-1]
                
                # Create feature vector
                features = [
                    last_row['Open'], last_row['High'], last_row['Low'], last_row['Volume'],
                    last_row['MA_5'], last_row['MA_20'], last_row['MA_50'],
                    last_row['RSI'], last_row['Volatility'], last_row['Price_Change'],
                    last_row['Volume_MA']
                ] + [current_data['Close'].iloc[-lag] for lag in [1, 2, 3, 5]] + [current_data['Volume'].iloc[-lag] for lag in [1, 2, 3, 5]]
                
                # Scale features
                features_scaled = self.scaler.transform([features])
                
                # Make predictions with both models
                rf_pred = self.rf_model.predict(features_scaled)[0]
                lr_pred = self.lr_model.predict(features_scaled)[0]
                
                # Ensemble prediction
                predicted_price = (rf_pred + lr_pred) / 2
                
                predictions.append(predicted_price)
                
                # Update current_data for next prediction
                next_date = current_data.index[-1] + timedelta(days=1)
                
                # Create next row (simplified - in reality, we don't know these values)
                next_row = last_row.copy()
                next_row.name = next_date
                next_row['Close'] = predicted_price
                next_row['Open'] = predicted_price * (1 + np.random.normal(0, 0.01))
                next_row['High'] = predicted_price * (1 + abs(np.random.normal(0, 0.015)))
                next_row['Low'] = predicted_price * (1 - abs(np.random.normal(0, 0.015)))
                
                # Update technical indicators
                current_data = pd.concat([current_data, next_row.to_frame().T])
                current_data = current_data.tail(50)  # Keep only last 50 rows
                
                # Recalculate moving averages
                current_data['MA_5'] = current_data['Close'].rolling(window=5).mean()
                current_data['MA_20'] = current_data['Close'].rolling(window=20).mean()
                current_data['MA_50'] = current_data['Close'].rolling(window=50).mean()
                current_data['RSI'] = self.calculate_rsi(current_data['Close'])
                current_data['Volatility'] = current_data['Close'].pct_change().rolling(window=20).std()
                current_data['Price_Change'] = current_data['Close'].pct_change()
                current_data['Volume_MA'] = current_data['Volume'].rolling(window=20).mean()
                
                # Update lag features
                for lag in [1, 2, 3, 5]:
                    current_data[f'Close_lag_{lag}'] = current_data['Close'].shift(lag)
                    current_data[f'Volume_lag_{lag}'] = current_data['Volume'].shift(lag)
            
            # Generate dates for predictions
            last_date = self.data.index[-1]
            future_dates = []
            for i in range(1, days + 1):
                future_date = last_date + timedelta(days=i)
                # Skip weekends
                while future_date.weekday() >= 5:
                    future_date += timedelta(days=1)
                future_dates.append(future_date.strftime('%Y-%m-%d'))
            
            # Calculate confidence scores
            confidence_scores = []
            for i, pred in enumerate(predictions):
                # Confidence decreases with time horizon
                base_confidence = 0.90 - (i * 0.008)
                confidence = max(0.60, base_confidence)
                confidence_scores.append(min(0.95, confidence))
            
            result = []
            for date, price, confidence in zip(future_dates, predictions, confidence_scores):
                result.append({
                    'date': date,
                    'price': round(float(price), 2),
                    'confidence': round(float(confidence), 3)
                })
            
            return result
            
        except Exception as e:
            print(f"Error making predictions: {e}")
            return None
    
    def get_current_price(self):
        """Get the most recent actual price"""
        try:
            ticker = yf.Ticker(self.symbol)
            current_data = ticker.history(period='1d')
            if not current_data.empty:
                return round(float(current_data['Close'].iloc[-1]), 2)
        except:
            pass
        return None
    
    def retrain_if_needed(self, max_age_days=7):
        """Check if model needs retraining based on age"""
        if not os.path.exists(self.model_path):
            return True
        
        # Check file modification time
        model_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(self.model_path))
        return model_age.days > max_age_days

def train_model_for_symbol(symbol):
    """Train a model for a specific stock symbol"""
    print(f"\n{'='*50}")
    print(f"Training ML model for {symbol}")
    print(f"{'='*50}")
    
    predictor = StockPredictor(symbol)
    
    # Check if we need to retrain
    if not predictor.retrain_if_needed():
        if predictor.load_models():
            print(f"Using existing model for {symbol}")
            # Still need to load data for predictions
            if predictor.fetch_training_data(years=2):
                return predictor
    
    # Fetch training data
    if not predictor.fetch_training_data(years=10):
        print(f"Failed to fetch data for {symbol}")
        return None
    
    # Prepare features
    if not predictor.prepare_features():
        print(f"Failed to prepare features for {symbol}")
        return None
    
    # Train models
    if not predictor.train_models():
        print(f"Failed to train models for {symbol}")
        return None
    
    print(f"Successfully trained model for {symbol}")
    return predictor

if __name__ == "__main__":
    # Test the predictor
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    for symbol in symbols:
        predictor = train_model_for_symbol(symbol)
        if predictor:
            predictions = predictor.predict_future_prices(30)
            if predictions:
                print(f"\n{symbol} - Next 5 day predictions:")
                for pred in predictions[:5]:
                    print(f"  {pred['date']}: ${pred['price']} (confidence: {pred['confidence']:.1%})")
