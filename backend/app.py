#!/usr/bin/env python3
"""
Simple Stock Prediction Backend API
Built from scratch - clean and reliable
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
import hashlib
import random
import yfinance as yf
import requests
import numpy as np
import threading
import os

# Try to import ML predictor, gracefully handle failure
try:
    from ml_predictor_v2 import StockPredictor, train_model_for_symbol
    ML_AVAILABLE = True
    print("✅ ML Predictor loaded successfully")
except Exception as e:
    print(f"⚠️  ML Predictor not available: {e}")
    ML_AVAILABLE = False
    StockPredictor = None
    train_model_for_symbol = None

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global dictionary to store trained models
trained_models = {}
training_in_progress = set()

# Stock symbols that we support
SUPPORTED_SYMBOLS = [
    'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 
    'META', 'NVDA', 'NFLX', 'ADBE', 'CRM'
]

def get_or_train_model(symbol):
    """Get a trained model for the symbol, training if necessary"""
    if not ML_AVAILABLE:
        return None
        
    symbol = symbol.upper()
    
    if symbol in trained_models:
        return trained_models[symbol]
    
    # Check if model file exists
    predictor = StockPredictor(symbol)
    if predictor.load_models():
        # Also need to load recent data for predictions
        if predictor.fetch_training_data(years=1):  # Load recent data for context
            trained_models[symbol] = predictor
            return predictor
    
    return None

def train_model_async(symbol):
    """Train model in background"""
    if not ML_AVAILABLE:
        print(f"ML not available for training {symbol}")
        return
        
    try:
        print(f"Starting background training for {symbol}")
        predictor = train_model_for_symbol(symbol)
        if predictor:
            trained_models[symbol] = predictor
            print(f"Background training completed for {symbol}")
        else:
            print(f"Background training failed for {symbol}")
    except Exception as e:
        print(f"Error in background training for {symbol}: {e}")
    finally:
        training_in_progress.discard(symbol)

def get_real_time_data(symbol, period='1y'):
    """Get real-time stock data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        
        # Map our periods to yfinance periods
        period_map = {
            '30d': '1mo',
            '6m': '6mo', 
            '1y': '1y'
        }
        
        yf_period = period_map.get(period, '1y')
        
        # Get historical data
        hist = ticker.history(period=yf_period)
        
        if hist.empty:
            return None
            
        # Convert to our format
        data = []
        for date, row in hist.iterrows():
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'price': round(float(row['Close']), 2),
                'high': round(float(row['High']), 2),
                'low': round(float(row['Low']), 2),
                'volume': int(row['Volume'])
            })
        
        return data
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def get_company_info_real(symbol):
    """Get real company information from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        if not info:
            return None
            
        return {
            'symbol': symbol.upper(),
            'name': info.get('shortName', f'{symbol} Inc.'),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'currentPrice': round(float(info.get('currentPrice', 0)), 2),
            'marketCap': info.get('marketCap', 0),
            'volume': info.get('volume', 0),
            'dayHigh': round(float(info.get('dayHigh', 0)), 2),
            'dayLow': round(float(info.get('dayLow', 0)), 2),
            'fiftyTwoWeekHigh': round(float(info.get('fiftyTwoWeekHigh', 0)), 2),
            'fiftyTwoWeekLow': round(float(info.get('fiftyTwoWeekLow', 0)), 2),
            'peRatio': info.get('trailingPE', 0),
            'beta': info.get('beta', 0),
            'description': info.get('longBusinessSummary', f'{symbol} is a publicly traded company.')
        }
        
    except Exception as e:
        print(f"Error fetching company info for {symbol}: {e}")
        return None

def get_fallback_data(symbol, period='1y'):
    """Fallback to mock data if real data fails"""
    print(f"Using fallback data for {symbol}")
    
    # Use the original mock data generation
    period_mapping = {
        '30d': 30,
        '6m': 180,
        '1y': 365
    }
    
    days = period_mapping.get(period, 365)
    base_price = STOCK_PRICES.get(symbol.upper(), 100.0)
    
    data = []
    today = datetime.now()
    
    for i in range(days):
        date = today - timedelta(days=days-i)
        date_str = date.strftime('%Y-%m-%d')
        price = get_consistent_price(symbol, date_str)
        
        # Add some daily variations
        seed = hashlib.md5(f"{symbol}_{date_str}_daily".encode()).hexdigest()[:8]
        random.seed(int(seed, 16))
        
        high = price * random.uniform(1.01, 1.03)
        low = price * random.uniform(0.97, 0.99)
        volume = random.randint(1000000, 10000000)
        
        data.append({
            'date': date_str,
            'price': price,
            'high': round(high, 2),
            'low': round(low, 2),
            'volume': volume
        })
    
    return data

# Stock base prices for fallback data
STOCK_PRICES = {
    'AAPL': 185.0,
    'GOOGL': 145.0, 
    'MSFT': 415.0,
    'AMZN': 155.0,
    'TSLA': 245.0,
    'META': 320.0,
    'NVDA': 480.0,
    'NFLX': 450.0,
    'ADBE': 580.0,
    'CRM': 240.0
}

def get_consistent_price(symbol, date_str):
    """Generate consistent price for symbol on specific date"""
    seed = hashlib.md5(f"{symbol}_{date_str}".encode()).hexdigest()[:8]
    random.seed(int(seed, 16))
    
    base = STOCK_PRICES.get(symbol.upper(), 100.0)
    variation = random.uniform(-0.1, 0.1)  # ±10%
    return round(base * (1 + variation), 2)

# ===== API ENDPOINTS =====

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'message': 'Stock Prediction API is running'
    })

@app.route('/api/stock/<symbol>/historical', methods=['GET'])  
def get_historical_data(symbol):
    """Get historical stock data - now with real-time data"""
    try:
        period = request.args.get('period', '1y')
        symbol = symbol.upper()
        
        print(f"Fetching real-time data for {symbol} ({period})")
        
        # Try to get real data first
        data = get_real_time_data(symbol, period)
        
        # If real data fails, use fallback
        if not data:
            print(f"Real data failed for {symbol}, using fallback")
            data = get_fallback_data(symbol, period)
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'data': data,
            'count': len(data),
            'period': period,
            'source': 'real-time' if data != get_fallback_data(symbol, period) else 'fallback'
        })
        
    except Exception as e:
        print(f"Error in get_historical_data: {e}")
        # Return fallback data on any error
        data = get_fallback_data(symbol.upper(), period)
        return jsonify({
            'success': True,
            'symbol': symbol.upper(),
            'data': data,
            'count': len(data),
            'period': period,
            'source': 'fallback'
        })

@app.route('/api/stock/<symbol>/predict', methods=['GET'])
def predict_stock(symbol):
    """Generate ML-based stock predictions"""
    try:
        days = int(request.args.get('days', 30))
        symbol = symbol.upper()
        
        if days > 90:
            return jsonify({
                'error': 'Maximum 90 days prediction supported',
                'symbol': symbol
            }), 400
        
        print(f"Generating ML predictions for {symbol} ({days} days)")
        
        # Try to get ML model
        predictor = get_or_train_model(symbol) if ML_AVAILABLE else None
        
        if predictor:
            # Use ML model for predictions
            print(f"Using ML model for {symbol} predictions")
            predictions = predictor.predict_future_prices(days)
            
            if predictions:
                current_price = predictor.get_current_price()
                return jsonify({
                    'success': True,
                    'symbol': symbol,
                    'predictions': predictions,
                    'count': len(predictions),
                    'basePrice': current_price,
                    'source': 'ml_model',
                    'model_info': {
                        'type': 'LSTM',
                        'training_years': 10,
                        'features': predictor.data.columns.tolist() if predictor.data is not None else []
                    }
                })
        
        # Fallback to statistical prediction if ML model not available
        print(f"ML model not available for {symbol}, using statistical fallback")
        
        # Start training in background if not already training
        if (ML_AVAILABLE and symbol not in training_in_progress and 
            symbol in SUPPORTED_SYMBOLS):
            training_in_progress.add(symbol)
            thread = threading.Thread(target=train_model_async, args=(symbol,))
            thread.daemon = True
            thread.start()
            print(f"Started background training for {symbol}")
        elif not ML_AVAILABLE:
            print(f"ML not available - using statistical prediction for {symbol}")
        
        # Generate statistical predictions as fallback
        try:
            ticker = yf.Ticker(symbol)
            current_price = ticker.history(period='1d')['Close'].iloc[-1]
            current_price = round(float(current_price), 2)
            print(f"Using real current price for statistical prediction: ${current_price}")
        except:
            today = datetime.now()
            current_price = get_consistent_price(symbol, today.strftime('%Y-%m-%d'))
            print(f"Using fallback current price for statistical prediction: ${current_price}")
        
        predictions = []
        for i in range(days):
            future_date = datetime.now() + timedelta(days=i+1)
            date_str = future_date.strftime('%Y-%m-%d')
            
            # Generate prediction with trend and volatility
            seed = hashlib.md5(f"{symbol}_{date_str}_stat_pred".encode()).hexdigest()[:8]
            random.seed(int(seed, 16))
            
            # More sophisticated statistical model
            trend = 1 + (i * 0.0005)  # 0.05% daily growth
            volatility = random.uniform(-0.03, 0.03)  # ±3% volatility
            seasonal_factor = 1 + 0.01 * np.sin(i * 0.1)  # Small seasonal component
            
            predicted_price = current_price * trend * (1 + volatility) * seasonal_factor
            
            # Confidence decreases with time
            confidence = max(0.5, 0.8 - (i * 0.01))
            
            predictions.append({
                'date': date_str,
                'price': round(predicted_price, 2),
                'confidence': round(confidence, 3)
            })
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'predictions': predictions,
            'count': len(predictions),
            'basePrice': current_price,
            'source': 'statistical_fallback',
            'ml_training_status': 'in_progress' if (ML_AVAILABLE and symbol in training_in_progress) else 'ml_not_available'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'symbol': symbol.upper()
        }), 500

@app.route('/api/stock/<symbol>/company-info', methods=['GET'])
def get_company_info(symbol):
    """Get company information - now with real-time data"""
    try:
        symbol = symbol.upper()
        print(f"Fetching real company info for {symbol}")
        
        # Try to get real company info
        company_info = get_company_info_real(symbol)
        
        if not company_info:
            print(f"Real company info failed for {symbol}, using fallback")
            # Fallback to mock data
            companies = {
                'AAPL': {
                    'name': 'Apple Inc.',
                    'sector': 'Technology',
                    'industry': 'Consumer Electronics',
                    'description': 'Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.'
                },
                'GOOGL': {
                    'name': 'Alphabet Inc.',
                    'sector': 'Communication Services', 
                    'industry': 'Internet Content & Information',
                    'description': 'Alphabet Inc. provides online advertising services in the United States, Europe, the Middle East, Africa, the Asia-Pacific, Canada, and Latin America.'
                },
                'MSFT': {
                    'name': 'Microsoft Corporation',
                    'sector': 'Technology',
                    'industry': 'Software—Infrastructure', 
                    'description': 'Microsoft Corporation develops, licenses, and supports software, services, devices, and solutions worldwide.'
                },
                'AMZN': {
                    'name': 'Amazon.com Inc.',
                    'sector': 'Consumer Discretionary',
                    'industry': 'Internet Retail',
                    'description': 'Amazon.com, Inc. engages in the retail sale of consumer products and subscriptions in North America and internationally.'
                },
                'TSLA': {
                    'name': 'Tesla Inc.',
                    'sector': 'Consumer Discretionary', 
                    'industry': 'Auto Manufacturers',
                    'description': 'Tesla, Inc. designs, develops, manufactures, leases, and sells electric vehicles, and energy generation and storage systems.'
                },
                'META': {
                    'name': 'Meta Platforms Inc.',
                    'sector': 'Communication Services',
                    'industry': 'Internet Content & Information',
                    'description': 'Meta Platforms, Inc. develops products that enable people to connect and share with friends and family through mobile devices, personal computers, virtual reality headsets, wearables, and in-home devices worldwide.'
                },
                'NVDA': {
                    'name': 'NVIDIA Corporation',
                    'sector': 'Technology',
                    'industry': 'Semiconductors',
                    'description': 'NVIDIA Corporation operates as a computing company in the United States, Taiwan, China, Hong Kong, and internationally.'
                },
                'NFLX': {
                    'name': 'Netflix Inc.',
                    'sector': 'Communication Services',
                    'industry': 'Entertainment',
                    'description': 'Netflix, Inc. provides entertainment services. It offers TV series, documentaries, feature films, and mobile games.'
                }
            }
            
            company = companies.get(symbol)
            if not company:
                return jsonify({
                    'error': f'Company info not found for {symbol}',
                    'symbol': symbol
                }), 404
            
            # Get current price (try real, fallback to mock)
            try:
                ticker = yf.Ticker(symbol)
                current_price = ticker.history(period='1d')['Close'].iloc[-1]
                current_price = round(float(current_price), 2)
            except:
                current_price = get_consistent_price(symbol, datetime.now().strftime('%Y-%m-%d'))
            
            # Generate mock metrics
            seed = hashlib.md5(f"{symbol}_metrics".encode()).hexdigest()[:8]
            random.seed(int(seed, 16))
            
            company_info = {
                'symbol': symbol,
                'name': company['name'],
                'sector': company['sector'],
                'industry': company['industry'],
                'currentPrice': current_price,
                'marketCap': random.randint(100000000000, 3000000000000),
                'volume': random.randint(10000000, 100000000),
                'dayHigh': current_price * random.uniform(1.01, 1.05),
                'dayLow': current_price * random.uniform(0.95, 0.99),
                'fiftyTwoWeekHigh': current_price * random.uniform(1.1, 1.4),
                'fiftyTwoWeekLow': current_price * random.uniform(0.6, 0.9),
                'peRatio': random.uniform(15, 35),
                'beta': random.uniform(0.8, 1.5),
                'description': company['description']
            }
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'company': company_info,
            'source': 'real-time' if get_company_info_real(symbol) else 'fallback'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'symbol': symbol.upper()
        }), 500

@app.route('/api/stock/<symbol>/metrics', methods=['GET'])
def get_metrics(symbol):
    """Get model metrics - now with real ML metrics"""
    try:
        symbol = symbol.upper()
        
        # Try to get ML model metrics
        predictor = get_or_train_model(symbol) if ML_AVAILABLE else None
        
        if predictor and predictor.metrics:
            print(f"Returning real ML metrics for {symbol}")
            return jsonify({
                'success': True,
                'symbol': symbol,
                'metrics': {
                    'accuracy': predictor.metrics['accuracy'] / 100,  # Convert to 0-1 range
                    'mse': predictor.metrics['mse'],
                    'mae': predictor.metrics['mae'],
                    'rmse': predictor.metrics['rmse'],
                    'r2_score': predictor.metrics['r2_score'],
                    'lastUpdated': predictor.metrics['last_updated'],
                    'modelName': predictor.metrics['model_name'],
                    'confidenceScore': min(0.95, predictor.metrics['r2_score']),
                    'predictionRange': '30 days',
                    'features': predictor.metrics['features'],
                    'trainingSamples': predictor.metrics['training_samples'],
                    'testSamples': predictor.metrics['test_samples']
                },
                'source': 'ml_model'
            })
        
        # Fallback to mock metrics
        print(f"Using mock metrics for {symbol}")
        seed = hashlib.md5(f"{symbol}_model_metrics".encode()).hexdigest()[:8]
        random.seed(int(seed, 16))
        
        metrics = {
            'accuracy': random.uniform(0.75, 0.95),
            'mse': random.uniform(0.001, 0.01),
            'mae': random.uniform(0.5, 2.0),
            'rmse': random.uniform(0.8, 3.0),
            'r2_score': random.uniform(0.7, 0.9),
            'lastUpdated': datetime.now().isoformat(),
            'modelName': 'Statistical',
            'confidenceScore': random.uniform(0.8, 0.95),
            'predictionRange': '30 days',
            'features': ['price', 'volume', 'moving_average', 'rsi']
        }
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'metrics': metrics,
            'source': 'mock_data',
            'ml_training_status': 'in_progress' if (ML_AVAILABLE and symbol in training_in_progress) else 'ml_not_available'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'symbol': symbol.upper()
        }), 500

@app.route('/api/stocks/popular', methods=['GET'])
def get_popular_stocks():
    """Get popular stocks with real-time prices"""
    try:
        stocks = []
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
        
        print("Fetching real-time data for popular stocks...")
        
        for symbol in symbols:
            try:
                # Try to get real data
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='2d')  # Get last 2 days
                
                if not hist.empty and len(hist) >= 2:
                    current_price = round(float(hist['Close'].iloc[-1]), 2)
                    prev_price = round(float(hist['Close'].iloc[-2]), 2)
                    change = round(current_price - prev_price, 2)
                    change_percent = round((change / prev_price) * 100, 2)
                    
                    stocks.append({
                        'symbol': symbol,
                        'price': current_price,
                        'change': change,
                        'changePercent': change_percent,
                        'source': 'real-time'
                    })
                else:
                    raise Exception("Insufficient data")
                    
            except Exception as e:
                print(f"Failed to get real data for {symbol}, using fallback: {e}")
                # Fallback to mock data
                today = datetime.now().strftime('%Y-%m-%d')
                yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                
                current_price = get_consistent_price(symbol, today)
                prev_price = get_consistent_price(symbol, yesterday)
                change = current_price - prev_price
                change_percent = (change / prev_price) * 100
                
                stocks.append({
                    'symbol': symbol,
                    'price': current_price,
                    'change': round(change, 2),
                    'changePercent': round(change_percent, 2),
                    'source': 'fallback'
                })
        
        return jsonify({
            'success': True,
            'stocks': stocks,
            'count': len(stocks)
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'stocks': []
        }), 500

@app.route('/api/stocks/search', methods=['GET'])
def search_stocks():
    """Search for stocks"""
    try:
        query = request.args.get('q', '').upper()
        
        if not query:
            return jsonify({
                'error': 'Query required',
                'results': []
            }), 400
        
        companies = {
            'AAPL': {'name': 'Apple Inc.', 'industry': 'Technology'},
            'GOOGL': {'name': 'Alphabet Inc.', 'industry': 'Technology'},
            'MSFT': {'name': 'Microsoft Corporation', 'industry': 'Technology'},
            'AMZN': {'name': 'Amazon.com Inc.', 'industry': 'Consumer Discretionary'},
            'TSLA': {'name': 'Tesla Inc.', 'industry': 'Consumer Discretionary'},
            'META': {'name': 'Meta Platforms Inc.', 'industry': 'Technology'},
            'NVDA': {'name': 'NVIDIA Corporation', 'industry': 'Technology'},
            'NFLX': {'name': 'Netflix Inc.', 'industry': 'Communication Services'}
        }
        
        results = []
        for symbol, info in companies.items():
            if query in symbol or query in info['name'].upper():
                results.append({
                    'symbol': symbol,
                    'name': info['name'],
                    'industry': info['industry']
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'query': query
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'results': []
        }), 500

@app.route('/api/stock/<symbol>/training-status', methods=['GET'])
def get_training_status(symbol):
    """Get ML model training status"""
    try:
        symbol = symbol.upper()
        
        # Check if model exists
        model_exists = os.path.exists(f"models/{symbol}_model.h5")
        
        # Check if currently training
        is_training = symbol in training_in_progress
        
        # Check if model is loaded in memory
        model_loaded = symbol in trained_models
        
        status = {
            'symbol': symbol,
            'model_exists': model_exists,
            'is_training': is_training,
            'model_loaded': model_loaded,
            'supported': symbol in SUPPORTED_SYMBOLS
        }
        
        if model_exists:
            # Get model file age
            model_path = f"models/{symbol}_model.h5"
            model_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(model_path))
            status['model_age_days'] = model_age.days
            status['needs_retraining'] = model_age.days > 7
        
        return jsonify({
            'success': True,
            'status': status
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'symbol': symbol.upper()
        }), 500

@app.route('/api/stock/<symbol>/train', methods=['POST'])
def trigger_training(symbol):
    """Trigger ML model training"""
    try:
        symbol = symbol.upper()
        
        if symbol not in SUPPORTED_SYMBOLS:
            return jsonify({
                'error': f'Symbol {symbol} not supported',
                'supported_symbols': SUPPORTED_SYMBOLS
            }), 400
        
        if symbol in training_in_progress:
            return jsonify({
                'success': True,
                'message': f'Training already in progress for {symbol}',
                'status': 'already_training'
            })
        
        # Start training in background
        training_in_progress.add(symbol)
        thread = threading.Thread(target=train_model_async, args=(symbol,))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': f'Started training for {symbol}',
            'status': 'training_started'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'symbol': symbol.upper()
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    print("🚀 Starting Clean Stock Prediction API with ML...")
    print("📍 Running on: http://localhost:5000")
    print("✅ All endpoints ready!")
    print(f"🤖 ML Available: {ML_AVAILABLE}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
