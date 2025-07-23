#!/usr/bin/env python3
"""
Simple yfinance test
"""

import yfinance as yf
from datetime import datetime

print("Testing yfinance...")

try:
    # Test basic functionality
    ticker = yf.Ticker('AAPL')
    
    # Test 1: Get recent data
    data = ticker.history(period='5d')
    print(f"Test 1 - History: {len(data)} rows")
    if not data.empty:
        print(f"  Latest price: ${data['Close'].iloc[-1]:.2f}")
        print(f"  Date range: {data.index[0].date()} to {data.index[-1].date()}")
    
    # Test 2: Get info
    info = ticker.info
    print(f"Test 2 - Info: {len(info) if info else 0} fields")
    if info:
        print(f"  Company: {info.get('shortName', 'Unknown')}")
        print(f"  Sector: {info.get('sector', 'Unknown')}")
    
    # Test 3: Multiple symbols
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    print("Test 3 - Multiple symbols:")
    for symbol in symbols:
        try:
            t = yf.Ticker(symbol)
            h = t.history(period='1d')
            if not h.empty:
                price = h['Close'].iloc[-1]
                print(f"  {symbol}: ${price:.2f}")
            else:
                print(f"  {symbol}: No data")
        except Exception as e:
            print(f"  {symbol}: Error - {e}")
    
    print("\n✅ yfinance tests completed!")
    
except Exception as e:
    print(f"❌ yfinance test failed: {e}")
    import traceback
    traceback.print_exc()
