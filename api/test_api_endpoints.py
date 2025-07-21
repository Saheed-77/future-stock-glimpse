#!/usr/bin/env python3
"""
Test the API endpoints with requests
"""

import requests
import json

def test_endpoint(url, description):
    """Test a single endpoint"""
    print(f"\n🧪 {description}")
    print(f"URL: {url}")
    print("-" * 40)
    
    try:
        response = requests.get(url, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Success!")
            #print(data)
            # Show key information
            if 'source' in data:
                print(f"📊 Data Source: {data['source']}")
            
            if 'count' in data:
                print(f"📊 Data Count: {data['count']}")
            
            if 'stocks' in data:
                real_time = sum(1 for s in data['stocks'] if s.get('source') == 'real-time')
                fallback = sum(1 for s in data['stocks'] if s.get('source') == 'fallback')
                print(f"📊 Stocks: {real_time} real-time, {fallback} fallback")
                
                # Show first stock as example
                if data['stocks']:
                    stock = data['stocks'][0]
                    print(f"📊 Example: {stock['symbol']} = ${stock['price']} ({stock.get('source', 'unknown')})")
            
            if 'company' in data:
                company = data['company']
                print(f"📊 Company: {company.get('name', 'Unknown')} - ${company.get('currentPrice', 'N/A')}")
            
            return True
            
        else:
            print(f"❌ Failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    base_url = "http://localhost:5000"
    
    print("🚀 Testing API Endpoints for Real Data")
    print("=" * 50)
    
    # Test endpoints that should show real data
    endpoints = [
        (f"{base_url}/api/health", "Health Check - Check capabilities"),
        (f"{base_url}/api/stock/AAPL/historical?period=30d", "AAPL Historical - Check source"),
        (f"{base_url}/api/stock/AAPL/company-info", "AAPL Company Info - Check source"),
        (f"{base_url}/api/stocks/popular", "Popular Stocks - Check sources"),
    ]
    
    success_count = 0
    for url, description in endpoints:
        if test_endpoint(url, description):
            success_count += 1
    
    print(f"\n📊 Summary: {success_count}/{len(endpoints)} endpoints successful")

if __name__ == "__main__":
    main()
