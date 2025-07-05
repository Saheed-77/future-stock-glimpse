// Mock data for the Stock Price Predictor dashboard

export interface StockDataPoint {
  date: string;
  price: number;
}

export interface CompanyData {
  symbol: string;
  name: string;
  industry: string;
  marketCap: string;
  currentPrice: number;
  priceChange: number;
  priceChangePercent: number;
  volume: string;
  peRatio: number;
  high52Week: number;
  low52Week: number;
}

// Generate mock historical data for the last 30 days
export const generateHistoricalData = (basePrice: number, days: number = 30): StockDataPoint[] => {
  const data: StockDataPoint[] = [];
  let currentPrice = basePrice;
  
  for (let i = days; i >= 0; i--) {
    const date = new Date();
    date.setDate(date.getDate() - i);
    
    // Add some realistic price movement
    const changePercent = (Math.random() - 0.5) * 0.06; // ±3% max daily change
    currentPrice = currentPrice * (1 + changePercent);
    
    data.push({
      date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      price: parseFloat(currentPrice.toFixed(2))
    });
  }
  
  return data;
};

// Generate mock prediction data for the next 30 days
export const generatePredictionData = (currentPrice: number, days: number = 30): StockDataPoint[] => {
  const data: StockDataPoint[] = [];
  let price = currentPrice;
  
  // Add slight upward trend for predictions
  const trendFactor = 0.002; // 0.2% daily trend
  
  for (let i = 1; i <= days; i++) {
    const date = new Date();
    date.setDate(date.getDate() + i);
    
    // Add trend and some volatility
    const dailyChange = trendFactor + (Math.random() - 0.5) * 0.04; // ±2% volatility
    price = price * (1 + dailyChange);
    
    data.push({
      date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      price: parseFloat(price.toFixed(2))
    });
  }
  
  return data;
};

// Mock company data
export const mockCompanyData: Record<string, CompanyData> = {
  'AAPL': {
    symbol: 'AAPL',
    name: 'Apple Inc.',
    industry: 'Technology',
    marketCap: '$2.8T',
    currentPrice: 185.92,
    priceChange: 2.34,
    priceChangePercent: 1.28,
    volume: '52.4M',
    peRatio: 28.5,
    high52Week: 199.62,
    low52Week: 124.17
  },
  'GOOGL': {
    symbol: 'GOOGL',
    name: 'Alphabet Inc.',
    industry: 'Technology',
    marketCap: '$1.7T',
    currentPrice: 138.45,
    priceChange: -1.23,
    priceChangePercent: -0.88,
    volume: '28.9M',
    peRatio: 24.2,
    high52Week: 151.55,
    low52Week: 83.34
  },
  'MSFT': {
    symbol: 'MSFT',
    name: 'Microsoft Corporation',
    industry: 'Technology',
    marketCap: '$2.9T',
    currentPrice: 415.26,
    priceChange: 5.67,
    priceChangePercent: 1.38,
    volume: '23.1M',
    peRatio: 32.1,
    high52Week: 468.35,
    low52Week: 309.45
  },
  'TSLA': {
    symbol: 'TSLA',
    name: 'Tesla Inc.',
    industry: 'Consumer Discretionary',
    marketCap: '$800B',
    currentPrice: 248.50,
    priceChange: -3.22,
    priceChangePercent: -1.28,
    volume: '89.3M',
    peRatio: 45.7,
    high52Week: 299.29,
    low52Week: 138.80
  },
  'NVDA': {
    symbol: 'NVDA',
    name: 'NVIDIA Corporation',
    industry: 'Technology',
    marketCap: '$1.8T',
    currentPrice: 731.25,
    priceChange: 12.45,
    priceChangePercent: 1.73,
    volume: '41.2M',
    peRatio: 58.3,
    high52Week: 950.02,
    low52Week: 180.96
  }
};

// Mock model metrics
export const mockModelMetrics = {
  modelName: 'LSTM Neural Network v2.1',
  accuracy: 87.3,
  confidenceScore: 79.2,
  predictionRange: '30 Days',
  lastTrained: '2024-01-15 14:30 UTC',
  features: ['Price History', 'Volume', 'Moving Averages', 'RSI', 'Market Sentiment', 'News Analysis']
};