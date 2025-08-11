// API Service for Stock Prediction Backend
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000/api';

export interface StockData {
  date: string;
  price: number;
  volume?: number;
  high?: number;
  low?: number;
  open?: number;
}

export interface PredictionData {
  date: string;
  price: number;
  confidence?: number;
  day?: number;
}

export interface CompanyInfo {
  symbol: string;
  name: string;
  sector: string;
  industry: string;
  currentPrice: number;
  marketCap: number;
  volume: number;
  averageVolume: number;
  dayHigh: number;
  dayLow: number;
  fiftyTwoWeekHigh: number;
  fiftyTwoWeekLow: number;
  dividendYield: number;
  peRatio: number;
  beta: number;
  description: string;
}

export interface ModelMetrics {
  mse: number;
  mae: number;
  accuracy: number;
  rmse: number;
  r2_score?: number;
  testSamples?: number;
  lastUpdated: string;
  // Mock data compatibility
  modelName?: string;
  confidenceScore?: number;
  predictionRange?: string;
  lastTrained?: string;
  features?: string[];
}

export interface PopularStock {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
}

class StockAPIService {
  private baseURL: string;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  async healthCheck(): Promise<{ status: string; message: string }> {
    try {
      const response = await fetch(`${this.baseURL}/health`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Health check failed:', error);
      throw new Error('Backend API is not available');
    }
  }

  async getHistoricalData(symbol: string, period: string = '1y'): Promise<StockData[]> {
    try {
      const response = await fetch(`${this.baseURL}/stock/${symbol}/historical?period=${period}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      
      if (!data.success) {
        throw new Error(data.error || 'Failed to fetch historical data');
      }
      
      return data.data;
    } catch (error) {
      console.error(`Error fetching historical data for ${symbol}:`, error);
      throw error;
    }
  }

  async predictStockPrice(symbol: string, days: number = 30): Promise<PredictionData[]> {
    try {
      const response = await fetch(`${this.baseURL}/stock/${symbol}/predict?days=${days}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      
      if (!data.success) {
        throw new Error(data.error || 'Failed to generate predictions');
      }
      
      return data.predictions;
    } catch (error) {
      console.error(`Error predicting stock price for ${symbol}:`, error);
      throw error;
    }
  }

  async trainModel(symbol: string, epochs: number = 50, batchSize: number = 32): Promise<void> {
    try {
      const response = await fetch(`${this.baseURL}/stock/${symbol}/train`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          epochs,
          batch_size: batchSize,
        }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (!data.success) {
        throw new Error(data.error || 'Failed to train model');
      }
    } catch (error) {
      console.error(`Error training model for ${symbol}:`, error);
      throw error;
    }
  }

  async getModelMetrics(symbol: string): Promise<ModelMetrics> {
    try {
      const response = await fetch(`${this.baseURL}/stock/${symbol}/metrics`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      
      if (!data.success) {
        throw new Error(data.error || 'Failed to fetch model metrics');
      }
      
      return data.metrics;
    } catch (error) {
      console.error(`Error fetching model metrics for ${symbol}:`, error);
      throw error;
    }
  }

  async getCompanyInfo(symbol: string): Promise<CompanyInfo> {
    try {
      const response = await fetch(`${this.baseURL}/stock/${symbol}/company-info`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      
      if (!data.success) {
        throw new Error(data.error || 'Failed to fetch company info');
      }
      
      return data.company;
    } catch (error) {
      console.error(`Error fetching company info for ${symbol}:`, error);
      throw error;
    }
  }

  async searchStocks(query: string): Promise<Array<{ symbol: string; name: string; industry: string }>> {
    try {
      const response = await fetch(`${this.baseURL}/stocks/search?q=${encodeURIComponent(query)}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      
      if (!data.success) {
        throw new Error(data.error || 'Failed to search stocks');
      }
      
      return data.results;
    } catch (error) {
      console.error(`Error searching stocks for "${query}":`, error);
      throw error;
    }
  }

  async getPopularStocks(): Promise<PopularStock[]> {
    try {
      const response = await fetch(`${this.baseURL}/stocks/popular`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      
      if (!data.success) {
        throw new Error(data.error || 'Failed to fetch popular stocks');
      }
      
      return data.stocks;
    } catch (error) {
      console.error('Error fetching popular stocks:', error);
      throw error;
    }
  }

  // Utility method to check if backend is available
  async isBackendAvailable(): Promise<boolean> {
    try {
      await this.healthCheck();
      return true;
    } catch {
      return false;
    }
  }
}

// Create and export a singleton instance
export const stockAPI = new StockAPIService();

// Export the class for custom instances
export default StockAPIService;
