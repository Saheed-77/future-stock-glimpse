import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { CompanyProfile } from "@/components/CompanyProfile";
import { StockChart } from "@/components/StockChart";
import { ProfitLossChart } from "@/components/ProfitLossChart";
import { ModelMetrics } from "@/components/ModelMetrics";
import { MLPredictionDisplay } from "@/components/MLPredictionDisplay";
import { ModelTrainingStatus } from "@/components/ModelTrainingStatus";
import { CompanySearch } from "@/components/CompanySearch";
import { ThemeToggle } from "@/components/ThemeToggle";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { stockAPI, type StockData, type PredictionData, type CompanyInfo, type ModelMetrics as APIModelMetrics } from "@/services/stockAPI";
import { 
  mockCompanyData, 
  generateHistoricalData, 
  generatePredictionData, 
  mockModelMetrics 
} from "@/utils/mockData";
import { Brain, ArrowLeft, AlertCircle, Loader2, Wifi, WifiOff, BarChart3, TrendingUp, Activity } from "lucide-react";

interface Company {
  symbol: string;
  name: string;
  industry: string;
}

const StockPrediction = () => {
  const { symbol } = useParams<{ symbol: string }>();
  const navigate = useNavigate();
  const [selectedCompany, setSelectedCompany] = useState<Company | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [useRealData, setUseRealData] = useState(false);
  const [historicalData, setHistoricalData] = useState<StockData[]>([]);
  const [predictionData, setPredictionData] = useState<PredictionData[]>([]);
  const [companyInfo, setCompanyInfo] = useState<CompanyInfo | null>(null);
  const [modelMetrics, setModelMetrics] = useState<APIModelMetrics | null>(null);
  const [selectedPeriod, setSelectedPeriod] = useState<string>('1y');
  const [isLoadingChart, setIsLoadingChart] = useState(false);
  const [predictionSource, setPredictionSource] = useState<'ml_model' | 'statistical_fallback' | 'mock_data'>('mock_data');
  const [isModelTraining, setIsModelTraining] = useState(false);

  useEffect(() => {
    if (symbol) {
      initializeStock(symbol.toUpperCase());
    }
  }, [symbol]);

  const initializeStock = async (stockSymbol: string) => {
    setIsLoading(true);
    setError(null);
    
    try {
      // Check if backend is available
      const isBackendAvailable = await stockAPI.isBackendAvailable();
      setUseRealData(isBackendAvailable);
      
      if (isBackendAvailable) {
        // Load real data from API
        await loadRealData(stockSymbol);
      } else {
        // Fall back to mock data
        loadMockData(stockSymbol);
      }
    } catch (err) {
      console.error('Error initializing stock data:', err);
      setError(err instanceof Error ? err.message : 'Failed to load stock data');
      // Fall back to mock data
      loadMockData(stockSymbol);
    } finally {
      setIsLoading(false);
    }
  };

  const loadRealData = async (stockSymbol: string) => {
    try {
      // Load data in parallel
      const [historical, predictions, company, metrics] = await Promise.allSettled([
        stockAPI.getHistoricalData(stockSymbol, selectedPeriod),
        stockAPI.predictStockPrice(stockSymbol, 30),
        stockAPI.getCompanyInfo(stockSymbol),
        stockAPI.getModelMetrics(stockSymbol)
      ]);

      // Handle historical data
      if (historical.status === 'fulfilled') {
        setHistoricalData(historical.value);
      } else {
        console.error('Error loading historical data:', historical.reason);
      }

      // Handle predictions
      if (predictions.status === 'fulfilled') {
        const predResponse = predictions.value;
        setPredictionData(predResponse.predictions || predResponse);
        
        // Extract metadata if available
        if (predResponse.source) {
          setPredictionSource(predResponse.source);
        }
        if (predResponse.ml_training_status === 'in_progress') {
          setIsModelTraining(true);
        }
      } else {
        console.error('Error loading predictions:', predictions.reason);
      }

      // Handle company info
      if (company.status === 'fulfilled') {
        setCompanyInfo(company.value);
        setSelectedCompany({
          symbol: stockSymbol,
          name: company.value.name,
          industry: company.value.industry
        });
      } else {
        console.error('Error loading company info:', company.reason);
        // Fall back to mock data for company info
        const mockCompany = mockCompanyData[stockSymbol];
        if (mockCompany) {
          setSelectedCompany({
            symbol: stockSymbol,
            name: mockCompany.name,
            industry: mockCompany.industry
          });
        }
      }

      // Handle model metrics
      if (metrics.status === 'fulfilled') {
        setModelMetrics(metrics.value);
      } else {
        console.error('Error loading model metrics:', metrics.reason);
      }

    } catch (err) {
      console.error('Error loading real data:', err);
      throw err;
    }
  };

  const handlePeriodChange = async (period: string) => {
    if (!selectedCompany) return;
    
    setSelectedPeriod(period);
    setIsLoadingChart(true);
    
    try {
      if (useRealData) {
        const historical = await stockAPI.getHistoricalData(selectedCompany.symbol, period);
        setHistoricalData(historical);
      } else {
        // Generate mock data for the selected period
        const mockCompany = mockCompanyData[selectedCompany.symbol];
        if (mockCompany) {
          const days = period === '30d' ? 30 : period === '6m' ? 180 : 365;
          const mockHistorical = generateHistoricalData(mockCompany.currentPrice, days);
          setHistoricalData(mockHistorical);
        }
      }
    } catch (err) {
      console.error('Error loading data for period:', err);
    } finally {
      setIsLoadingChart(false);
    }
  };

  const loadMockData = (stockSymbol: string) => {
    const mockCompany = mockCompanyData[stockSymbol];
    if (mockCompany) {
      setSelectedCompany({
        symbol: stockSymbol,
        name: mockCompany.name,
        industry: mockCompany.industry
      });
      
      // Convert mock data to API format with selected period
      const days = selectedPeriod === '30d' ? 30 : selectedPeriod === '6m' ? 180 : 365;
      const mockHistorical = generateHistoricalData(mockCompany.currentPrice, days);
      const mockPredictions = generatePredictionData(mockCompany.currentPrice, 30);
      
      setHistoricalData(mockHistorical);
      setPredictionData(mockPredictions);
      setModelMetrics(mockModelMetrics);
      setPredictionSource('mock_data');
    } else {
      // If company not found, redirect to home
      navigate("/");
    }
  };

  const handleCompanySelect = (company: Company) => {
    setSelectedCompany(company);
    navigate(`/stock/${company.symbol.toLowerCase()}`);
  };

  const handleTrainingComplete = () => {
    setIsModelTraining(false);
    // Reload predictions to get the new ML model results
    if (selectedCompany) {
      loadRealData(selectedCompany.symbol);
    }
  };

  const handleRetry = () => {
    if (symbol) {
      initializeStock(symbol.toUpperCase());
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background">
        <div className="container mx-auto px-4 py-8">
          <div className="flex items-center justify-between mb-8">
            <div className="flex items-center gap-4">
              <Button 
                variant="outline" 
                size="sm" 
                onClick={() => navigate("/")}
              >
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back to Home
              </Button>
              <Skeleton className="h-8 w-48" />
            </div>
            <ThemeToggle />
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="lg:col-span-2 space-y-6">
              <Skeleton className="h-64 w-full" />
              <Skeleton className="h-64 w-full" />
              <Skeleton className="h-64 w-full" />
            </div>
            <div className="space-y-6">
              <Skeleton className="h-48 w-full" />
              <Skeleton className="h-48 w-full" />
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-background">
        <div className="container mx-auto px-4 py-8">
          <div className="flex items-center justify-between mb-8">
            <div className="flex items-center gap-4">
              <Button 
                variant="outline" 
                size="sm" 
                onClick={() => navigate("/")}
              >
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back to Home
              </Button>
            </div>
            <ThemeToggle />
          </div>
          
          <Alert className="max-w-2xl mx-auto">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription className="flex items-center justify-between">
              <span>{error}</span>
              <Button variant="outline" size="sm" onClick={handleRetry}>
                Try Again
              </Button>
            </AlertDescription>
          </Alert>
        </div>
      </div>
    );
  }

  if (!selectedCompany) {
    return (
      <div className="min-h-screen bg-background">
        <div className="container mx-auto px-4 py-8">
          <div className="flex items-center justify-between mb-8">
            <div className="flex items-center gap-4">
              <Button 
                variant="outline" 
                size="sm" 
                onClick={() => navigate("/")}
              >
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back to Home
              </Button>
              <h1 className="text-2xl font-bold">Stock not found</h1>
            </div>
            <ThemeToggle />
          </div>
          
          <Alert className="max-w-2xl mx-auto">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              The requested stock symbol was not found. Please try searching for a different company.
            </AlertDescription>
          </Alert>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="bg-gradient-card border-b border-border shadow-card">
        <div className="container mx-auto px-4 py-6">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
            <div className="flex items-center gap-4">
              <Button
                variant="ghost"
                onClick={() => navigate("/")}
                className="p-2 hover:bg-muted rounded-lg"
              >
                <ArrowLeft className="h-5 w-5" />
              </Button>
              <div className="relative p-3 bg-gradient-accent rounded-xl shadow-lg">
                <Brain className="h-8 w-8 text-white" />
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-chart-secondary rounded-full border-2 border-background"></div>
              </div>
              <div>
                <h1 className="text-3xl font-bold text-foreground">Future Stock Glimpse</h1>
                <p className="text-muted-foreground">AI-powered financial forecasting platform</p>
              </div>
            </div>
            <div className="flex items-center gap-6">
              <div className="flex items-center gap-2">
                {useRealData ? (
                  <div className="flex items-center gap-2 text-green-600">
                    <Wifi className="h-4 w-4" />
                    <span className="text-sm">Live Data</span>
                  </div>
                ) : (
                  <div className="flex items-center gap-2 text-amber-600">
                    <WifiOff className="h-4 w-4" />
                    <span className="text-sm">Demo Mode</span>
                  </div>
                )}
              </div>
              <CompanySearch 
                onCompanySelect={handleCompanySelect}
                selectedCompany={selectedCompany}
              />
              <ThemeToggle />
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        <div className="space-y-8">
          {/* Company Profile */}
          <CompanyProfile 
            company={companyInfo ? {
              symbol: companyInfo.symbol,
              name: companyInfo.name,
              industry: companyInfo.industry,
              marketCap: companyInfo.marketCap ? `$${(companyInfo.marketCap / 1000000000).toFixed(1)}B` : 'N/A',
              currentPrice: companyInfo.currentPrice,
              priceChange: 0, // We don't have this from API, could calculate from historical data
              priceChangePercent: 0, // We don't have this from API, could calculate from historical data
              volume: companyInfo.volume ? companyInfo.volume.toLocaleString() : 'N/A',
              peRatio: companyInfo.peRatio,
              high52Week: companyInfo.fiftyTwoWeekHigh,
              low52Week: companyInfo.fiftyTwoWeekLow
            } : mockCompanyData[selectedCompany.symbol]} 
          />

          {/* ML Training Status */}
          <ModelTrainingStatus 
            symbol={selectedCompany.symbol}
            onTrainingComplete={handleTrainingComplete}
          />

          {/* Enhanced ML Predictions */}
          <MLPredictionDisplay
            historicalData={historicalData}
            predictionData={predictionData}
            symbol={selectedCompany.symbol}
            modelSource={predictionSource}
            isTraining={isModelTraining}
          />

          {/* Analysis Tabs */}
          <Tabs defaultValue="charts" className="space-y-6">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="charts" className="flex items-center gap-2">
                <BarChart3 className="h-4 w-4" />
                Charts
              </TabsTrigger>
              <TabsTrigger value="analysis" className="flex items-center gap-2">
                <TrendingUp className="h-4 w-4" />
                Analysis
              </TabsTrigger>
              <TabsTrigger value="metrics" className="flex items-center gap-2">
                <Activity className="h-4 w-4" />
                Metrics
              </TabsTrigger>
            </TabsList>

            <TabsContent value="charts" className="space-y-6">
              {/* Time Period Filter */}
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-foreground">Historical Data</h3>
                <div className="flex items-center gap-2">
                  {['30d', '6m', '1y'].map((period) => (
                    <Button
                      key={period}
                      variant={selectedPeriod === period ? "default" : "outline"}
                      size="sm"
                      onClick={() => handlePeriodChange(period)}
                      disabled={isLoadingChart}
                      className="min-w-[60px]"
                    >
                      {isLoadingChart && selectedPeriod === period ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        period === '30d' ? '30D' : period === '6m' ? '6M' : '1Y'
                      )}
                    </Button>
                  ))}
                </div>
              </div>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <StockChart
                  data={historicalData}
                  title={`Historical Price (${selectedPeriod === '30d' ? '30 Days' : selectedPeriod === '6m' ? '6 Months' : '1 Year'})`}
                  variant="primary"
                />
                <StockChart
                  data={predictionData}
                  title="AI Prediction (30 Days)"
                  variant="secondary"
                  isPrediction={true}
                />
              </div>
            </TabsContent>

            <TabsContent value="analysis" className="space-y-6">
              {/* Profit/Loss Analysis */}
              <ProfitLossChart
                data={historicalData}
                title="Daily Price Changes - Profit/Loss Analysis"
                variant="primary"
              />
            </TabsContent>

            <TabsContent value="metrics" className="space-y-6">
              {/* Model Metrics */}
              <ModelMetrics {...(modelMetrics || mockModelMetrics)} />
            </TabsContent>
          </Tabs>
        </div>
      </main>
    </div>
  );
};

export default StockPrediction;
