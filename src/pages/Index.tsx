import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { CompanySearch } from "@/components/CompanySearch";
import { ThemeToggle } from "@/components/ThemeToggle";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { stockAPI, type PopularStock } from "@/services/stockAPI";
import { 
  LineChart, 
  Brain, 
  TrendingUp, 
  Target, 
  Clock, 
  BarChart3,
  ArrowRight,
  TrendingDown,
  Sparkles,
  Wifi,
  WifiOff,
  Building2,
  Eye,
  Globe,
  Shield,
  Zap,
  Users,
  Award,
  ChevronRight
} from "lucide-react";

interface Company {
  symbol: string;
  name: string;
  industry: string;
}

const Index = () => {
  const navigate = useNavigate();
  const [selectedCompany, setSelectedCompany] = useState<Company | null>(null);
  const [popularStocks, setPopularStocks] = useState<PopularStock[]>([]);
  const [isLoadingStocks, setIsLoadingStocks] = useState(true);
  const [useRealData, setUseRealData] = useState(false);

  useEffect(() => {
    loadPopularStocks();
  }, []);

  const loadPopularStocks = async () => {
    try {
      const isBackendAvailable = await stockAPI.isBackendAvailable();
      setUseRealData(isBackendAvailable);
      
      if (isBackendAvailable) {
        const stocks = await stockAPI.getPopularStocks();
        setPopularStocks(stocks);
      } else {
        // Fall back to mock data
        const mockStocks = [
          { symbol: "AAPL", price: 185.64, change: 2.34, changePercent: 1.28 },
          { symbol: "GOOGL", price: 2847.29, change: -15.42, changePercent: -0.54 },
          { symbol: "MSFT", price: 428.73, change: 8.91, changePercent: 2.12 },
          { symbol: "TSLA", price: 219.85, change: 12.67, changePercent: 6.11 }
        ];
        setPopularStocks(mockStocks);
      }
    } catch (error) {
      console.error('Error loading popular stocks:', error);
      // Fall back to mock data
      const mockStocks = [
        { symbol: "AAPL", price: 185.64, change: 2.34, changePercent: 1.28 },
        { symbol: "GOOGL", price: 2847.29, change: -15.42, changePercent: -0.54 },
        { symbol: "MSFT", price: 428.73, change: 8.91, changePercent: 2.12 },
        { symbol: "TSLA", price: 219.85, change: 12.67, changePercent: 6.11 }
      ];
      setPopularStocks(mockStocks);
    } finally {
      setIsLoadingStocks(false);
    }
  };

  const handleCompanySelect = (company: Company) => {
    setSelectedCompany(company);
    navigate(`/stock/${company.symbol.toLowerCase()}`);
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="bg-gradient-card border-b border-border shadow-card">
        <div className="container mx-auto px-4 py-6">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
            <div className="flex items-center gap-4">
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
        {/* Enhanced Welcome State */}
        <div className="space-y-16">
          {/* Hero Section */}
          <div className="flex flex-col items-center justify-center min-h-[70vh] text-center space-y-8">
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-primary/20 to-chart-secondary/20 rounded-full blur-3xl animate-pulse"></div>
              <div className="relative p-8 bg-gradient-accent rounded-full shadow-2xl">
                <Brain className="h-16 w-16 text-white" />
              </div>
            </div>
            
            <div className="space-y-6">
              <div className="inline-flex items-center gap-2 px-6 py-3 rounded-full bg-gradient-to-r from-primary/10 to-chart-secondary/10 border border-primary/20 text-primary text-sm font-medium">
                <Sparkles className="h-4 w-4" />
                <span>Powered by Advanced AI & Machine Learning</span>
              </div>
              <h1 className="text-5xl md:text-7xl font-bold text-foreground leading-tight">
                Future Stock
                <span className="bg-gradient-to-r from-primary to-chart-secondary bg-clip-text text-transparent"> Glimpse</span>
              </h1>
              <p className="text-xl md:text-2xl text-muted-foreground max-w-3xl leading-relaxed">
                Unlock the power of AI-driven stock predictions with real-time market analysis, 
                comprehensive insights, and precision forecasting for smarter investment decisions.
              </p>
            </div>
            
            <div className="flex flex-col sm:flex-row items-center gap-6 pt-6">
              <Button 
                size="lg" 
                className="bg-gradient-accent hover:opacity-90 text-white px-8 py-4 text-lg shadow-xl transition-all duration-300 hover:scale-105"
                onClick={() => navigate("/companies")}
              >
                Explore Companies
                <Building2 className="ml-2 h-5 w-5" />
              </Button>
              <Button 
                size="lg" 
                variant="outline" 
                className="px-8 py-4 text-lg border-2 hover:bg-muted/50 transition-all duration-300"
              >
                View Demo
                <Eye className="ml-2 h-5 w-5" />
              </Button>
            </div>
            
            <div className="flex items-center gap-8 pt-4 text-sm text-muted-foreground">
              <div className="flex items-center gap-2">
                <Globe className="h-4 w-4 text-chart-secondary" />
                <span>Global Markets</span>
              </div>
              <div className="flex items-center gap-2">
                <Shield className="h-4 w-4 text-primary" />
                <span>Secure & Reliable</span>
              </div>
              <div className="flex items-center gap-2">
                <Zap className="h-4 w-4 text-chart-accent" />
                <span>Real-time Data</span>
              </div>
            </div>
          </div>

          {/* Key Features */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-6xl mx-auto">
            <Card className="text-center p-8 bg-gradient-card border-border shadow-card hover:shadow-elevated transition-all duration-300 transform hover:scale-105 group">
              <div className="p-4 bg-gradient-to-br from-primary/20 to-primary/10 rounded-xl w-fit mx-auto mb-4 group-hover:from-primary/30 group-hover:to-primary/20 transition-all duration-300">
                <Target className="h-8 w-8 text-primary" />
              </div>
              <h3 className="text-xl font-semibold text-foreground mb-3">Precise Predictions</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">Advanced LSTM and ML algorithms delivering 95%+ accuracy in stock price forecasting</p>
            </Card>
            
            <Card className="text-center p-8 bg-gradient-card border-border shadow-card hover:shadow-elevated transition-all duration-300 transform hover:scale-105 group">
              <div className="p-4 bg-gradient-to-br from-chart-secondary/20 to-chart-secondary/10 rounded-xl w-fit mx-auto mb-4 group-hover:from-chart-secondary/30 group-hover:to-chart-secondary/20 transition-all duration-300">
                <Clock className="h-8 w-8 text-chart-secondary" />
              </div>
              <h3 className="text-xl font-semibold text-foreground mb-3">Real-time Analysis</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">Live market data processing with instant technical analysis and trend detection</p>
            </Card>
            
            <Card className="text-center p-8 bg-gradient-card border-border shadow-card hover:shadow-elevated transition-all duration-300 transform hover:scale-105 group">
              <div className="p-4 bg-gradient-to-br from-chart-accent/20 to-chart-accent/10 rounded-xl w-fit mx-auto mb-4 group-hover:from-chart-accent/30 group-hover:to-chart-accent/20 transition-all duration-300">
                <BarChart3 className="h-8 w-8 text-chart-accent" />
              </div>
              <h3 className="text-xl font-semibold text-foreground mb-3">Visual Insights</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">Interactive dashboards with comprehensive charts and analytical visualizations</p>
            </Card>

            <Card className="text-center p-8 bg-gradient-card border-border shadow-card hover:shadow-elevated transition-all duration-300 transform hover:scale-105 group">
              <div className="p-4 bg-gradient-to-br from-chart-danger/20 to-chart-danger/10 rounded-xl w-fit mx-auto mb-4 group-hover:from-chart-danger/30 group-hover:to-chart-danger/20 transition-all duration-300">
                <Brain className="h-8 w-8 text-chart-danger" />
              </div>
              <h3 className="text-xl font-semibold text-foreground mb-3">AI Intelligence</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">Machine learning models continuously learning from market patterns and behaviors</p>
            </Card>
          </div>

          {/* Platform Stats */}
          <div className="bg-gradient-to-r from-muted/50 to-muted/30 rounded-3xl p-12 backdrop-blur-sm border border-border/50">
            <div className="text-center mb-8">
              <h3 className="text-2xl font-bold text-foreground mb-2">Trusted by Investors Worldwide</h3>
              <p className="text-muted-foreground">Our platform delivers consistent results across global markets</p>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8 text-center">
              <div className="space-y-3">
                <div className="text-4xl font-bold bg-gradient-to-r from-primary to-chart-secondary bg-clip-text text-transparent">95%</div>
                <div className="text-sm font-medium text-muted-foreground">Prediction Accuracy</div>
                <Award className="h-5 w-5 text-primary mx-auto" />
              </div>
              <div className="space-y-3">
                <div className="text-4xl font-bold bg-gradient-to-r from-chart-secondary to-chart-accent bg-clip-text text-transparent">500+</div>
                <div className="text-sm font-medium text-muted-foreground">Companies Tracked</div>
                <Building2 className="h-5 w-5 text-chart-secondary mx-auto" />
              </div>
              <div className="space-y-3">
                <div className="text-4xl font-bold bg-gradient-to-r from-chart-accent to-chart-danger bg-clip-text text-transparent">100K+</div>
                <div className="text-sm font-medium text-muted-foreground">Predictions Generated</div>
                <TrendingUp className="h-5 w-5 text-chart-accent mx-auto" />
              </div>
              <div className="space-y-3">
                <div className="text-4xl font-bold bg-gradient-to-r from-chart-danger to-primary bg-clip-text text-transparent">10K+</div>
                <div className="text-sm font-medium text-muted-foreground">Active Users</div>
                <Users className="h-5 w-5 text-chart-danger mx-auto" />
              </div>
            </div>
          </div>

          {/* Popular Stocks Preview */}
          <div className="space-y-8">
            <div className="text-center space-y-4">
              <h3 className="text-3xl font-bold text-foreground">Market Leaders</h3>
              <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
                Explore top-performing stocks with real-time data and AI-powered predictions
              </p>
              <Button 
                variant="outline" 
                size="lg"
                onClick={() => navigate("/companies")}
                className="mt-4 group"
              >
                View All Companies
                <ChevronRight className="ml-2 h-4 w-4 group-hover:translate-x-1 transition-transform" />
              </Button>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {isLoadingStocks ? (
                // Loading skeletons
                [...Array(4)].map((_, i) => (
                  <Card key={i} className="bg-gradient-card border-border shadow-card">
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between mb-2">
                        <Skeleton className="h-4 w-12" />
                        <Skeleton className="h-8 w-8 rounded-full" />
                      </div>
                      <Skeleton className="h-3 w-24 mb-2" />
                      <Skeleton className="h-6 w-20 mb-1" />
                      <Skeleton className="h-3 w-16" />
                    </CardContent>
                  </Card>
                ))
              ) : (
                // Actual stock data
                popularStocks.map((stock) => (
                  <Card 
                    key={stock.symbol} 
                    className="cursor-pointer bg-gradient-card border-border shadow-card hover:shadow-elevated transition-all duration-300 transform hover:scale-105 group relative overflow-hidden"
                    onClick={() => navigate(`/stock/${stock.symbol.toLowerCase()}`)}
                  >
                    <div className="absolute top-0 right-0 w-20 h-20 bg-gradient-to-br from-primary/10 to-transparent rounded-bl-full opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                    <CardContent className="p-6 relative">
                      <div className="flex items-center justify-between mb-4">
                        <div className="space-y-1">
                          <div className="font-bold text-lg text-foreground group-hover:text-primary transition-colors">{stock.symbol}</div>
                          <div className="text-xs text-muted-foreground uppercase tracking-wider">Stock</div>
                        </div>
                        <div className={`p-3 rounded-full transition-all duration-300 ${stock.change >= 0 ? 'bg-chart-secondary/10 group-hover:bg-chart-secondary/20' : 'bg-chart-danger/10 group-hover:bg-chart-danger/20'}`}>
                          {stock.change >= 0 ? (
                            <TrendingUp className="h-5 w-5 text-chart-secondary" />
                          ) : (
                            <TrendingDown className="h-5 w-5 text-chart-danger" />
                          )}
                        </div>
                      </div>
                      <div className="space-y-2">
                        <div className="text-2xl font-bold text-foreground">${stock.price.toFixed(2)}</div>
                        <div className={`text-sm font-semibold flex items-center gap-1 ${stock.change >= 0 ? 'text-chart-secondary' : 'text-chart-danger'}`}>
                          <span>{stock.change >= 0 ? '+' : ''}{stock.change.toFixed(2)}</span>
                          <span>({stock.change >= 0 ? '+' : ''}{stock.changePercent.toFixed(2)}%)</span>
                        </div>
                      </div>
                      <div className="mt-4 pt-4 border-t border-border opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                        <div className="flex items-center justify-between text-xs text-muted-foreground">
                          <span>View Analysis</span>
                          <ArrowRight className="h-3 w-3" />
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Index;
