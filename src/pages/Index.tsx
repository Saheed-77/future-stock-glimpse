import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { CompanySearch } from "@/components/CompanySearch";
import { ThemeToggle } from "@/components/ThemeToggle";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { 
  LineChart, 
  Brain, 
  TrendingUp, 
  Target, 
  Clock, 
  BarChart3,
  ArrowRight,
  TrendingDown,
  Sparkles
} from "lucide-react";

interface Company {
  symbol: string;
  name: string;
  industry: string;
}

const Index = () => {
  const navigate = useNavigate();
  const [selectedCompany, setSelectedCompany] = useState<Company | null>(null);

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
          <div className="flex flex-col items-center justify-center min-h-[60vh] text-center space-y-8">
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-primary/20 to-chart-secondary/20 rounded-full blur-2xl"></div>
              <div className="relative p-8 bg-gradient-accent rounded-full shadow-xl">
                <LineChart className="h-16 w-16 text-white" />
              </div>
            </div>
            
            <div className="space-y-4">
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 text-primary text-sm font-medium">
                <Sparkles className="h-4 w-4" />
                <span>AI-Powered Predictions</span>
              </div>
              <h2 className="text-4xl md:text-5xl font-bold text-foreground">
                Welcome to Future Stock Glimpse
              </h2>
              <p className="text-xl text-muted-foreground max-w-2xl">
                Get AI-powered stock price predictions and comprehensive market analysis. 
                Start by searching for a company above.
              </p>
            </div>
            
            <div className="flex flex-col sm:flex-row items-center gap-6 pt-4">
              <Button size="lg" className="bg-gradient-accent hover:opacity-90 text-white px-8 shadow-lg">
                Get Started
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
              <div className="flex items-center gap-6 text-sm text-muted-foreground">
                <div className="flex items-center gap-2">
                  <TrendingUp className="h-4 w-4 text-chart-secondary" />
                  <span>Real-time Analysis</span>
                </div>
                <div className="flex items-center gap-2">
                  <Brain className="h-4 w-4 text-primary" />
                  <span>ML Predictions</span>
                </div>
              </div>
            </div>
          </div>

          {/* Key Features */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto">
            <Card className="text-center p-6 bg-gradient-card border-border shadow-card hover:shadow-elevated transition-all duration-300">
              <div className="p-3 bg-primary/10 rounded-lg w-fit mx-auto mb-4">
                <Target className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-lg font-semibold text-foreground mb-2">Accurate Predictions</h3>
              <p className="text-sm text-muted-foreground">95% accuracy rate with advanced ML algorithms</p>
            </Card>
            
            <Card className="text-center p-6 bg-gradient-card border-border shadow-card hover:shadow-elevated transition-all duration-300">
              <div className="p-3 bg-chart-secondary/10 rounded-lg w-fit mx-auto mb-4">
                <Clock className="h-6 w-6 text-chart-secondary" />
              </div>
              <h3 className="text-lg font-semibold text-foreground mb-2">Real-time Data</h3>
              <p className="text-sm text-muted-foreground">Live market data and instant analysis</p>
            </Card>
            
            <Card className="text-center p-6 bg-gradient-card border-border shadow-card hover:shadow-elevated transition-all duration-300">
              <div className="p-3 bg-chart-accent/10 rounded-lg w-fit mx-auto mb-4">
                <BarChart3 className="h-6 w-6 text-chart-accent" />
              </div>
              <h3 className="text-lg font-semibold text-foreground mb-2">Visual Analytics</h3>
              <p className="text-sm text-muted-foreground">Interactive charts and comprehensive insights</p>
            </Card>
          </div>

          {/* Quick Stats */}
          <div className="bg-muted/30 rounded-2xl p-8">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8 text-center">
              <div className="space-y-2">
                <div className="text-3xl font-bold text-foreground">95%</div>
                <div className="text-sm text-muted-foreground">Prediction Accuracy</div>
              </div>
              <div className="space-y-2">
                <div className="text-3xl font-bold text-foreground">500+</div>
                <div className="text-sm text-muted-foreground">Companies Analyzed</div>
              </div>
              <div className="space-y-2">
                <div className="text-3xl font-bold text-foreground">10K+</div>
                <div className="text-sm text-muted-foreground">Predictions Made</div>
              </div>
            </div>
          </div>

          {/* Popular Stocks Preview */}
          <div className="space-y-6">
            <div className="text-center space-y-2">
              <h3 className="text-2xl font-bold text-foreground">Popular Stocks</h3>
              <p className="text-muted-foreground">Click on any stock to see AI predictions</p>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {[
                { symbol: "AAPL", name: "Apple Inc.", price: 185.64, change: +2.34, changePercent: +1.28 },
                { symbol: "GOOGL", name: "Alphabet Inc.", price: 2847.29, change: -15.42, changePercent: -0.54 },
                { symbol: "MSFT", name: "Microsoft Corp.", price: 428.73, change: +8.91, changePercent: +2.12 },
                { symbol: "TSLA", name: "Tesla Inc.", price: 219.85, change: +12.67, changePercent: +6.11 }
              ].map((stock) => (
                <Card 
                  key={stock.symbol} 
                  className="cursor-pointer bg-gradient-card border-border shadow-card hover:shadow-elevated transition-all duration-300 transform hover:scale-105"
                  onClick={() => navigate(`/stock/${stock.symbol.toLowerCase()}`)}
                >
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between mb-2">
                      <div className="font-bold text-foreground">{stock.symbol}</div>
                      <div className={`p-2 rounded-full ${stock.change >= 0 ? 'bg-chart-secondary/10' : 'bg-chart-danger/10'}`}>
                        {stock.change >= 0 ? (
                          <TrendingUp className="h-4 w-4 text-chart-secondary" />
                        ) : (
                          <TrendingDown className="h-4 w-4 text-chart-danger" />
                        )}
                      </div>
                    </div>
                    <div className="text-sm text-muted-foreground mb-2">{stock.name}</div>
                    <div className="text-lg font-bold text-foreground">${stock.price.toFixed(2)}</div>
                    <div className={`text-xs font-medium ${stock.change >= 0 ? 'text-chart-secondary' : 'text-chart-danger'}`}>
                      {stock.change >= 0 ? '+' : ''}{stock.change.toFixed(2)} ({stock.change >= 0 ? '+' : ''}{stock.changePercent.toFixed(2)}%)
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Index;
