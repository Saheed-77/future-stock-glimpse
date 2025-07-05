import { useState } from "react";
import { Navbar } from "@/components/Navbar";
import { CompanyProfile } from "@/components/CompanyProfile";
import { StockChart } from "@/components/StockChart";
import { ModelMetrics } from "@/components/ModelMetrics";
import { 
  mockCompanyData, 
  generateHistoricalData, 
  generatePredictionData, 
  mockModelMetrics 
} from "@/utils/mockData";
import { LineChart, Brain, TrendingUp } from "lucide-react";

interface Company {
  symbol: string;
  name: string;
  industry: string;
}

const Index = () => {
  const [selectedCompany, setSelectedCompany] = useState<Company | null>(null);

  const handleCompanySelect = (company: Company) => {
    setSelectedCompany(company);
  };

  const companyData = selectedCompany ? mockCompanyData[selectedCompany.symbol] : null;
  const historicalData = companyData ? generateHistoricalData(companyData.currentPrice - 10) : [];
  const predictionData = companyData ? generatePredictionData(companyData.currentPrice) : [];

  return (
    <div className="min-h-screen bg-background">
      <Navbar 
        onCompanySelect={handleCompanySelect}
        selectedCompany={selectedCompany}
      />

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        {!selectedCompany ? (
          // Welcome State
          <div className="flex flex-col items-center justify-center min-h-[60vh] text-center space-y-6">
            <div className="p-6 bg-gradient-primary rounded-full">
              <LineChart className="h-16 w-16 text-primary" />
            </div>
            <div className="space-y-3">
              <h2 className="text-2xl font-bold text-foreground">
                Welcome to Stock Price Predictor
              </h2>
              <p className="text-lg text-muted-foreground max-w-md">
                Get AI-powered stock price predictions and comprehensive market analysis. 
                Start by searching for a company above.
              </p>
            </div>
            <div className="flex items-center gap-4 text-sm text-muted-foreground">
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
        ) : (
          // Dashboard Content
          <div className="space-y-8">
            {/* Company Profile */}
            {companyData && (
              <CompanyProfile company={companyData} />
            )}

            {/* Charts Section */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <StockChart
                data={historicalData}
                title="Historical Price (30 Days)"
                color="hsl(217, 91%, 60%)"
              />
              <StockChart
                data={predictionData}
                title="AI Prediction (30 Days)"
                color="hsl(142, 86%, 28%)"
                isPrediction={true}
              />
            </div>

            {/* Model Metrics */}
            <ModelMetrics {...mockModelMetrics} />
          </div>
        )}
      </main>
    </div>
  );
};

export default Index;
