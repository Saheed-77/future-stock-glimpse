import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { CompanyProfile } from "@/components/CompanyProfile";
import { StockChart } from "@/components/StockChart";
import { ModelMetrics } from "@/components/ModelMetrics";
import { CompanySearch } from "@/components/CompanySearch";
import { ThemeToggle } from "@/components/ThemeToggle";
import { Button } from "@/components/ui/button";
import { 
  mockCompanyData, 
  generateHistoricalData, 
  generatePredictionData, 
  mockModelMetrics 
} from "@/utils/mockData";
import { Brain, ArrowLeft } from "lucide-react";

interface Company {
  symbol: string;
  name: string;
  industry: string;
}

const StockPrediction = () => {
  const { symbol } = useParams<{ symbol: string }>();
  const navigate = useNavigate();
  const [selectedCompany, setSelectedCompany] = useState<Company | null>(null);

  useEffect(() => {
    if (symbol) {
      // Find company data based on symbol
      const companyData = mockCompanyData[symbol.toUpperCase()];
      if (companyData) {
        setSelectedCompany({
          symbol: symbol.toUpperCase(),
          name: companyData.name,
          industry: companyData.industry
        });
      } else {
        // If company not found, redirect to home
        navigate("/");
      }
    }
  }, [symbol, navigate]);

  const handleCompanySelect = (company: Company) => {
    setSelectedCompany(company);
    navigate(`/stock/${company.symbol.toLowerCase()}`);
  };

  const companyData = selectedCompany ? mockCompanyData[selectedCompany.symbol] : null;
  const historicalData = companyData ? generateHistoricalData(companyData.currentPrice - 10) : [];
  const predictionData = companyData ? generatePredictionData(companyData.currentPrice) : [];

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
        {!selectedCompany || !companyData ? (
          <div className="flex flex-col items-center justify-center min-h-[60vh] text-center space-y-6">
            <div className="text-2xl font-bold text-foreground">Company not found</div>
            <p className="text-muted-foreground">
              The requested stock symbol could not be found. Please search for a different company.
            </p>
            <Button onClick={() => navigate("/")} className="bg-gradient-accent hover:opacity-90 text-white">
              Return to Home
            </Button>
          </div>
        ) : (
          <div className="space-y-8">
            {/* Company Profile */}
            <CompanyProfile company={companyData} />

            {/* Charts Section */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <StockChart
                data={historicalData}
                title="Historical Price (30 Days)"
                variant="primary"
              />
              <StockChart
                data={predictionData}
                title="AI Prediction (30 Days)"
                variant="secondary"
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

export default StockPrediction;
