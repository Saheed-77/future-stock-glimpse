import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { ThemeToggle } from "@/components/ThemeToggle";
import { CompanySearch } from "@/components/CompanySearch";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { stockAPI, type PopularStock } from "@/services/stockAPI";
import { 
  ArrowLeft,
  Search,
  TrendingUp,
  TrendingDown,
  Building2,
  Filter,
  SortAsc,
  SortDesc,
  Eye,
  Activity,
  DollarSign,
  BarChart3,
  Sparkles,
  Brain,
  Wifi,
  WifiOff
} from "lucide-react";

interface Company {
  symbol: string;
  name: string;
  industry: string;
}

interface CompanyDetails extends PopularStock {
  name?: string;
  industry?: string;
  marketCap?: number;
  volume?: number;
}

const Companies = () => {
  const navigate = useNavigate();
  const [companies, setCompanies] = useState<CompanyDetails[]>([]);
  const [filteredCompanies, setFilteredCompanies] = useState<CompanyDetails[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedIndustry, setSelectedIndustry] = useState<string>("all");
  const [sortBy, setSortBy] = useState<"symbol" | "price" | "change">("symbol");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("asc");
  const [useRealData, setUseRealData] = useState(false);

  // Extended list of companies with industry information
  const companyData = [
    { symbol: "AAPL", name: "Apple Inc.", industry: "Technology" },
    { symbol: "GOOGL", name: "Alphabet Inc.", industry: "Technology" },
    { symbol: "MSFT", name: "Microsoft Corporation", industry: "Technology" },
    { symbol: "AMZN", name: "Amazon.com Inc.", industry: "Consumer Discretionary" },
    { symbol: "TSLA", name: "Tesla Inc.", industry: "Automotive" },
    { symbol: "META", name: "Meta Platforms Inc.", industry: "Technology" },
    { symbol: "NVDA", name: "NVIDIA Corporation", industry: "Technology" },
    { symbol: "NFLX", name: "Netflix Inc.", industry: "Entertainment" },
    { symbol: "ADBE", name: "Adobe Inc.", industry: "Technology" },
    { symbol: "CRM", name: "Salesforce Inc.", industry: "Technology" },
    { symbol: "ORCL", name: "Oracle Corporation", industry: "Technology" },
    { symbol: "IBM", name: "IBM Corporation", industry: "Technology" },
    { symbol: "JPM", name: "JPMorgan Chase & Co.", industry: "Financial Services" },
    { symbol: "BAC", name: "Bank of America Corp.", industry: "Financial Services" },
    { symbol: "WMT", name: "Walmart Inc.", industry: "Consumer Staples" },
    { symbol: "KO", name: "The Coca-Cola Company", industry: "Consumer Staples" },
    { symbol: "PG", name: "Procter & Gamble Co.", industry: "Consumer Staples" },
    { symbol: "JNJ", name: "Johnson & Johnson", industry: "Healthcare" },
    { symbol: "PFE", name: "Pfizer Inc.", industry: "Healthcare" },
    { symbol: "UNH", name: "UnitedHealth Group Inc.", industry: "Healthcare" }
  ];

  useEffect(() => {
    loadCompanies();
  }, []);

  useEffect(() => {
    filterAndSortCompanies();
  }, [companies, searchTerm, selectedIndustry, sortBy, sortOrder]);

  const loadCompanies = async () => {
    try {
      const isBackendAvailable = await stockAPI.isBackendAvailable();
      setUseRealData(isBackendAvailable);
      
      if (isBackendAvailable) {
        // Get popular stocks first
        const popularStocks = await stockAPI.getPopularStocks();
        
        // Merge with company data
        const enrichedCompanies = companyData.map(company => {
          const stockData = popularStocks.find(stock => stock.symbol === company.symbol);
          return {
            ...company,
            price: stockData?.price || generateMockPrice(company.symbol),
            change: stockData?.change || generateMockChange(),
            changePercent: stockData?.changePercent || generateMockChangePercent(),
            marketCap: generateMockMarketCap(),
            volume: generateMockVolume()
          };
        });
        
        setCompanies(enrichedCompanies);
      } else {
        // Use mock data for all companies
        const mockCompanies = companyData.map(company => ({
          ...company,
          price: generateMockPrice(company.symbol),
          change: generateMockChange(),
          changePercent: generateMockChangePercent(),
          marketCap: generateMockMarketCap(),
          volume: generateMockVolume()
        }));
        
        setCompanies(mockCompanies);
      }
    } catch (error) {
      console.error('Error loading companies:', error);
      // Fallback to mock data
      const mockCompanies = companyData.map(company => ({
        ...company,
        price: generateMockPrice(company.symbol),
        change: generateMockChange(),
        changePercent: generateMockChangePercent(),
        marketCap: generateMockMarketCap(),
        volume: generateMockVolume()
      }));
      setCompanies(mockCompanies);
    } finally {
      setIsLoading(false);
    }
  };

  const generateMockPrice = (symbol: string): number => {
    const basePrice = { 
      AAPL: 185, GOOGL: 145, MSFT: 415, AMZN: 155, TSLA: 245,
      META: 320, NVDA: 480, NFLX: 450, ADBE: 580, CRM: 240,
      ORCL: 120, IBM: 140, JPM: 160, BAC: 35, WMT: 160,
      KO: 60, PG: 155, JNJ: 160, PFE: 35, UNH: 520
    }[symbol] || 100;
    return basePrice + (Math.random() - 0.5) * 20;
  };

  const generateMockChange = (): number => (Math.random() - 0.5) * 10;
  const generateMockChangePercent = (): number => (Math.random() - 0.5) * 5;
  const generateMockMarketCap = (): number => Math.random() * 2000000000000 + 100000000000;
  const generateMockVolume = (): number => Math.random() * 50000000 + 10000000;

  const filterAndSortCompanies = () => {
    let filtered = companies.filter(company => {
      const matchesSearch = company.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           company.name?.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesIndustry = selectedIndustry === "all" || company.industry === selectedIndustry;
      return matchesSearch && matchesIndustry;
    });

    // Sort companies
    filtered.sort((a, b) => {
      let aValue, bValue;
      
      switch (sortBy) {
        case "price":
          aValue = a.price;
          bValue = b.price;
          break;
        case "change":
          aValue = a.changePercent;
          bValue = b.changePercent;
          break;
        default:
          aValue = a.symbol;
          bValue = b.symbol;
      }

      if (sortOrder === "asc") {
        return aValue > bValue ? 1 : -1;
      } else {
        return aValue < bValue ? 1 : -1;
      }
    });

    setFilteredCompanies(filtered);
  };

  const getUniqueIndustries = () => {
    const industries = [...new Set(companies.map(c => c.industry))];
    return industries.sort();
  };

  const formatMarketCap = (marketCap: number): string => {
    if (marketCap >= 1000000000000) {
      return `$${(marketCap / 1000000000000).toFixed(1)}T`;
    } else if (marketCap >= 1000000000) {
      return `$${(marketCap / 1000000000).toFixed(1)}B`;
    } else {
      return `$${(marketCap / 1000000).toFixed(1)}M`;
    }
  };

  const formatVolume = (volume: number): string => {
    if (volume >= 1000000) {
      return `${(volume / 1000000).toFixed(1)}M`;
    } else if (volume >= 1000) {
      return `${(volume / 1000).toFixed(1)}K`;
    } else {
      return volume.toString();
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="bg-gradient-card border-b border-border shadow-card">
        <div className="container mx-auto px-4 py-6">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
            <div className="flex items-center gap-4">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => navigate("/")}
                className="text-muted-foreground hover:text-foreground"
              >
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back to Home
              </Button>
              <div className="flex items-center gap-4">
                <div className="relative p-3 bg-gradient-accent rounded-xl shadow-lg">
                  <Building2 className="h-8 w-8 text-white" />
                  <div className="absolute -top-1 -right-1 w-3 h-3 bg-chart-secondary rounded-full border-2 border-background"></div>
                </div>
                <div>
                  <h1 className="text-3xl font-bold text-foreground">All Companies</h1>
                  <p className="text-muted-foreground">Explore and analyze market leaders</p>
                </div>
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
                onCompanySelect={(company) => navigate(`/stock/${company.symbol.toLowerCase()}`)}
              />
              <ThemeToggle />
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        {/* Filters and Search */}
        <div className="mb-8 space-y-4">
          <div className="flex flex-col md:flex-row gap-4 items-start md:items-center justify-between">
            {/* Search */}
            <div className="relative flex-1 max-w-md">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search companies..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10"
              />
            </div>

            {/* Filters */}
            <div className="flex flex-wrap gap-4 items-center">
              {/* Industry Filter */}
              <div className="flex items-center gap-2">
                <Filter className="h-4 w-4 text-muted-foreground" />
                <select
                  value={selectedIndustry}
                  onChange={(e) => setSelectedIndustry(e.target.value)}
                  className="bg-background border border-border rounded-md px-3 py-2 text-sm"
                >
                  <option value="all">All Industries</option>
                  {getUniqueIndustries().map(industry => (
                    <option key={industry} value={industry}>{industry}</option>
                  ))}
                </select>
              </div>

              {/* Sort */}
              <div className="flex items-center gap-2">
                <select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value as "symbol" | "price" | "change")}
                  className="bg-background border border-border rounded-md px-3 py-2 text-sm"
                >
                  <option value="symbol">Sort by Symbol</option>
                  <option value="price">Sort by Price</option>
                  <option value="change">Sort by Change</option>
                </select>
                
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setSortOrder(sortOrder === "asc" ? "desc" : "asc")}
                >
                  {sortOrder === "asc" ? <SortAsc className="h-4 w-4" /> : <SortDesc className="h-4 w-4" />}
                </Button>
              </div>
            </div>
          </div>

          {/* Stats */}
          <div className="flex flex-wrap gap-4 text-sm text-muted-foreground">
            <span>Showing {filteredCompanies.length} of {companies.length} companies</span>
            {selectedIndustry !== "all" && (
              <Badge variant="secondary">{selectedIndustry}</Badge>
            )}
          </div>
        </div>

        {/* Companies Grid */}
        {isLoading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {[...Array(12)].map((_, i) => (
              <Card key={i} className="bg-gradient-card border-border shadow-card">
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <Skeleton className="h-6 w-16" />
                    <Skeleton className="h-8 w-8 rounded-full" />
                  </div>
                  <Skeleton className="h-4 w-32" />
                </CardHeader>
                <CardContent className="space-y-3">
                  <Skeleton className="h-6 w-20" />
                  <Skeleton className="h-4 w-24" />
                  <div className="flex justify-between">
                    <Skeleton className="h-4 w-16" />
                    <Skeleton className="h-4 w-12" />
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {filteredCompanies.map((company) => (
              <Card 
                key={company.symbol}
                className="cursor-pointer bg-gradient-card border-border shadow-card hover:shadow-elevated transition-all duration-300 transform hover:scale-105 group"
                onClick={() => navigate(`/stock/${company.symbol.toLowerCase()}`)}
              >
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <div className="space-y-1">
                      <CardTitle className="text-lg font-bold text-foreground group-hover:text-primary transition-colors">
                        {company.symbol}
                      </CardTitle>
                      <p className="text-sm text-muted-foreground truncate">{company.name}</p>
                    </div>
                    <div className={`p-2 rounded-full ${company.change >= 0 ? 'bg-chart-secondary/10' : 'bg-chart-danger/10'}`}>
                      {company.change >= 0 ? (
                        <TrendingUp className="h-4 w-4 text-chart-secondary" />
                      ) : (
                        <TrendingDown className="h-4 w-4 text-chart-danger" />
                      )}
                    </div>
                  </div>
                  <Badge variant="outline" className="w-fit text-xs">
                    {company.industry}
                  </Badge>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <DollarSign className="h-4 w-4 text-muted-foreground" />
                      <span className="text-xl font-bold text-foreground">${company.price.toFixed(2)}</span>
                    </div>
                    <div className={`text-sm font-medium ${company.change >= 0 ? 'text-chart-secondary' : 'text-chart-danger'}`}>
                      {company.change >= 0 ? '+' : ''}{company.change.toFixed(2)} ({company.change >= 0 ? '+' : ''}{company.changePercent.toFixed(2)}%)
                    </div>
                  </div>
                  
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <div className="flex items-center gap-1">
                      <BarChart3 className="h-3 w-3" />
                      <span>{formatMarketCap(company.marketCap || 0)}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Activity className="h-3 w-3" />
                      <span>{formatVolume(company.volume || 0)}</span>
                    </div>
                  </div>
                  
                  <div className="pt-2 border-t border-border">
                    <Button size="sm" variant="ghost" className="w-full text-xs group-hover:bg-primary/10">
                      <Eye className="h-3 w-3 mr-1" />
                      View Analysis
                      <Sparkles className="h-3 w-3 ml-1 opacity-0 group-hover:opacity-100 transition-opacity" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}

        {/* Empty State */}
        {!isLoading && filteredCompanies.length === 0 && (
          <div className="text-center py-12">
            <div className="p-6 bg-muted/20 rounded-full w-fit mx-auto mb-4">
              <Search className="h-12 w-12 text-muted-foreground" />
            </div>
            <h3 className="text-lg font-semibold text-foreground mb-2">No companies found</h3>
            <p className="text-muted-foreground mb-4">
              Try adjusting your search terms or filters.
            </p>
            <Button 
              variant="outline" 
              onClick={() => {
                setSearchTerm("");
                setSelectedIndustry("all");
              }}
            >
              Clear Filters
            </Button>
          </div>
        )}
      </main>
    </div>
  );
};

export default Companies;