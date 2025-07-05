import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { TrendingUp, TrendingDown, Building, DollarSign } from "lucide-react";

interface CompanyData {
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

interface CompanyProfileProps {
  company: CompanyData;
}

export const CompanyProfile = ({ company }: CompanyProfileProps) => {
  const isPositiveChange = company.priceChange >= 0;

  return (
    <Card className="bg-gradient-card border-border shadow-card">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-primary/10 rounded-lg">
              <Building className="h-6 w-6 text-primary" />
            </div>
            <div>
              <CardTitle className="text-2xl font-bold text-foreground">{company.symbol}</CardTitle>
              <p className="text-muted-foreground">{company.name}</p>
            </div>
          </div>
          <Badge variant="secondary" className="bg-surface-elevated">
            {company.industry}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Price Information */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <DollarSign className="h-5 w-5 text-chart-primary" />
              <span className="text-sm font-medium text-muted-foreground">Current Price</span>
            </div>
            <div className="flex items-end gap-3">
              <span className="text-3xl font-bold text-foreground">
                ${company.currentPrice.toFixed(2)}
              </span>
              <div className={`flex items-center gap-1 ${isPositiveChange ? 'text-chart-secondary' : 'text-chart-danger'}`}>
                {isPositiveChange ? (
                  <TrendingUp className="h-4 w-4" />
                ) : (
                  <TrendingDown className="h-4 w-4" />
                )}
                <span className="font-medium">
                  {isPositiveChange ? '+' : ''}{company.priceChange.toFixed(2)} 
                  ({isPositiveChange ? '+' : ''}{company.priceChangePercent.toFixed(2)}%)
                </span>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="text-center p-3 bg-surface-elevated rounded-lg">
              <div className="text-sm text-muted-foreground">Market Cap</div>
              <div className="text-lg font-semibold text-foreground">{company.marketCap}</div>
            </div>
            <div className="text-center p-3 bg-surface-elevated rounded-lg">
              <div className="text-sm text-muted-foreground">Volume</div>
              <div className="text-lg font-semibold text-foreground">{company.volume}</div>
            </div>
          </div>
        </div>

        {/* Additional Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <div className="text-center p-3 bg-surface-interactive rounded-lg border border-border">
            <div className="text-xs text-muted-foreground uppercase tracking-wide">P/E Ratio</div>
            <div className="text-xl font-bold text-foreground">{company.peRatio}</div>
          </div>
          <div className="text-center p-3 bg-surface-interactive rounded-lg border border-border">
            <div className="text-xs text-muted-foreground uppercase tracking-wide">52W High</div>
            <div className="text-xl font-bold text-chart-secondary">${company.high52Week}</div>
          </div>
          <div className="text-center p-3 bg-surface-interactive rounded-lg border border-border">
            <div className="text-xs text-muted-foreground uppercase tracking-wide">52W Low</div>
            <div className="text-xl font-bold text-chart-danger">${company.low52Week}</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};