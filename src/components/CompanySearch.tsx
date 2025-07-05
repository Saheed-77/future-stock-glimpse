import { useState } from "react";
import { Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";

interface Company {
  symbol: string;
  name: string;
  industry: string;
}

const mockCompanies: Company[] = [
  { symbol: "AAPL", name: "Apple Inc.", industry: "Technology" },
  { symbol: "GOOGL", name: "Alphabet Inc.", industry: "Technology" },
  { symbol: "MSFT", name: "Microsoft Corporation", industry: "Technology" },
  { symbol: "AMZN", name: "Amazon.com Inc.", industry: "Consumer Discretionary" },
  { symbol: "TSLA", name: "Tesla Inc.", industry: "Consumer Discretionary" },
  { symbol: "NVDA", name: "NVIDIA Corporation", industry: "Technology" },
  { symbol: "META", name: "Meta Platforms Inc.", industry: "Technology" },
  { symbol: "BRK.B", name: "Berkshire Hathaway Inc.", industry: "Financial Services" },
  { symbol: "V", name: "Visa Inc.", industry: "Financial Services" },
  { symbol: "JNJ", name: "Johnson & Johnson", industry: "Healthcare" },
];

interface CompanySearchProps {
  onCompanySelect: (company: Company) => void;
  selectedCompany: Company | null;
}

export const CompanySearch = ({ onCompanySelect, selectedCompany }: CompanySearchProps) => {
  const [searchTerm, setSearchTerm] = useState("");
  const [isOpen, setIsOpen] = useState(false);

  const filteredCompanies = mockCompanies.filter(
    (company) =>
      company.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      company.symbol.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleCompanySelect = (company: Company) => {
    onCompanySelect(company);
    setSearchTerm("");
    setIsOpen(false);
  };

  return (
    <div className="relative w-full max-w-md">
      <div className="relative">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
        <Input
          type="text"
          placeholder="Search companies..."
          value={searchTerm}
          onChange={(e) => {
            setSearchTerm(e.target.value);
            setIsOpen(true);
          }}
          onFocus={() => setIsOpen(true)}
          className="pl-10 bg-surface-elevated border-border transition-smooth"
        />
      </div>

      {isOpen && searchTerm && filteredCompanies.length > 0 && (
        <Card className="absolute z-50 w-full mt-1 bg-surface-elevated border-border shadow-elevated">
          <div className="max-h-60 overflow-y-auto">
            {filteredCompanies.slice(0, 8).map((company) => (
              <div
                key={company.symbol}
                className="px-4 py-3 cursor-pointer hover:bg-surface-interactive transition-smooth border-b border-border last:border-b-0"
                onClick={() => handleCompanySelect(company)}
              >
                <div className="flex justify-between items-center">
                  <div>
                    <div className="font-medium text-foreground">{company.symbol}</div>
                    <div className="text-sm text-muted-foreground">{company.name}</div>
                  </div>
                  <div className="text-xs text-muted-foreground">{company.industry}</div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {selectedCompany && (
        <div className="mt-3 p-3 bg-gradient-primary rounded-lg border border-primary/20">
          <div className="flex justify-between items-center">
            <div>
              <div className="font-semibold text-foreground">{selectedCompany.symbol}</div>
              <div className="text-sm text-muted-foreground">{selectedCompany.name}</div>
            </div>
            <div className="text-xs text-primary font-medium">{selectedCompany.industry}</div>
          </div>
        </div>
      )}
    </div>
  );
};