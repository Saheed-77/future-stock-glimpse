import { Brain } from "lucide-react";
import { ThemeToggle } from "./ThemeToggle";
import { CompanySearch } from "./CompanySearch";

interface Company {
  symbol: string;
  name: string;
  industry: string;
}

interface NavbarProps {
  onCompanySelect: (company: Company) => void;
  selectedCompany: Company | null;
}

export const Navbar = ({ onCompanySelect, selectedCompany }: NavbarProps) => {
  return (
    <header className="sticky top-0 z-50 w-full bg-gradient-card border-b border-border shadow-card backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto px-4 py-4">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-primary/10 rounded-xl">
                <Brain className="h-6 w-6 text-primary" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-foreground">Stock Price Predictor</h1>
                <p className="text-sm text-muted-foreground hidden sm:block">AI-powered financial forecasting</p>
              </div>
            </div>
            <ThemeToggle />
          </div>
          
          <div className="lg:max-w-md w-full">
            <CompanySearch 
              onCompanySelect={onCompanySelect}
              selectedCompany={selectedCompany}
            />
          </div>
        </div>
      </div>
    </header>
  );
};