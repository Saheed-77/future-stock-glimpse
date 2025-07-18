import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  ReferenceLine,
  Cell
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { TrendingUp, TrendingDown, BarChart3 } from "lucide-react";
import { useThemeColors } from "@/hooks/useThemeColors";

interface StockDataPoint {
  date: string;
  price: number;
}

interface ProfitLossChartProps {
  data: StockDataPoint[];
  title: string;
  variant?: 'primary' | 'secondary';
}

// Google-style colors for profit/loss
const PROFIT_LOSS_COLORS = {
  profit: '#34a853',      // Google Green
  loss: '#ea4335',        // Google Red
  neutral: '#9aa0a6',     // Google Grey
  profitLight: '#e8f5e8', // Light green
  lossLight: '#fce8e6'    // Light red
};

export const ProfitLossChart = ({ data, title, variant = 'primary' }: ProfitLossChartProps) => {
  const colors = useThemeColors();
  
  // Transform data to show profit/loss changes
  const chartData = data.map((point, index) => {
    const previousPrice = index > 0 ? data[index - 1].price : point.price;
    const change = point.price - previousPrice;
    const changePercent = previousPrice > 0 ? (change / previousPrice) * 100 : 0;
    
    return {
      ...point,
      date: new Date(point.date).toLocaleDateString('en-US', { 
        month: 'short', 
        day: 'numeric' 
      }),
      change,
      changePercent,
      isProfit: change > 0,
      isLoss: change < 0,
      isNeutral: change === 0
    };
  });

  // Calculate statistics
  const totalProfit = chartData.filter(d => d.isProfit).length;
  const totalLoss = chartData.filter(d => d.isLoss).length;
  const totalNeutral = chartData.filter(d => d.isNeutral).length;
  const avgChange = chartData.reduce((sum, d) => sum + d.change, 0) / chartData.length;
  const maxChange = Math.max(...chartData.map(d => Math.abs(d.change)));

  // Custom tooltip component
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      const isProfit = data.isProfit;
      const isLoss = data.isLoss;
      
      return (
        <div className="bg-background border border-border rounded-lg p-3 shadow-lg">
          <p className="text-sm text-muted-foreground mb-1">{label}</p>
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">Price:</span>
              <span className="font-bold text-foreground">
                ${data.price.toFixed(2)}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">Change:</span>
              <span className={`font-bold flex items-center gap-1 ${
                isProfit ? 'text-green-600' : isLoss ? 'text-red-600' : 'text-gray-600'
              }`}>
                {isProfit ? <TrendingUp className="h-3 w-3" /> : 
                 isLoss ? <TrendingDown className="h-3 w-3" /> : null}
                {isProfit ? '+' : ''}${data.change.toFixed(2)}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">Percent:</span>
              <span className={`font-bold ${
                isProfit ? 'text-green-600' : isLoss ? 'text-red-600' : 'text-gray-600'
              }`}>
                {isProfit ? '+' : ''}{data.changePercent.toFixed(2)}%
              </span>
            </div>
          </div>
        </div>
      );
    }
    return null;
  };

  // Custom bar color function
  const getBarColor = (entry: any) => {
    if (entry.isProfit) return PROFIT_LOSS_COLORS.profit;
    if (entry.isLoss) return PROFIT_LOSS_COLORS.loss;
    return PROFIT_LOSS_COLORS.neutral;
  };

  return (
    <Card className="bg-gradient-card border-border shadow-card hover:shadow-elevated transition-all duration-300">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-500/10 rounded-lg">
              <BarChart3 className="h-5 w-5 text-blue-500" />
            </div>
            <CardTitle className="text-xl font-semibold text-foreground">{title}</CardTitle>
          </div>
          <div className="flex items-center gap-2">
            <Badge 
              variant="default"
              className="bg-green-500/10 text-green-600 border-green-500/20 font-medium"
            >
              +{totalProfit} days
            </Badge>
            <Badge 
              variant="destructive"
              className="bg-red-500/10 text-red-600 border-red-500/20 font-medium"
            >
              -{totalLoss} days
            </Badge>
          </div>
        </div>
      </CardHeader>
      
      <CardContent>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart 
              data={chartData} 
              margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
            >
              <CartesianGrid 
                strokeDasharray="3 3" 
                stroke={colors.chartGrid} 
                horizontal={true}
                vertical={false}
                strokeOpacity={0.3}
              />
              
              <XAxis 
                dataKey="date" 
                tick={{ fill: colors.chartText, fontSize: 12 }}
                axisLine={false}
                tickLine={false}
                interval={'preserveStartEnd'}
              />
              
              <YAxis 
                domain={[-maxChange * 1.1, maxChange * 1.1]}
                tick={{ fill: colors.chartText, fontSize: 12 }}
                axisLine={false}
                tickLine={false}
                tickFormatter={(value) => `$${value.toFixed(1)}`}
              />
              
              <Tooltip content={<CustomTooltip />} />
              
              {/* Zero reference line */}
              <ReferenceLine 
                y={0} 
                stroke={colors.chartText} 
                strokeWidth={2}
                strokeOpacity={0.8}
              />
              
              {/* Average change line */}
              <ReferenceLine 
                y={avgChange} 
                stroke={avgChange > 0 ? PROFIT_LOSS_COLORS.profit : PROFIT_LOSS_COLORS.loss} 
                strokeDasharray="5 5" 
                strokeOpacity={0.7}
                label={{ 
                  value: `Avg: $${avgChange.toFixed(2)}`, 
                  position: "insideTopRight",
                  fill: colors.chartText,
                  fontSize: 10
                }}
              />
              
              <Bar 
                dataKey="change" 
                radius={[2, 2, 2, 2]}
                maxBarSize={40}
              >
                {chartData.map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={getBarColor(entry)}
                    stroke={getBarColor(entry)}
                    strokeWidth={1}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        
        {/* Chart statistics */}
        <div className="mt-4 pt-4 border-t border-border">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {totalProfit}
              </div>
              <div className="text-xs text-muted-foreground">
                Profit Days
              </div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-red-600">
                {totalLoss}
              </div>
              <div className="text-xs text-muted-foreground">
                Loss Days
              </div>
            </div>
            <div className="text-center">
              <div className={`text-2xl font-bold ${
                avgChange > 0 ? 'text-green-600' : avgChange < 0 ? 'text-red-600' : 'text-gray-600'
              }`}>
                {avgChange > 0 ? '+' : ''}${avgChange.toFixed(2)}
              </div>
              <div className="text-xs text-muted-foreground">
                Avg Change
              </div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-foreground">
                {((totalProfit / (totalProfit + totalLoss)) * 100).toFixed(0)}%
              </div>
              <div className="text-xs text-muted-foreground">
                Win Rate
              </div>
            </div>
          </div>
        </div>
        
        {/* Chart legend */}
        <div className="flex items-center justify-between mt-4 pt-4 border-t border-border">
          <div className="flex items-center gap-4 text-sm text-muted-foreground">
            <div className="flex items-center gap-2">
              <div 
                className="w-3 h-3 rounded-sm" 
                style={{ backgroundColor: PROFIT_LOSS_COLORS.profit }}
              />
              <span>Profit</span>
            </div>
            <div className="flex items-center gap-2">
              <div 
                className="w-3 h-3 rounded-sm" 
                style={{ backgroundColor: PROFIT_LOSS_COLORS.loss }}
              />
              <span>Loss</span>
            </div>
            <div className="flex items-center gap-2">
              <div 
                className="w-3 h-1 bg-muted-foreground opacity-50" 
                style={{ backgroundImage: 'repeating-linear-gradient(to right, transparent, transparent 2px, currentColor 2px, currentColor 4px)' }}
              />
              <span>Average</span>
            </div>
          </div>
          <div className="text-xs text-muted-foreground">
            Daily Price Changes
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
