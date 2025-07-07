import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Area,
  AreaChart,
  ReferenceLine,
  Legend
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Calendar, TrendingUp, TrendingDown } from "lucide-react";
import { useThemeColors } from "@/hooks/useThemeColors";

interface StockDataPoint {
  date: string;
  price: number;
}

interface StockChartProps {
  data: StockDataPoint[];
  title: string;
  isPrediction?: boolean;
  variant?: 'primary' | 'secondary';
}

// Google-style colors
const GOOGLE_COLORS = {
  blue: '#4285f4',
  green: '#34a853',
  red: '#ea4335',
  yellow: '#fbbc04',
  purple: '#9c27b0',
  teal: '#00acc1',
  orange: '#ff9800',
  pink: '#e91e63',
  lightBlue: '#03a9f4',
  lime: '#8bc34a'
};

export const StockChart = ({ data, title, isPrediction = false, variant = 'primary' }: StockChartProps) => {
  const colors = useThemeColors();
  
  // Transform data for recharts
  const chartData = data.map((point, index) => ({
    ...point,
    date: new Date(point.date).toLocaleDateString('en-US', { 
      month: 'short', 
      day: 'numeric' 
    }),
    change: index > 0 ? point.price - data[index - 1].price : 0
  }));

  // Determine trend colors
  const isPositiveTrend = chartData.length > 1 && 
    chartData[chartData.length - 1].price > chartData[0].price;
  
  const primaryColor = variant === 'primary' 
    ? (isPrediction ? GOOGLE_COLORS.purple : GOOGLE_COLORS.blue)
    : (isPrediction ? GOOGLE_COLORS.orange : GOOGLE_COLORS.green);
  
  const gradientId = `gradient-${variant}-${isPrediction ? 'prediction' : 'historical'}`;

  // Custom tooltip component
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      const isPositive = data.change >= 0;
      
      return (
        <div className="bg-background border border-border rounded-lg p-3 shadow-lg">
          <p className="text-sm text-muted-foreground mb-1">{label}</p>
          <div className="flex items-center gap-2">
            <span className="font-bold text-foreground">
              ${payload[0].value.toFixed(2)}
            </span>
            {data.change !== 0 && (
              <span className={`text-xs flex items-center gap-1 ${
                isPositive ? 'text-green-600' : 'text-red-600'
              }`}>
                {isPositive ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
                {isPositive ? '+' : ''}${data.change.toFixed(2)}
              </span>
            )}
          </div>
        </div>
      );
    }
    return null;
  };

  // Calculate min and max for better y-axis scaling
  const prices = chartData.map(d => d.price);
  const minPrice = Math.min(...prices);
  const maxPrice = Math.max(...prices);
  const padding = (maxPrice - minPrice) * 0.1;
  
  return (
    <Card className="bg-gradient-card border-border shadow-card hover:shadow-elevated transition-all duration-300">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded-lg ${
              isPrediction 
                ? 'bg-purple-500/10' 
                : variant === 'primary' 
                  ? 'bg-blue-500/10' 
                  : 'bg-green-500/10'
            }`}>
              {isPrediction ? (
                <TrendingUp className={`h-5 w-5 ${
                  variant === 'primary' ? 'text-purple-500' : 'text-orange-500'
                }`} />
              ) : (
                <Calendar className={`h-5 w-5 ${
                  variant === 'primary' ? 'text-blue-500' : 'text-green-500'
                }`} />
              )}
            </div>
            <CardTitle className="text-xl font-semibold text-foreground">{title}</CardTitle>
          </div>
          <div className="flex items-center gap-2">
            {isPrediction && (
              <Badge 
                variant="secondary" 
                className="bg-purple-500/10 text-purple-600 border-purple-500/20 font-medium"
              >
                AI Prediction
              </Badge>
            )}
            <Badge 
              variant={isPositiveTrend ? "default" : "destructive"}
              className={`${
                isPositiveTrend 
                  ? 'bg-green-500/10 text-green-600 border-green-500/20' 
                  : 'bg-red-500/10 text-red-600 border-red-500/20'
              } font-medium`}
            >
              {isPositiveTrend ? '+' : ''}
              {((chartData[chartData.length - 1]?.price - chartData[0]?.price) || 0).toFixed(2)}
            </Badge>
          </div>
        </div>
      </CardHeader>
      
      <CardContent>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart 
              data={chartData} 
              margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
            >
              <defs>
                <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={primaryColor} stopOpacity={0.3}/>
                  <stop offset="95%" stopColor={primaryColor} stopOpacity={0.05}/>
                </linearGradient>
              </defs>
              
              <CartesianGrid 
                strokeDasharray="3 3" 
                stroke={colors.chartGrid} 
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
                domain={[minPrice - padding, maxPrice + padding]}
                tick={{ fill: colors.chartText, fontSize: 12 }}
                axisLine={false}
                tickLine={false}
                tickFormatter={(value) => `$${value.toFixed(0)}`}
              />
              
              <Tooltip content={<CustomTooltip />} />
              
              <Area
                type="monotone"
                dataKey="price"
                stroke={primaryColor}
                fill={`url(#${gradientId})`}
                strokeWidth={3}
                dot={{ 
                  fill: primaryColor, 
                  stroke: primaryColor, 
                  strokeWidth: 2, 
                  r: isPrediction ? 4 : 3 
                }}
                activeDot={{ 
                  r: 6, 
                  fill: primaryColor,
                  stroke: colors.chartBackground,
                  strokeWidth: 3
                }}
                strokeDasharray={isPrediction ? "8 8" : "0"}
              />
              
              {/* Average line */}
              <ReferenceLine 
                y={prices.reduce((a, b) => a + b, 0) / prices.length} 
                stroke={colors.chartText} 
                strokeDasharray="2 2" 
                strokeOpacity={0.5}
                label={{ 
                  value: "Avg", 
                  position: "insideTopRight",
                  fill: colors.chartText,
                  fontSize: 10
                }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        
        {/* Chart legend */}
        <div className="flex items-center justify-between mt-4 pt-4 border-t border-border">
          <div className="flex items-center gap-4 text-sm text-muted-foreground">
            <div className="flex items-center gap-2">
              <div 
                className="w-3 h-3 rounded-full" 
                style={{ backgroundColor: primaryColor }}
              />
              <span>{isPrediction ? 'Predicted Price' : 'Historical Price'}</span>
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
            {chartData.length} data points
          </div>
        </div>
      </CardContent>
    </Card>
  );
};