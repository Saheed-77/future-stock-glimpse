import { useState } from "react";
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
  Legend,
  ComposedChart,
  Bar
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  Calendar, 
  TrendingUp, 
  TrendingDown, 
  Brain,
  Target,
  Zap,
  Activity,
  BarChart3
} from "lucide-react";
import { useThemeColors } from "@/hooks/useThemeColors";

interface PredictionDataPoint {
  date: string;
  price: number;
  confidence?: number;
  day?: number;
}

interface HistoricalDataPoint {
  date: string;
  price: number;
  volume?: number;
  high?: number;
  low?: number;
}

interface MLPredictionDisplayProps {
  historicalData: HistoricalDataPoint[];
  predictionData: PredictionDataPoint[];
  symbol: string;
  modelSource: 'ml_model' | 'statistical_fallback' | 'mock_data';
  isTraining?: boolean;
}

export const MLPredictionDisplay = ({ 
  historicalData, 
  predictionData, 
  symbol,
  modelSource,
  isTraining = false
}: MLPredictionDisplayProps) => {
  const colors = useThemeColors();
  const [selectedDays, setSelectedDays] = useState(30);
  const [activeTab, setActiveTab] = useState("combined");

  // Prepare combined data for the chart
  const currentPrice = historicalData[historicalData.length - 1]?.price || 0;
  const lastHistoricalDate = new Date(historicalData[historicalData.length - 1]?.date || new Date());
  
  // Get recent historical data (last 30 days) for context
  const recentHistorical = historicalData.slice(-30).map(point => ({
    ...point,
    type: 'historical',
    formattedDate: new Date(point.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
  }));

  // Filter predictions based on selected days
  const filteredPredictions = predictionData.slice(0, selectedDays).map(point => ({
    ...point,
    type: 'prediction',
    formattedDate: new Date(point.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
  }));

  // Combine data for the chart
  const combinedData = [
    ...recentHistorical,
    ...filteredPredictions
  ];

  // Calculate trend and statistics
  const firstPrediction = filteredPredictions[0]?.price || currentPrice;
  const lastPrediction = filteredPredictions[filteredPredictions.length - 1]?.price || currentPrice;
  const predictedChange = lastPrediction - currentPrice;
  const predictedChangePercent = (predictedChange / currentPrice) * 100;
  const avgConfidence = filteredPredictions.reduce((sum, p) => sum + (p.confidence || 0.8), 0) / filteredPredictions.length;

  // Model status badge
  const getModelBadge = () => {
    if (isTraining) {
      return (
        <Badge className="bg-blue-500/10 text-blue-600 border-blue-500/20 animate-pulse">
          <Activity className="h-3 w-3 mr-1" />
          Training in Progress
        </Badge>
      );
    }
    
    switch (modelSource) {
      case 'ml_model':
        return (
          <Badge className="bg-green-500/10 text-green-600 border-green-500/20">
            <Brain className="h-3 w-3 mr-1" />
            ML Model Active
          </Badge>
        );
      case 'statistical_fallback':
        return (
          <Badge className="bg-orange-500/10 text-orange-600 border-orange-500/20">
            <BarChart3 className="h-3 w-3 mr-1" />
            Statistical Model
          </Badge>
        );
      default:
        return (
          <Badge className="bg-gray-500/10 text-gray-600 border-gray-500/20">
            <Target className="h-3 w-3 mr-1" />
            Demo Mode
          </Badge>
        );
    }
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      const isHistorical = data.type === 'historical';
      
      return (
        <div className="bg-background border border-border rounded-lg p-3 shadow-lg">
          <p className="text-sm text-muted-foreground mb-1">{data.formattedDate}</p>
          <div className="flex items-center gap-2">
            <span className="font-bold text-foreground">
              ${payload[0].value.toFixed(2)}
            </span>
            <Badge 
              variant="outline" 
              className={`text-xs ${isHistorical ? 'text-blue-600' : 'text-purple-600'}`}
            >
              {isHistorical ? 'Historical' : 'Predicted'}
            </Badge>
          </div>
          {!isHistorical && data.confidence && (
            <p className="text-xs text-muted-foreground mt-1">
              Confidence: {(data.confidence * 100).toFixed(1)}%
            </p>
          )}
          {data.volume && (
            <p className="text-xs text-muted-foreground mt-1">
              Volume: {data.volume.toLocaleString()}
            </p>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-6">
      {/* Header with Model Status */}
      <Card className="bg-gradient-card border-border">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-purple-500/10">
                <Brain className="h-6 w-6 text-purple-500" />
              </div>
              <div>
                <CardTitle className="text-xl">AI Stock Prediction - {symbol}</CardTitle>
                <p className="text-sm text-muted-foreground">
                  Machine Learning powered price forecasting
                </p>
              </div>
            </div>
            {getModelBadge()}
          </div>
        </CardHeader>
        
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <p className="text-sm text-muted-foreground">Current Price</p>
              <p className="text-lg font-bold text-foreground">${currentPrice.toFixed(2)}</p>
            </div>
            <div className="text-center">
              <p className="text-sm text-muted-foreground">Predicted ({selectedDays}d)</p>
              <p className={`text-lg font-bold ${predictedChange >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                ${lastPrediction.toFixed(2)}
              </p>
            </div>
            <div className="text-center">
              <p className="text-sm text-muted-foreground">Expected Change</p>
              <p className={`text-lg font-bold ${predictedChange >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {predictedChange >= 0 ? '+' : ''}${predictedChange.toFixed(2)} 
                ({predictedChangePercent >= 0 ? '+' : ''}{predictedChangePercent.toFixed(1)}%)
              </p>
            </div>
            <div className="text-center">
              <p className="text-sm text-muted-foreground">Avg Confidence</p>
              <p className="text-lg font-bold text-foreground">{(avgConfidence * 100).toFixed(1)}%</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Prediction Controls */}
      <Card className="bg-gradient-card border-border">
        <CardContent className="pt-6">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-semibold text-foreground">Prediction Timeline</h3>
              <p className="text-sm text-muted-foreground">Select forecast duration</p>
            </div>
            <div className="flex gap-2">
              {[7, 15, 30, 60, 90].map((days) => (
                <Button
                  key={days}
                  variant={selectedDays === days ? "default" : "outline"}
                  size="sm"
                  onClick={() => setSelectedDays(days)}
                  disabled={days > predictionData.length}
                >
                  {days}d
                </Button>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Prediction Charts */}
      <Card className="bg-gradient-card border-border">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5 text-yellow-500" />
            Price Prediction Analysis
          </CardTitle>
        </CardHeader>
        
        <CardContent>
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="combined">Combined View</TabsTrigger>
              <TabsTrigger value="prediction">Prediction Only</TabsTrigger>
              <TabsTrigger value="confidence">Confidence Levels</TabsTrigger>
            </TabsList>
            
            <TabsContent value="combined" className="mt-6">
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={combinedData}>
                    <CartesianGrid strokeDasharray="3 3" stroke={colors.chartGrid} strokeOpacity={0.3} />
                    <XAxis 
                      dataKey="formattedDate" 
                      tick={{ fill: colors.chartText, fontSize: 12 }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <YAxis 
                      tick={{ fill: colors.chartText, fontSize: 12 }}
                      axisLine={false}
                      tickLine={false}
                      tickFormatter={(value) => `$${value.toFixed(0)}`}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <ReferenceLine 
                      x={recentHistorical[recentHistorical.length - 1]?.formattedDate} 
                      stroke="#8884d8" 
                      strokeDasharray="5 5" 
                      label={{ value: "Today", position: "top" }}
                    />
                    
                    {/* Historical line */}
                    <Line
                      type="monotone"
                      dataKey="price"
                      stroke="#4285f4"
                      strokeWidth={2}
                      dot={false}
                      connectNulls={false}
                      name="Historical"
                      data={recentHistorical}
                    />
                    
                    {/* Prediction line */}
                    <Line
                      type="monotone"
                      dataKey="price"
                      stroke="#9c27b0"
                      strokeWidth={2}
                      strokeDasharray="5 5"
                      dot={false}
                      connectNulls={false}
                      name="Prediction"
                      data={filteredPredictions}
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </TabsContent>
            
            <TabsContent value="prediction" className="mt-6">
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={filteredPredictions}>
                    <defs>
                      <linearGradient id="predictionGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#9c27b0" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="#9c27b0" stopOpacity={0.05}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke={colors.chartGrid} strokeOpacity={0.3} />
                    <XAxis 
                      dataKey="formattedDate" 
                      tick={{ fill: colors.chartText, fontSize: 12 }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <YAxis 
                      tick={{ fill: colors.chartText, fontSize: 12 }}
                      axisLine={false}
                      tickLine={false}
                      tickFormatter={(value) => `$${value.toFixed(0)}`}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Area
                      type="monotone"
                      dataKey="price"
                      stroke="#9c27b0"
                      fillOpacity={1}
                      fill="url(#predictionGradient)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </TabsContent>
            
            <TabsContent value="confidence" className="mt-6">
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={filteredPredictions}>
                    <CartesianGrid strokeDasharray="3 3" stroke={colors.chartGrid} strokeOpacity={0.3} />
                    <XAxis 
                      dataKey="formattedDate" 
                      tick={{ fill: colors.chartText, fontSize: 12 }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <YAxis 
                      yAxisId="price"
                      orientation="left"
                      tick={{ fill: colors.chartText, fontSize: 12 }}
                      axisLine={false}
                      tickLine={false}
                      tickFormatter={(value) => `$${value.toFixed(0)}`}
                    />
                    <YAxis 
                      yAxisId="confidence"
                      orientation="right"
                      tick={{ fill: colors.chartText, fontSize: 12 }}
                      axisLine={false}
                      tickLine={false}
                      domain={[0, 1]}
                      tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar 
                      yAxisId="confidence"
                      dataKey="confidence" 
                      fill="#34a853" 
                      fillOpacity={0.6} 
                      name="Confidence"
                    />
                    <Line
                      yAxisId="price"
                      type="monotone"
                      dataKey="price"
                      stroke="#9c27b0"
                      strokeWidth={2}
                      dot={false}
                      name="Price"
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
};
