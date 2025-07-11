import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Brain, Target, Activity, Zap } from "lucide-react";

interface ModelMetricsProps {
  // API format
  mse?: number;
  mae?: number;
  accuracy: number;
  rmse?: number;
  r2_score?: number;
  testSamples?: number;
  lastUpdated?: string;
  // Mock data format
  modelName?: string;
  confidenceScore?: number;
  predictionRange?: string;
  lastTrained?: string;
  features?: string[];
}

export const ModelMetrics = (props: ModelMetricsProps) => {
  // Extract values with fallbacks for both API and mock data
  const modelName = props.modelName || "LSTM Neural Network";
  const accuracy = props.accuracy;
  const confidenceScore = props.confidenceScore || (props.accuracy * 0.9); // Approximate confidence from accuracy
  const predictionRange = props.predictionRange || "30 days";
  const lastTrained = props.lastTrained || props.lastUpdated || "Recently";
  const features = props.features || ["Price", "Volume", "Technical Indicators"];

  const getConfidenceColor = (score: number) => {
    if (score >= 80) return "text-chart-secondary";
    if (score >= 60) return "text-chart-accent";
    return "text-chart-danger";
  };

  const getAccuracyColor = (acc: number) => {
    if (acc >= 85) return "bg-chart-secondary";
    if (acc >= 70) return "bg-chart-accent";
    return "bg-chart-danger";
  };

  return (
    <Card className="bg-gradient-card border-border shadow-card">
      <CardHeader className="pb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-primary/10 rounded-lg">
            <Brain className="h-6 w-6 text-primary" />
          </div>
          <div>
            <CardTitle className="text-xl font-semibold text-foreground">ML Model Insights</CardTitle>
            <p className="text-sm text-muted-foreground">Performance metrics and prediction details</p>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Model Information */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Zap className="h-4 w-4 text-chart-primary" />
              <span className="text-sm font-medium text-muted-foreground">Model</span>
            </div>
            <Badge variant="outline" className="bg-surface-elevated border-primary/20 text-primary">
              {modelName}
            </Badge>
          </div>

          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Target className="h-4 w-4 text-chart-primary" />
              <span className="text-sm font-medium text-muted-foreground">Prediction Range</span>
            </div>
            <div className="text-lg font-semibold text-foreground">{predictionRange}</div>
          </div>
        </div>

        {/* Performance Metrics */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Activity className="h-4 w-4 text-chart-primary" />
              <span className="text-sm font-medium text-muted-foreground">Model Accuracy</span>
            </div>
            <span className="text-lg font-bold text-foreground">{accuracy}%</span>
          </div>
          <Progress 
            value={accuracy} 
            className="h-2 bg-surface-elevated"
            style={{ 
              "--progress-background": getAccuracyColor(accuracy).replace("bg-", "hsl(var(--chart-") + "))"
            } as React.CSSProperties}
          />
        </div>

        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-muted-foreground">Confidence Score</span>
            <span className={`text-2xl font-bold ${getConfidenceColor(confidenceScore)}`}>
              {confidenceScore}%
            </span>
          </div>
          <Progress 
            value={confidenceScore} 
            className="h-2 bg-surface-elevated"
          />
        </div>

        {/* Additional Info */}
        <div className="grid grid-cols-1 gap-4">
          <div className="p-4 bg-surface-elevated rounded-lg border border-border">
            <div className="text-sm text-muted-foreground mb-2">Last Training Session</div>
            <div className="text-sm font-medium text-foreground">{lastTrained}</div>
          </div>

          <div className="p-4 bg-surface-elevated rounded-lg border border-border">
            <div className="text-sm text-muted-foreground mb-3">Key Features</div>
            <div className="flex flex-wrap gap-2">
              {features.map((feature, index) => (
                <Badge 
                  key={index} 
                  variant="secondary" 
                  className="bg-surface-interactive text-xs"
                >
                  {feature}
                </Badge>
              ))}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};