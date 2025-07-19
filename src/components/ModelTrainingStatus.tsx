import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { 
  Brain, 
  Activity, 
  CheckCircle, 
  XCircle, 
  Clock,
  Zap,
  RefreshCw,
  Download,
  Settings
} from "lucide-react";
import { stockAPI } from "@/services/stockAPI";

interface TrainingStatusProps {
  symbol: string;
  onTrainingComplete?: () => void;
}

interface TrainingStatus {
  symbol: string;
  model_exists: boolean;
  is_training: boolean;
  model_loaded: boolean;
  supported: boolean;
  model_age_days?: number;
  needs_retraining?: boolean;
}

export const ModelTrainingStatus = ({ symbol, onTrainingComplete }: TrainingStatusProps) => {
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isTrainingStarted, setIsTrainingStarted] = useState(false);

  useEffect(() => {
    loadTrainingStatus();
    
    // Poll for status updates if training is in progress
    let interval: NodeJS.Timeout;
    if (status?.is_training || isTrainingStarted) {
      interval = setInterval(() => {
        loadTrainingStatus();
      }, 3000); // Check every 3 seconds
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [symbol, status?.is_training, isTrainingStarted]);

  const loadTrainingStatus = async () => {
    try {
      setError(null);
      const response = await fetch(`http://localhost:5000/api/stock/${symbol}/training-status`);
      const data = await response.json();
      
      if (data.success) {
        const previousTraining = status?.is_training;
        setStatus(data.status);
        
        // Check if training just completed
        if (previousTraining && !data.status.is_training && data.status.model_exists) {
          setIsTrainingStarted(false);
          if (onTrainingComplete) {
            onTrainingComplete();
          }
        }
      } else {
        setError(data.error || 'Failed to load training status');
      }
    } catch (err) {
      setError('Unable to connect to backend service');
      console.error('Error loading training status:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const startTraining = async () => {
    try {
      setError(null);
      setIsTrainingStarted(true);
      
      const response = await fetch(`http://localhost:5000/api/stock/${symbol}/train`, {
        method: 'POST'
      });
      const data = await response.json();
      
      if (data.success) {
        // Refresh status immediately
        await loadTrainingStatus();
      } else {
        setError(data.error || 'Failed to start training');
        setIsTrainingStarted(false);
      }
    } catch (err) {
      setError('Failed to start training');
      setIsTrainingStarted(false);
      console.error('Error starting training:', err);
    }
  };

  const getStatusBadge = () => {
    if (!status) return null;

    if (status.is_training || isTrainingStarted) {
      return (
        <Badge className="bg-blue-500/10 text-blue-600 border-blue-500/20 animate-pulse">
          <Activity className="h-3 w-3 mr-1" />
          Training in Progress
        </Badge>
      );
    }

    if (status.model_exists && status.model_loaded) {
      return (
        <Badge className="bg-green-500/10 text-green-600 border-green-500/20">
          <CheckCircle className="h-3 w-3 mr-1" />
          Model Ready
        </Badge>
      );
    }

    if (status.model_exists && !status.model_loaded) {
      return (
        <Badge className="bg-yellow-500/10 text-yellow-600 border-yellow-500/20">
          <Download className="h-3 w-3 mr-1" />
          Model Available
        </Badge>
      );
    }

    if (!status.model_exists && status.supported) {
      return (
        <Badge className="bg-orange-500/10 text-orange-600 border-orange-500/20">
          <XCircle className="h-3 w-3 mr-1" />
          No Model
        </Badge>
      );
    }

    return (
      <Badge className="bg-red-500/10 text-red-600 border-red-500/20">
        <XCircle className="h-3 w-3 mr-1" />
        Not Supported
      </Badge>
    );
  };

  const getStatusMessage = () => {
    if (!status) return "Loading...";

    if (status.is_training || isTrainingStarted) {
      return "AI model is currently being trained with 10 years of historical data. This may take 1-2 minutes.";
    }

    if (status.model_exists && status.model_loaded) {
      const ageMessage = status.model_age_days !== undefined 
        ? ` (trained ${status.model_age_days} days ago)`
        : "";
      return `ML model is active and ready for predictions${ageMessage}.`;
    }

    if (status.model_exists && !status.model_loaded) {
      return "Model file exists but needs to be loaded. It will be loaded automatically when making predictions.";
    }

    if (!status.model_exists && status.supported) {
      return "No trained model found. Click 'Start Training' to create a new ML model.";
    }

    return "This symbol is not currently supported for ML predictions.";
  };

  const shouldShowRetraining = () => {
    return status?.needs_retraining && !status.is_training && !isTrainingStarted;
  };

  if (isLoading) {
    return (
      <Card className="bg-gradient-card border-border">
        <CardContent className="flex items-center justify-center py-8">
          <div className="flex items-center gap-2 text-muted-foreground">
            <RefreshCw className="h-4 w-4 animate-spin" />
            Loading model status...
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="bg-gradient-card border-border">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-blue-500/10">
              <Brain className="h-5 w-5 text-blue-500" />
            </div>
            <div>
              <CardTitle className="text-lg">ML Model Status</CardTitle>
              <p className="text-sm text-muted-foreground">
                Training and prediction status for {symbol}
              </p>
            </div>
          </div>
          {getStatusBadge()}
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {error && (
          <Alert className="border-red-500/20 bg-red-500/10">
            <XCircle className="h-4 w-4" />
            <AlertDescription className="text-red-600">
              {error}
            </AlertDescription>
          </Alert>
        )}

        <div className="space-y-3">
          <p className="text-sm text-muted-foreground">
            {getStatusMessage()}
          </p>

          {(status?.is_training || isTrainingStarted) && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span>Training Progress</span>
                <span className="text-muted-foreground">In progress...</span>
              </div>
              <Progress value={undefined} className="h-2" />
              <p className="text-xs text-muted-foreground">
                Training ensemble model (Random Forest + Linear Regression) with technical indicators
              </p>
            </div>
          )}

          {shouldShowRetraining() && (
            <Alert className="border-yellow-500/20 bg-yellow-500/10">
              <Clock className="h-4 w-4" />
              <AlertDescription className="text-yellow-700">
                Model is {status?.model_age_days} days old and may benefit from retraining with recent data.
              </AlertDescription>
            </Alert>
          )}

          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Model File:</span>
                <span className={status?.model_exists ? "text-green-600" : "text-red-600"}>
                  {status?.model_exists ? "✓ Exists" : "✗ Missing"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Model Loaded:</span>
                <span className={status?.model_loaded ? "text-green-600" : "text-gray-600"}>
                  {status?.model_loaded ? "✓ Ready" : "○ Not Loaded"}
                </span>
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Training:</span>
                <span className={(status?.is_training || isTrainingStarted) ? "text-blue-600" : "text-gray-600"}>
                  {(status?.is_training || isTrainingStarted) ? "● Active" : "○ Idle"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Supported:</span>
                <span className={status?.supported ? "text-green-600" : "text-red-600"}>
                  {status?.supported ? "✓ Yes" : "✗ No"}
                </span>
              </div>
            </div>
          </div>

          <div className="flex gap-2 pt-2">
            {(!status?.model_exists || shouldShowRetraining()) && status?.supported && !status?.is_training && !isTrainingStarted && (
              <Button
                onClick={startTraining}
                size="sm"
                className="bg-blue-600 hover:bg-blue-700"
              >
                <Zap className="h-4 w-4 mr-1" />
                {!status?.model_exists ? "Start Training" : "Retrain Model"}
              </Button>
            )}
            
            <Button
              variant="outline"
              size="sm"
              onClick={loadTrainingStatus}
            >
              <RefreshCw className="h-4 w-4 mr-1" />
              Refresh Status
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
