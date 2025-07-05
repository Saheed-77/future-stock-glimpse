import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Calendar, TrendingUp } from "lucide-react";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface StockDataPoint {
  date: string;
  price: number;
}

interface StockChartProps {
  data: StockDataPoint[];
  title: string;
  color: string;
  isPrediction?: boolean;
}

export const StockChart = ({ data, title, color, isPrediction = false }: StockChartProps) => {
  const chartData = {
    labels: data.map(point => point.date),
    datasets: [
      {
        label: title,
        data: data.map(point => point.price),
        borderColor: color,
        backgroundColor: `${color}10`,
        borderWidth: 2,
        fill: true,
        tension: 0.4,
        pointBackgroundColor: color,
        pointBorderColor: color,
        pointRadius: isPrediction ? 2 : 1,
        pointHoverRadius: 4,
        borderDash: isPrediction ? [5, 5] : [],
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        backgroundColor: 'hsl(210 24% 10%)',
        titleColor: 'hsl(213 31% 91%)',
        bodyColor: 'hsl(213 31% 91%)',
        borderColor: 'hsl(210 20% 20%)',
        borderWidth: 1,
        cornerRadius: 8,
        callbacks: {
          label: function(context: any) {
            return `$${context.parsed.y.toFixed(2)}`;
          },
        },
      },
    },
    scales: {
      x: {
        grid: {
          color: 'hsl(210 20% 20%)',
          drawBorder: false,
        },
        ticks: {
          color: 'hsl(215 20% 65%)',
          maxTicksLimit: 8,
        },
      },
      y: {
        grid: {
          color: 'hsl(210 20% 20%)',
          drawBorder: false,
        },
        ticks: {
          color: 'hsl(215 20% 65%)',
          callback: function(value: any) {
            return '$' + value.toFixed(0);
          },
        },
      },
    },
    interaction: {
      intersect: false,
      mode: 'index' as const,
    },
  };

  return (
    <Card className="bg-gradient-card border-border shadow-card">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-primary/10 rounded-lg">
              {isPrediction ? (
                <TrendingUp className="h-5 w-5 text-chart-primary" />
              ) : (
                <Calendar className="h-5 w-5 text-chart-primary" />
              )}
            </div>
            <CardTitle className="text-xl font-semibold text-foreground">{title}</CardTitle>
          </div>
          {isPrediction && (
            <Badge variant="secondary" className="bg-chart-primary/10 text-chart-primary border-chart-primary/20">
              Prediction
            </Badge>
          )}
        </div>
      </CardHeader>
      
      <CardContent>
        <div className="h-80">
          <Line data={chartData} options={options} />
        </div>
      </CardContent>
    </Card>
  );
};