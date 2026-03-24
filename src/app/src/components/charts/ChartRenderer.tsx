import { useMemo } from "react";
import { AlertCircle } from "lucide-react";
import { StockPriceChart } from "./StockPriceChart";
import { StockBarChart } from "./StockBarChart";
import { StockAreaChart } from "./StockAreaChart";
import { StockPieChart } from "./StockPieChart";
import type { ChartData } from "@/types/chat";

interface ChartRendererProps {
  chart: ChartData;
}

export function ChartRenderer({ chart }: ChartRendererProps) {
  const chartComponent = useMemo(() => {
    if (!chart || !chart.data || chart.data.length === 0) {
      return (
        <div className="my-4 rounded-lg border border-red-500/50 bg-red-500/10 p-4">
          <div className="flex items-center gap-2 text-red-400">
            <AlertCircle className="h-5 w-5" />
            <span className="font-medium">
              Chart data unavailable for {chart?.symbol || "Unknown"}
            </span>
          </div>
        </div>
      );
    }

    switch (chart.type) {
      case "line_chart":
        return <StockPriceChart chart={chart} />;
      case "bar_chart":
        return <StockBarChart chart={chart} />;
      case "area_chart":
        return <StockAreaChart chart={chart} />;
      case "pie_chart":
        return <StockPieChart chart={chart} />;
      default:
        return (
          <div className="my-4 rounded-lg border border-yellow-500/50 bg-yellow-500/10 p-4">
            <div className="flex items-center gap-2 text-yellow-400">
              <AlertCircle className="h-5 w-5" />
              <span className="font-medium">
                Unsupported chart type: {chart.type}
              </span>
            </div>
          </div>
        );
    }
  }, [chart]);

  return <div className="chart-wrapper">{chartComponent}</div>;
}
