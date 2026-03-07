import { useState } from "react";
import { AlertCircle } from "lucide-react";
import { StockPriceChart } from "./StockPriceChart";
import type { ChartData } from "@/types/chat";

interface ChartRendererProps {
  chart: ChartData;
}

export function ChartRenderer({ chart }: ChartRendererProps) {
  const [hasError, setHasError] = useState(false);

  // Error boundary for chart rendering
  if (hasError) {
    return (
      <div className="my-4 rounded-lg border border-destructive/50 bg-destructive/10 p-4">
        <div className="flex items-center gap-2 text-destructive">
          <AlertCircle className="h-5 w-5" />
          <span className="font-medium">
            Failed to render chart for {chart?.symbol || "Unknown"}
          </span>
        </div>
        <p className="mt-1 text-sm text-muted-foreground">
          There was an error displaying this chart. The data is still available
          in the text above.
        </p>
      </div>
    );
  }

  // Validate chart data
  if (!chart || !chart.data || chart.data.length === 0) {
    return (
      <div className="my-4 rounded-lg border border-destructive/50 bg-destructive/10 p-4">
        <div className="flex items-center gap-2 text-destructive">
          <AlertCircle className="h-5 w-5" />
          <span className="font-medium">
            Chart data unavailable for {chart?.symbol || "Unknown"}
          </span>
        </div>
      </div>
    );
  }

  // Render appropriate chart type
  try {
    switch (chart.type) {
      case "line_chart":
        return <StockPriceChart chart={chart} />;
      default:
        return (
          <div className="my-4 rounded-lg border border-yellow-500/50 bg-yellow-500/10 p-4">
            <div className="flex items-center gap-2 text-yellow-700">
              <AlertCircle className="h-5 w-5" />
              <span className="font-medium">
                Unsupported chart type: {chart.type}
              </span>
            </div>
          </div>
        );
    }
  } catch (error) {
    console.error("Error rendering chart:", error);
    setHasError(true);
    return null;
  }
}
