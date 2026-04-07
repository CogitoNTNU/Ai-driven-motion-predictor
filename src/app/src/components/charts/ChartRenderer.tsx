import { useMemo } from "react";
import { AlertCircle } from "lucide-react";
import { KaareStockChart } from "./KaareStockChart";
import { SentimentChart } from "./SentimentChart";
import { CombinedStockSentimentChart } from "./CombinedStockSentimentChart";
import type { ChartData } from "@/types/chat";

interface ChartRendererProps {
  chart: ChartData;
  sentimentChart?: ChartData;
}

export function ChartRenderer({ chart, sentimentChart }: ChartRendererProps) {
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

    // If we have both stock and sentiment charts, show combined view
    if (sentimentChart && chart.tool_name === "get_stock_growth") {
      // Map sentiment data safely - supports both 'sentiment' and 'price' fields
      const mappedSentimentData = sentimentChart.data?.map(d => ({
        date: d.date,
        sentiment: typeof d.sentiment === 'number' ? d.sentiment : typeof d.price === 'number' ? d.price : 0,
        volume: Math.floor(Math.random() * 50) + 10
      })) || [];
      
      return (
        <CombinedStockSentimentChart
          stockChart={chart}
          sentimentChart={{
            symbol: sentimentChart.symbol,
            data: mappedSentimentData,
            averageSentiment: sentimentChart.metadata?.avg_sentiment ?? 0,
            metadata: {
              start_date: sentimentChart.metadata?.start_date || chart.metadata?.start_date || "",
              end_date: sentimentChart.metadata?.end_date || chart.metadata?.end_date || "",
              total_articles: sentimentChart.metadata?.article_count,
            }
          }}
        />
      );
    }

    // Render based on tool_name and type
    switch (chart.tool_name) {
      case "get_stock_growth":
      case "get_current_price":
        return <KaareStockChart chart={chart} showHeader={true} />;
      
      case "get_stock_news_sentiment":
        // Map backend data - supports both 'sentiment' and 'price' fields
        const sentimentData = chart.data?.map(d => ({
          date: d.date,
          sentiment: typeof d.sentiment === 'number' ? d.sentiment : typeof d.price === 'number' ? d.price : 0,
          volume: Math.floor(Math.random() * 50) + 10
        })) || [];
        
        return (
          <SentimentChart
            symbol={chart.symbol}
            data={sentimentData}
            averageSentiment={chart.metadata?.avg_sentiment ?? 0}
            metadata={{
              start_date: chart.metadata?.start_date || "",
              end_date: chart.metadata?.end_date || "",
              total_articles: chart.metadata?.article_count,
            }}
          />
        );
      
      default:
        // Fallback to Kååre style for line/area charts
        if (chart.type === "line_chart" || chart.type === "area_chart") {
          return <KaareStockChart chart={chart} showHeader={true} />;
        }
        
        return (
          <div className="my-4 rounded-lg border border-yellow-500/50 bg-yellow-500/10 p-4">
            <div className="flex items-center gap-2 text-yellow-400">
              <AlertCircle className="h-5 w-5" />
              <span className="font-medium">
                Chart type: {chart.type} (tool: {chart.tool_name})
              </span>
            </div>
          </div>
        );
    }
  }, [chart, sentimentChart]);

  return <div className="chart-wrapper">{chartComponent}</div>;
}
