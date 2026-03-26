import { KaareStockChart } from "./KaareStockChart";
import { SentimentChart } from "./SentimentChart";
import type { ChartData } from "@/types/chat";

interface CombinedStockSentimentChartProps {
  stockChart: ChartData;
  sentimentChart?: {
    symbol: string;
    data?: Array<{
      date: string;
      sentiment: number;
      volume: number;
    }>;
    averageSentiment?: number;
    metadata?: {
      start_date: string;
      end_date: string;
      total_articles?: number;
    };
  };
}

export function CombinedStockSentimentChart({ 
  stockChart, 
  sentimentChart 
}: CombinedStockSentimentChartProps) {
  return (
    <div className="space-y-6">
      {/* Stock Chart - Full Width */}
      <KaareStockChart chart={stockChart} showHeader={true} />
      
      {/* Sentiment Chart - Below Stock */}
      {sentimentChart && (
        <div className="border-t border-[#4d4d4f]/30 pt-6">
          <SentimentChart
            symbol={sentimentChart.symbol}
            data={sentimentChart.data}
            averageSentiment={sentimentChart.averageSentiment}
            metadata={sentimentChart.metadata}
          />
        </div>
      )}
    </div>
  );
}
