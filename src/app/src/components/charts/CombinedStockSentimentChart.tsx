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
    <div className="flex flex-col gap-6 min-h-0">
      {/* Stock Chart - Full Width */}
      <div className="min-h-0">
        <KaareStockChart chart={stockChart} showHeader={true} />
      </div>
      
      {/* Sentiment Chart - Below Stock */}
      {sentimentChart && (
        <div className="border-t border-[#4d4d4f]/30 pt-6 min-h-0">
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
