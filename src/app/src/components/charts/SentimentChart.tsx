import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from "recharts";
import { useMemo } from "react";

interface SentimentDataPoint {
  date: string;
  sentiment: number;
  volume: number;
}

interface SentimentChartProps {
  data?: SentimentDataPoint[];
  symbol: string;
  averageSentiment?: number;
  metadata?: {
    start_date: string;
    end_date: string;
    total_articles?: number;
  };
}

export function SentimentChart({ 
  data = [], 
  symbol, 
  averageSentiment = 0,
  metadata 
}: SentimentChartProps) {
  
  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  };

  // Calculate dynamic Y-axis domain based on actual data
  const yDomain = useMemo(() => {
    if (data.length === 0) return [-0.5, 0.5];
    
    const sentiments = data.map(d => d.sentiment);
    const min = Math.min(...sentiments);
    const max = Math.max(...sentiments);
    
    // Add padding and ensure zero is centered
    const padding = Math.max(Math.abs(min), Math.abs(max)) * 0.2;
    const maxAbs = Math.max(Math.abs(min), Math.abs(max)) + padding;
    
    // Minimum range to avoid tiny domains
    const minRange = 0.2;
    const finalMax = Math.max(maxAbs, minRange / 2);
    
    return [-finalMax, finalMax];
  }, [data]);

  // Color based on sentiment value
  const getBarColor = (sentiment: number) => {
    const val = sentiment ?? 0;
    if (val > 0.2) return "#22c55e"; // Strong positive
    if (val > 0) return "#4ade80";   // Positive
    if (val === 0) return "#9ca3af"; // Neutral
    if (val > -0.2) return "#f87171"; // Negative
    return "#ef4444"; // Strong negative
  };

  const getSentimentLabel = (sentiment: number) => {
    const val = sentiment ?? 0;
    if (val > 0.2) return "Bullish";
    if (val > 0) return "Positive";
    if (val === 0) return "Neutral";
    if (val > -0.2) return "Negative";
    return "Bearish";
  };

  const sentimentLabel = getSentimentLabel(averageSentiment);
  const sentimentColor = getBarColor(averageSentiment);

  // Generate mock data if no data provided (for demo purposes)
  const chartData = useMemo(() => {
    if (data.length > 0) return data;
    
    // Generate sample data if none provided
    const sampleData: SentimentDataPoint[] = [];
    const days = 14;
    const endDate = new Date();
    
    for (let i = days - 1; i >= 0; i--) {
      const date = new Date(endDate);
      date.setDate(date.getDate() - i);
      
      // Random sentiment between -0.5 and 0.5
      const sentiment = (Math.random() - 0.5);
      const volume = Math.floor(Math.random() * 50) + 10;
      
      sampleData.push({
        date: date.toISOString().split('T')[0],
        sentiment: Number(sentiment.toFixed(2)),
        volume
      });
    }
    return sampleData;
  }, [data]);

  return (
    <div className="w-full">
      <div className="mb-4">
        <div className="flex items-baseline gap-3">
          <span className="text-3xl font-semibold text-[#ececf1]">
            {sentimentLabel}
          </span>
          <span 
            className="text-lg font-medium"
            style={{ color: sentimentColor }}
          >
            {averageSentiment > 0 ? "+" : ""}{averageSentiment.toFixed(2)}
          </span>
        </div>
        <div className="mt-1 flex items-center gap-2 text-sm text-[#9ca3af]">
          <span className="font-medium text-[#ececf1]">{symbol}</span>
          <span>·</span>
          <span>News Sentiment</span>
          {metadata?.total_articles && (
            <>
              <span>·</span>
              <span>{metadata.total_articles} articles</span>
            </>
          )}
        </div>
      </div>

      <div className="relative rounded-xl bg-[#2f2f2f]/50 p-4">
        <div className="h-[240px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={chartData}
              margin={{ top: 20, right: 0, left: 0, bottom: 0 }}
            >
              <XAxis
                dataKey="date"
                tickFormatter={formatDate}
                stroke="#4d4d4f"
                fontSize={11}
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                minTickGap={30}
              />
              
              <YAxis
                stroke="#4d4d4f"
                fontSize={11}
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                domain={yDomain}
                width={50}
                tickFormatter={(val) => val.toFixed(2)}
              />

              <ReferenceLine 
                y={0} 
                stroke="#4d4d4f" 
                strokeOpacity={0.5}
              />

              <ReferenceLine 
                y={averageSentiment} 
                stroke={sentimentColor} 
                strokeDasharray="3 3"
                strokeOpacity={0.8}
              />
              
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload as SentimentDataPoint;
                    const sentimentColor = getBarColor(data.sentiment);
                    
                    return (
                      <div className="rounded-lg border border-[#4d4d4f] bg-[#2f2f2f] p-3 shadow-lg">
                        <p className="text-xs text-[#9ca3af] mb-1">
                          {formatDate(data.date)}
                        </p>
                        <p className="text-lg font-semibold" style={{ color: sentimentColor }}>
                          {data.sentiment > 0 ? "+" : ""}{data.sentiment.toFixed(3)}
                        </p>
                        <p className="text-xs text-[#9ca3af]">
                          {getSentimentLabel(data.sentiment)}
                        </p>
                        {data.volume > 0 && (
                          <p className="text-xs text-[#6b7280] mt-1">
                            {data.volume} mentions
                          </p>
                        )}
                      </div>
                    );
                  }
                  return null;
                }}
                cursor={{ fill: "#404040", opacity: 0.3 }}
              />
              
              <Bar
                dataKey="sentiment"
                radius={[3, 3, 3, 3]}
                maxBarSize={35}
                minPointSize={3}
              >
                {chartData.map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={getBarColor(entry.sentiment ?? 0)}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Legend */}
        <div className="mt-4 flex flex-wrap items-center justify-center gap-x-4 gap-y-2 border-t border-[#4d4d4f]/30 pt-3">
          <div className="flex items-center gap-1.5">
            <div className="h-2 w-2 rounded-full bg-green-500" />
            <span className="text-xs text-[#9ca3af]">Bullish (&gt;0.2)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="h-2 w-2 rounded-full bg-green-400" />
            <span className="text-xs text-[#9ca3af]">Positive (0-0.2)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="h-2 w-2 rounded-full bg-gray-400" />
            <span className="text-xs text-[#9ca3af]">Neutral (0)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="h-2 w-2 rounded-full bg-red-400" />
            <span className="text-xs text-[#9ca3af]">Negative (-0.2-0)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="h-2 w-2 rounded-full bg-red-500" />
            <span className="text-xs text-[#9ca3af]">Bearish (&lt;-0.2)</span>
          </div>
        </div>

        {/* Data Table */}
        {chartData.length > 0 && (
          <div className="mt-4 border-t border-[#4d4d4f]/30 pt-4">
            <h4 className="mb-3 text-sm font-medium text-[#ececf1]">Daily Sentiment Trend</h4>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-[#4d4d4f]/50">
                    <th className="pb-2 text-left text-xs font-medium text-[#9ca3af]">Date (UTC)</th>
                    <th className="pb-2 text-right text-xs font-medium text-[#9ca3af]">Score</th>
                    <th className="pb-2 text-right text-xs font-medium text-[#9ca3af]">Label</th>
                  </tr>
                </thead>
                <tbody>
                  {chartData.slice().reverse().map((row, idx) => {
                    const sentimentValue = row.sentiment ?? 0;
                    const isPos = sentimentValue > 0;
                    return (
                      <tr key={idx} className="border-b border-[#4d4d4f]/20 last:border-0">
                        <td className="py-2 text-[#ececf1]">{row.date}</td>
                        <td className={`py-2 text-right font-mono ${isPos ? 'text-green-400' : sentimentValue < 0 ? 'text-red-400' : 'text-[#9ca3af]'}`}>
                          {isPos ? '+' : ''}{sentimentValue.toFixed(3)}
                        </td>
                        <td className="py-2 text-right text-[#9ca3af]">
                          {getSentimentLabel(sentimentValue)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
