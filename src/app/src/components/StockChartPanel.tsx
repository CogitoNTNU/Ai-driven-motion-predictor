import { useState, useEffect } from "react";
import { KaareStockChart } from "@/components/charts/KaareStockChart";
import type { ChartData } from "@/types/chat";

type Range = "1W" | "1M" | "3M" | "1Y";

interface HistoryResponse {
  symbol: string;
  range: string;
  data: Array<{ date: string; price: number }>;
  metadata: {
    start_date: string;
    end_date: string;
    start_price: number;
    end_price: number;
    percentage_growth: number;
    trading_days: number;
  };
}

interface StockChartPanelProps {
  ticker: string;
}

const RANGES: Range[] = ["1W", "1M", "3M", "1Y"];

export function StockChartPanel({ ticker }: StockChartPanelProps) {
  const [range, setRange] = useState<Range>("1M");
  const [chartData, setChartData] = useState<ChartData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    fetch(
      `${import.meta.env.VITE_API_URL ?? ""}/api/stock/${ticker}/history?range=${range}`
    )
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json() as Promise<HistoryResponse>;
      })
      .then((data) => {
        if (cancelled) return;
        const chart: ChartData = {
          tool_name: "get_stock_growth",
          chart_id: `${ticker}-${range}`,
          symbol: data.symbol,
          type: "area_chart",
          data: data.data,
          metadata: {
            start_date: data.metadata.start_date,
            end_date: data.metadata.end_date,
            start_price: data.metadata.start_price,
            end_price: data.metadata.end_price,
            percentage_growth: data.metadata.percentage_growth,
            trading_days: data.metadata.trading_days,
          },
        };
        setChartData(chart);
        setLoading(false);
      })
      .catch((err) => {
        if (cancelled) return;
        console.error("StockChartPanel fetch error:", err);
        setError("Failed to load price history");
        setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [ticker, range]);

  return (
    <div className="flex min-h-0 flex-col overflow-hidden rounded-xl border border-[#4d4d4f] bg-[#2f2f2f] p-4">
      {/* Header row */}
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-white">Price History</h2>
        <div className="flex gap-1">
          {RANGES.map((r) => (
            <button
              key={r}
              onClick={() => setRange(r)}
              className={`rounded px-2.5 py-1 text-xs font-medium transition-colors ${
                range === r
                  ? "bg-[#10a37f] text-white"
                  : "text-[#9ca3af] hover:text-white"
              }`}
            >
              {r}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      {loading ? (
        <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-hidden">
          <div className="h-8 w-32 animate-pulse rounded-lg bg-[#404040]" />
          <div className="min-h-0 flex-1 overflow-hidden rounded-xl bg-[#363636]">
            <div className="h-full w-full animate-pulse bg-[#404040]" />
          </div>
        </div>
      ) : error ? (
        <div className="flex h-[300px] items-center justify-center">
          <p className="text-sm text-[#9ca3af]">{error}</p>
        </div>
      ) : chartData ? (
        <div className="flex min-h-0 flex-1">
          <KaareStockChart chart={chartData} showHeader={true} />
        </div>
      ) : null}
    </div>
  );
}
