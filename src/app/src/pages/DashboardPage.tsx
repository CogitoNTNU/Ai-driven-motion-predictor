import { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import { HeaderBar } from "@/components/HeaderBar";
import { StockChartPanel } from "@/components/StockChartPanel";
import { ChatPanel } from "@/components/ChatPanel";
import { PredictionsPanel } from "@/components/PredictionsPanel";
import { SentimentGauge } from "@/components/SentimentGauge";

interface StockSummary {
  symbol: string;
  name: string;
  price: number;
  change: number;
  change_pct: number;
  currency: string;
}

export function DashboardPage() {
  const { ticker } = useParams<{ ticker: string }>();
  const symbol = ticker?.toUpperCase() ?? "";

  const [summary, setSummary] = useState<StockSummary | null>(null);

  useEffect(() => {
    if (!symbol) return;
    fetch(
      `${import.meta.env.VITE_API_URL ?? ""}/api/stock/${symbol}/summary`
    )
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json() as Promise<StockSummary>;
      })
      .then((data) => setSummary(data))
      .catch((err) => console.error("DashboardPage summary fetch error:", err));
  }, [symbol]);

  const stockContext = summary
    ? `${symbol} - ${summary.name}, current price $${summary.price.toFixed(2)}, change ${summary.change >= 0 ? "+" : ""}${summary.change.toFixed(2)} (${summary.change_pct.toFixed(2)}%)`
    : symbol;

  return (
    <div className="h-screen overflow-hidden bg-[#212121]">
      <HeaderBar ticker={symbol} summary={summary} />

      {/* Main grid */}
      <div
        className="grid gap-4 p-4"
        style={{
          height: "calc(100vh - 56px)",
          marginTop: "56px",
          gridTemplateColumns: "3fr 2fr",
        }}
      >
        {/* Left column */}
        <div className="grid min-h-0 gap-4" style={{ gridTemplateRows: "1fr 1fr" }}>
          <StockChartPanel ticker={symbol} />
          <ChatPanel ticker={symbol} stockContext={stockContext} />
        </div>

        {/* Right column */}
        <div className="flex min-h-0 flex-col gap-4 overflow-y-auto">
          <PredictionsPanel ticker={symbol} />
          <SentimentGauge ticker={symbol} />
        </div>
      </div>
    </div>
  );
}
