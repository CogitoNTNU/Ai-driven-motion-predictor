import { useNavigate } from "react-router-dom";
import { ArrowLeft } from "lucide-react";

interface StockSummary {
  name: string;
  price: number;
  change: number;
  change_pct: number;
}

interface HeaderBarProps {
  ticker: string;
  summary: StockSummary | null;
}

export function HeaderBar({ ticker, summary }: HeaderBarProps) {
  const navigate = useNavigate();
  const isPositive = summary ? summary.change >= 0 : true;

  return (
    <header className="fixed top-0 left-0 right-0 z-50 h-14 border-b border-[#4d4d4f] bg-[#212121]/95 backdrop-blur supports-[backdrop-filter]:bg-[#212121]/80">
      <div className="flex h-full items-center px-4 gap-4">
        {/* Back button */}
        <button
          onClick={() => navigate("/")}
          className="flex items-center justify-center rounded-lg p-1.5 text-[#9ca3af] transition-colors hover:bg-[#2f2f2f] hover:text-white"
          aria-label="Back to search"
        >
          <ArrowLeft className="h-5 w-5" />
        </button>

        {/* Center: ticker and name */}
        <div className="flex-1 text-center">
          {summary ? (
            <span className="font-medium text-white">
              {ticker} · {summary.name}
            </span>
          ) : (
            <div className="mx-auto h-5 w-48 animate-pulse rounded bg-[#404040]" />
          )}
        </div>

        {/* Right: price and change */}
        <div className="flex items-center gap-2 text-sm">
          {summary ? (
            <>
              <span className="font-semibold text-white">
                ${summary.price.toFixed(2)}
              </span>
              <span
                className={`font-medium ${isPositive ? "text-green-400" : "text-red-400"}`}
              >
                {isPositive ? "+" : ""}
                {summary.change.toFixed(2)} ({isPositive ? "+" : ""}
                {summary.change_pct.toFixed(2)}%)
              </span>
            </>
          ) : (
            <div className="h-5 w-28 animate-pulse rounded bg-[#404040]" />
          )}
        </div>
      </div>
    </header>
  );
}
