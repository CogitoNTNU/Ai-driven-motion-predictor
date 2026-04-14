import { useState, useEffect } from "react";

interface ModelPrediction {
  name: string;
  signal: "BUY" | "SELL" | "HOLD";
  target_price: number;
  change_pct: number;
  confidence: number;
}

interface EnsemblePrediction {
  signal: "BUY" | "SELL" | "HOLD";
  votes: number;
  total: number;
}

interface PredictionsResponse {
  symbol: string;
  current_price: number;
  models: ModelPrediction[];
  ensemble: EnsemblePrediction;
}

interface PredictionsPanelProps {
  ticker: string;
}

function SignalBadge({ signal }: { signal: "BUY" | "SELL" | "HOLD" }) {
  const styles = {
    BUY: "bg-green-500/20 text-green-400 border border-green-500/30",
    SELL: "bg-red-500/20 text-red-400 border border-red-500/30",
    HOLD: "bg-yellow-500/20 text-yellow-400 border border-yellow-500/30",
  };
  return (
    <span
      className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-semibold ${styles[signal]}`}
    >
      {signal}
    </span>
  );
}

function ConfidenceBar({ value }: { value: number }) {
  return (
    <div className="h-1.5 w-full rounded-full bg-[#404040]">
      <div
        className="h-full rounded-full bg-[#10a37f] transition-all duration-300"
        style={{ width: `${Math.round(value * 100)}%` }}
      />
    </div>
  );
}

function ModelRow({ model }: { model: ModelPrediction }) {
  const isPositive = model.change_pct >= 0;
  return (
    <div className="py-3">
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="font-medium text-white text-sm">{model.name}</span>
          <SignalBadge signal={model.signal} />
        </div>
        <div className="text-right">
          <span className="text-sm font-semibold text-white">
            {model.target_price != null ? `$${model.target_price.toFixed(2)}` : "—"}
          </span>
          <span
            className={`ml-1.5 text-xs font-medium ${isPositive ? "text-green-400" : "text-red-400"}`}
          >
            {isPositive ? "+" : ""}
            {model.change_pct.toFixed(2)}%
          </span>
        </div>
      </div>
      <div className="flex items-center gap-2">
        <ConfidenceBar value={model.confidence} />
        <span className="shrink-0 text-xs text-[#9ca3af]">
          {Math.round(model.confidence * 100)}%
        </span>
      </div>
    </div>
  );
}

export function PredictionsPanel({ ticker }: PredictionsPanelProps) {
  const [data, setData] = useState<PredictionsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    fetch(
      `${import.meta.env.VITE_API_URL ?? ""}/api/stock/${ticker}/predictions`
    )
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json() as Promise<PredictionsResponse>;
      })
      .then((result) => {
        if (!cancelled) {
          setData(result);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          console.error("PredictionsPanel fetch error:", err);
          setError("Failed to load predictions");
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [ticker]);

  return (
    <div className="rounded-xl border border-[#4d4d4f] bg-[#2f2f2f] p-4">
      <h2 className="mb-1 text-sm font-semibold text-white">Model Predictions</h2>

      {loading ? (
        <div className="space-y-3 pt-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="space-y-2 py-3 border-b border-[#4d4d4f]/50 last:border-b-0">
              <div className="animate-pulse h-4 w-40 rounded bg-[#404040]" />
              <div className="animate-pulse h-2 rounded bg-[#404040]" />
            </div>
          ))}
        </div>
      ) : error ? (
        <p className="mt-3 text-sm text-[#9ca3af]">{error}</p>
      ) : data ? (
        <>
          <div className="divide-y divide-[#4d4d4f]/50">
            {data.models.map((model) => (
              <ModelRow key={model.name} model={model} />
            ))}
          </div>

          <div className="mt-1 border-t border-[#4d4d4f] pt-3">
            <div className="flex items-center justify-between rounded-lg bg-[#404040]/40 px-3 py-2.5">
              <div className="flex items-center gap-2">
                <span className="font-semibold text-white text-sm">Ensemble</span>
                <SignalBadge signal={data.ensemble.signal} />
              </div>
              <span className="text-xs text-[#9ca3af]">
                {data.ensemble.votes}/{data.ensemble.total} votes
              </span>
            </div>
          </div>
        </>
      ) : null}
    </div>
  );
}
