import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import type { ChartData } from "@/types/chat";

interface KaareStockChartProps {
  chart: ChartData;
  showHeader?: boolean;
}

export function KaareStockChart({ chart, showHeader = true }: KaareStockChartProps) {
  const { symbol, data, metadata } = chart;

  const growth = metadata?.percentage_growth ?? 0;
  const isPositive = growth >= 0;
  const startPrice = metadata?.start_price ?? 0;
  const endPrice = metadata?.end_price ?? 0;

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  };

  const formatPrice = (price: number) => `$${price?.toFixed(2) ?? "0.00"}`;

  const formatGrowth = (val: number) => {
    const sign = val >= 0 ? "+" : "";
    return `${sign}${val?.toFixed(2) ?? "0.00"}%`;
  };

  const mainColor = isPositive ? "#22c55e" : "#ef4444";
  const bgGradient = isPositive 
    ? "from-green-500/5 to-transparent" 
    : "from-red-500/5 to-transparent";

  // Calculate min/max for Y-axis padding
  const prices = data?.map(d => d.price ?? 0).filter(p => p > 0) || [];
  const minPrice = prices.length > 0 ? Math.min(...prices) : 0;
  const maxPrice = prices.length > 0 ? Math.max(...prices) : 0;
  const priceRange = maxPrice - minPrice;
  const yDomainMin = Math.max(0, minPrice - priceRange * 0.1);
  const yDomainMax = maxPrice + priceRange * 0.1;

  return (
    <div className="w-full">
      {showHeader && (
        <div className="mb-4">
          <div className="flex items-baseline gap-3">
            <span className="text-4xl font-semibold text-[#ececf1]">
              {formatPrice(endPrice)}
            </span>
            <span className={`text-lg font-medium ${isPositive ? "text-green-400" : "text-red-400"}`}>
              {formatGrowth(growth)}
            </span>
          </div>
          <div className="mt-1 flex items-center gap-2 text-sm text-[#9ca3af]">
            <span className="font-medium text-[#ececf1]">{symbol}</span>
            <span>·</span>
            <span>{formatDate(metadata?.start_date)} - {formatDate(metadata?.end_date)}</span>
          </div>
        </div>
      )}

      <div className={`relative rounded-xl bg-gradient-to-b ${bgGradient} p-4`}>
        <div className="h-[280px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart
              data={data}
              margin={{ top: 10, right: 0, left: 0, bottom: 0 }}
            >
              <defs>
                <linearGradient id={`kaare-gradient-${symbol}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={mainColor} stopOpacity={0.3}/>
                  <stop offset="100%" stopColor={mainColor} stopOpacity={0}/>
                </linearGradient>
              </defs>
              
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
                tickFormatter={formatPrice}
                stroke="#4d4d4f"
                fontSize={11}
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                domain={[yDomainMin, yDomainMax]}
                width={50}
              />

              <ReferenceLine 
                y={startPrice} 
                stroke="#4d4d4f" 
                strokeDasharray="3 3" 
                strokeOpacity={0.5}
              />
              
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    const price = data.price;
                    const priceChange = ((price - startPrice) / startPrice) * 100;
                    const isUp = priceChange >= 0;
                    
                    return (
                      <div className="rounded-lg border border-[#4d4d4f] bg-[#2f2f2f] p-3 shadow-lg">
                        <p className="text-xs text-[#9ca3af] mb-1">
                          {formatDate(data.date)}
                        </p>
                        <p className="text-lg font-semibold text-[#ececf1]">
                          {formatPrice(price)}
                        </p>
                        <p className={`text-xs font-medium ${isUp ? "text-green-400" : "text-red-400"}`}>
                          {isUp ? "+" : ""}{priceChange.toFixed(2)}%
                        </p>
                      </div>
                    );
                  }
                  return null;
                }}
                cursor={{ stroke: mainColor, strokeWidth: 1, strokeDasharray: "4 4" }}
              />
              
              <Area
                type="monotone"
                dataKey="price"
                stroke={mainColor}
                strokeWidth={2}
                fill={`url(#kaare-gradient-${symbol})`}
                fillOpacity={1}
                dot={false}
                activeDot={{ r: 5, fill: mainColor, strokeWidth: 2, stroke: "#212121" }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Stats row */}
        <div className="mt-4 flex items-center justify-between border-t border-[#4d4d4f]/30 pt-3">
          <div className="flex gap-6">
            <div>
              <p className="text-xs text-[#9ca3af]">Open</p>
              <p className="text-sm font-medium text-[#ececf1]">{formatPrice(startPrice)}</p>
            </div>
            <div>
              <p className="text-xs text-[#9ca3af]">High</p>
              <p className="text-sm font-medium text-[#ececf1]">{formatPrice(maxPrice)}</p>
            </div>
            <div>
              <p className="text-xs text-[#9ca3af]">Low</p>
              <p className="text-sm font-medium text-[#ececf1]">{formatPrice(minPrice)}</p>
            </div>
          </div>
          <div className="text-right">
            <p className="text-xs text-[#9ca3af]">{metadata?.trading_days ?? 0} days</p>
          </div>
        </div>
      </div>
    </div>
  );
}
