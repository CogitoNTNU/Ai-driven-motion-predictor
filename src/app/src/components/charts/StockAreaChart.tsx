import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { ChartData } from "@/types/chat";

interface StockAreaChartProps {
  chart: ChartData;
}

export function StockAreaChart({ chart }: StockAreaChartProps) {
  const { symbol, data, metadata } = chart;

  // Safely get growth value with default
  const growth = metadata?.percentage_growth ?? 0;
  const isPositive = growth >= 0;

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  };

  const formatPrice = (price: number) => `$${price?.toFixed(2) ?? "0.00"}`;

  const areaColor = isPositive ? "#22c55e" : "#ef4444";

  const formatGrowth = (val: number) => {
    const sign = val >= 0 ? "+" : "";
    return `${sign}${val?.toFixed(2) ?? "0.00"}%`;
  };

  return (
    <Card className="border-[#4d4d4f] bg-[#2f2f2f] shadow-none">
      <CardHeader>
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
          <div>
            <CardTitle className="text-xl text-[#ececf1]">{symbol} Stock Price</CardTitle>
            <CardDescription className="text-[#9ca3af]">
              {formatDate(metadata?.start_date)} - {formatDate(metadata?.end_date)}
              {" "}({metadata?.trading_days ?? 0} trading days)
            </CardDescription>
          </div>
          <Badge 
            variant="outline" 
            className={cn(
              "w-fit",
              isPositive 
                ? "border-green-500/50 text-green-400" 
                : "border-red-500/50 text-red-400"
            )}
          >
            {formatGrowth(growth)}
          </Badge>
        </div>
      </CardHeader>

      <CardContent>
        <div className="h-[300px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <defs>
                <linearGradient id={`color-${symbol}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={areaColor} stopOpacity={0.3}/>
                  <stop offset="95%" stopColor={areaColor} stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#4d4d4f" strokeOpacity={0.5} />
              <XAxis
                dataKey="date"
                tickFormatter={formatDate}
                stroke="#9ca3af"
                fontSize={12}
                tickLine={false}
                axisLine={{ stroke: "#4d4d4f" }}
              />
              <YAxis
                tickFormatter={formatPrice}
                stroke="#9ca3af"
                fontSize={12}
                tickLine={false}
                axisLine={{ stroke: "#4d4d4f" }}
                domain={["auto", "auto"]}
              />
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    return (
                      <div className="rounded-lg border border-[#4d4d4f] bg-[#2f2f2f] p-3 shadow-lg space-y-1">
                        <p className="text-sm font-semibold text-[#ececf1]">
                          {formatDate(data.date)}
                        </p>
                        <p className="text-base font-bold text-[#ececf1]">
                          {formatPrice(data.price)}
                        </p>
                      </div>
                    );
                  }
                  return null;
                }}
                cursor={{ stroke: areaColor, strokeWidth: 1, strokeDasharray: "4 4" }}
              />
              <Area
                type="monotone"
                dataKey="price"
                stroke={areaColor}
                strokeWidth={2}
                fillOpacity={1}
                fill={`url(#color-${symbol})`}
                dot={false}
                activeDot={{ r: 6, fill: areaColor, strokeWidth: 2, stroke: "#212121" }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </CardContent>

      <CardFooter className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Badge variant="secondary" className="bg-[#404040] text-[#ececf1]">
            Start: {formatPrice(metadata?.start_price)}
          </Badge>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="secondary" className="bg-[#404040] text-[#ececf1]">
            End: {formatPrice(metadata?.end_price)}
          </Badge>
        </div>
        <Badge 
          variant={isPositive ? "default" : "destructive"} 
          className={cn(
            "font-semibold",
            isPositive 
              ? "bg-green-500/20 text-green-400 hover:bg-green-500/30" 
              : "bg-red-500/20 text-red-400 hover:bg-red-500/30"
          )}
        >
          {formatGrowth(growth)}
        </Badge>
      </CardFooter>
    </Card>
  );
}
