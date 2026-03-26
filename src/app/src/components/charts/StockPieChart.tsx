import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { ChartData } from "@/types/chat";

interface StockPieChartProps {
  chart: ChartData;
}

const COLORS = ["#22c55e", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6"];

export function StockPieChart({ chart }: StockPieChartProps) {
  const { symbol, metadata } = chart;

  // Safely get growth value with default
  const growth = metadata?.percentage_growth ?? 0;
  const isPositive = growth >= 0;

  const formatPrice = (price: number) => `$${price?.toFixed(2) ?? "0.00"}`;

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  };

  const formatGrowth = (val: number) => {
    const sign = val >= 0 ? "+" : "";
    return `${sign}${val?.toFixed(2) ?? "0.00"}%`;
  };

  const pieData = [
    { name: "Start Price", value: metadata?.start_price ?? 0 },
    { name: "End Price", value: metadata?.end_price ?? 0 },
    { name: "Absolute Growth", value: Math.abs(metadata?.absolute_growth ?? 0) },
  ];

  return (
    <Card className="border-[#4d4d4f] bg-[#2f2f2f] shadow-none">
      <CardHeader>
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
          <div>
            <CardTitle className="text-xl text-[#ececf1]">{symbol} Price Distribution</CardTitle>
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
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={(_entry) => {
                  const e = _entry as { name?: string; value?: number };
                  return `${e.name ?? ""}: ${formatPrice(e.value ?? 0)}`;
                }}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {pieData.map((_entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={COLORS[index % COLORS.length]}
                  />
                ))}
              </Pie>
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0];
                    return (
                      <div className="rounded-lg border border-[#4d4d4f] bg-[#2f2f2f] p-3 shadow-lg">
                        <p className="text-sm font-semibold text-[#ececf1]">
                          {data.name}
                        </p>
                        <p className="text-base font-bold text-[#ececf1]">
                          {formatPrice(Number(data.value) || 0)}
                        </p>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Legend 
                wrapperStyle={{ color: '#9ca3af' }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </CardContent>

      <CardFooter className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Badge variant="secondary" className="bg-[#404040] text-[#ececf1]">
            Start: {formatPrice(metadata?.start_price ?? 0)}
          </Badge>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="secondary" className="bg-[#404040] text-[#ececf1]">
            End: {formatPrice(metadata?.end_price ?? 0)}
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
