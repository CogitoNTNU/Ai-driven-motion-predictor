import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import type { ChartData } from "@/types/chat";

interface StockPieChartProps {
  chart: ChartData;
}

const COLORS = ["#22c55e", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6"];

export function StockPieChart({ chart }: StockPieChartProps) {
  const { symbol, metadata } = chart;

  const formatPrice = (price: number) => `$${price.toFixed(2)}`;

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  };

  const pieData = [
    { name: "Start Price", value: metadata.start_price },
    { name: "End Price", value: metadata.end_price },
    { name: "Absolute Growth", value: Math.abs(metadata.absolute_growth) },
  ];

  return (
    <div className="w-full rounded-lg border bg-card p-4 my-4">
      <div className="mb-4">
        <h3 className="text-lg font-semibold">{symbol} Price Distribution</h3>
        <p className="text-sm text-muted-foreground">
          {formatDate(metadata.start_date)} - {formatDate(metadata.end_date)}
          {" "}({metadata.trading_days} trading days)
        </p>
      </div>

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
              formatter={(value) => [formatPrice(Number(value) || 0), "Price"]}
            />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-4 flex items-center justify-between text-sm">
        <div>
          <span className="text-muted-foreground">Start: </span>
          <span className="font-medium">
            {formatPrice(metadata.start_price)}
          </span>
        </div>
        <div
          className={`font-semibold ${
            metadata.percentage_growth >= 0
              ? "text-green-600"
              : "text-red-600"
          }`}
        >
          {metadata.percentage_growth >= 0 ? "+" : ""}
          {metadata.percentage_growth.toFixed(2)}%
        </div>
        <div>
          <span className="text-muted-foreground">End: </span>
          <span className="font-medium">{formatPrice(metadata.end_price)}</span>
        </div>
      </div>
    </div>
  );
}