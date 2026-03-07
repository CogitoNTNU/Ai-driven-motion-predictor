import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { ChartData } from "@/types/chat";

interface StockPriceChartProps {
  chart: ChartData;
}

export function StockPriceChart({ chart }: StockPriceChartProps) {
  const { symbol, data, metadata } = chart;

  // Format date for display
  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
    });
  };

  // Format price for Y-axis
  const formatPrice = (price: number) => {
    return `$${price.toFixed(2)}`;
  };

  // Determine line color based on growth
  const lineColor =
    metadata.percentage_growth >= 0 ? "#22c55e" : "#ef4444"; // green-500 : red-500

  return (
    <div className="w-full rounded-lg border bg-card p-4 my-4">
      <div className="mb-4">
        <h3 className="text-lg font-semibold">{symbol} Stock Price</h3>
        <p className="text-sm text-muted-foreground">
          {formatDate(metadata.start_date)} - {formatDate(metadata.end_date)}
          {" "}({metadata.trading_days} trading days)
        </p>
      </div>

      <div className="h-[300px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data}
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="date"
              tickFormatter={formatDate}
              stroke="#6b7280"
              fontSize={12}
              tickLine={false}
            />
            <YAxis
              tickFormatter={formatPrice}
              stroke="#6b7280"
              fontSize={12}
              tickLine={false}
              domain={["auto", "auto"]}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload;
                  return (
                    <div className="rounded-lg border bg-background p-2 shadow-sm">
                      <p className="text-sm font-medium">
                        {formatDate(data.date)}
                      </p>
                      <p className="text-sm text-muted-foreground">
                        Price: {formatPrice(data.price)}
                      </p>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Line
              type="monotone"
              dataKey="price"
              stroke={lineColor}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 6, fill: lineColor }}
            />
          </LineChart>
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
