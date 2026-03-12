/* eslint-disable react-refresh/only-export-components */
import type { Components } from "react-markdown";
import type { ReactNode } from "react";
import { AlertCircle } from "lucide-react";
import { ChartRenderer } from "./ChartRenderer";
import type { ChartData } from "@/types/chat";

function splitChartMarkers(
  text: string,
  charts: ChartData[],
): Array<string | { symbol: string; chart: ChartData | null }> {
  const chartMarkerRe = /\[Chart:\s*([A-Z0-9.]+)\]/g;
  const parts: Array<string | { symbol: string; chart: ChartData | null }> =
    [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = chartMarkerRe.exec(text)) !== null) {
    const before = text.slice(lastIndex, match.index);
    if (before) parts.push(before);

    const symbol = match[1];
    const chart =
      charts.findLast((c) => c.symbol.toUpperCase() === symbol.toUpperCase()) ??
      null;
    parts.push({ symbol, chart });

    lastIndex = match.index + match[0].length;
  }

  const remaining = text.slice(lastIndex);
  if (remaining) parts.push(remaining);

  return parts;
}

function childrenToText(children: ReactNode): string {
  if (typeof children === "string") return children;
  if (typeof children === "number") return String(children);
  if (Array.isArray(children)) return children.map(childrenToText).join("");
  return "";
}

interface ChartUnavailableProps {
  symbol: string;
}

function ChartUnavailable({ symbol }: ChartUnavailableProps) {
  return (
    <span className="my-1 flex items-center gap-2 rounded-md border border-yellow-500/50 bg-yellow-500/10 px-3 py-2 text-sm text-yellow-700 dark:text-yellow-400">
      <AlertCircle className="h-4 w-4 shrink-0" />
      Chart unavailable: {symbol}
    </span>
  );
}

export function makeMarkdownComponents(charts: ChartData[]): Components {
  return {
    p({ children }) {
      const text = childrenToText(children);

      if (!/\[Chart:\s*[A-Z0-9.]+\]/i.test(text)) {
        return <p>{children}</p>;
      }

      const parts = splitChartMarkers(text, charts);

      const hasChartPart = parts.some((p) => typeof p !== "string");
      if (!hasChartPart) {
        return <p>{children}</p>;
      }

      return (
        <span className="block">
          {parts.map((part, i) => {
            if (typeof part === "string") {
              return part ? <span key={i}>{part}</span> : null;
            }
            if (part.chart) {
              return <ChartRenderer key={i} chart={part.chart} />;
            }
            return <ChartUnavailable key={i} symbol={part.symbol} />;
          })}
        </span>
      );
    },
  };
}
