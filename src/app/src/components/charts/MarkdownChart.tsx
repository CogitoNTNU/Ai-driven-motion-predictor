import type { Components } from "react-markdown";
import type { ReactNode } from "react";
import { AlertCircle } from "lucide-react";
import { ChartRenderer } from "./ChartRenderer";
import type { ChartData } from "@/types/chat";

/**
 * Splits a text string on [Chart: SYMBOL] markers, returning an array of
 * strings and resolved ChartData objects (or the original marker string when
 * no matching chart is found).
 *
 * When multiple charts share the same symbol (e.g. different date ranges),
 * the last one in the array is used so that the most recently received data
 * wins.
 */
function splitChartMarkers(
  text: string,
  charts: ChartData[],
): Array<string | { symbol: string; chart: ChartData | null }> {
  // Regex created locally to avoid shared `lastIndex` state across renders
  const chartMarkerRe = /\[Chart:\s*([A-Z0-9.]+)\]/g;
  const parts: Array<string | { symbol: string; chart: ChartData | null }> = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = chartMarkerRe.exec(text)) !== null) {
    const before = text.slice(lastIndex, match.index);
    if (before) parts.push(before);

    const symbol = match[1];
    // Use findLast so that when multiple charts share a symbol (different date
    // ranges), the most recently received one is displayed.
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

/**
 * Extracts the plain text content from react-markdown's children nodes
 * so we can run the regex over the raw text before it gets rendered.
 */
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

/**
 * Returns a react-markdown `components` map that intercepts paragraph nodes,
 * detects [Chart: SYMBOL] markers in their text content, and renders the
 * corresponding chart (looked up by symbol from `charts`) in-place.
 *
 * Non-chart paragraphs are rendered normally.
 */
export function makeMarkdownComponents(charts: ChartData[]): Components {
  return {
    p({ children }) {
      const text = childrenToText(children);

      // Only process paragraphs that contain at least one chart marker
      if (!/\[Chart:\s*[A-Z0-9.]+\]/i.test(text)) {
        return <p>{children}</p>;
      }

      const parts = splitChartMarkers(text, charts);

      // If every part is plain text (no markers matched), render normally
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
