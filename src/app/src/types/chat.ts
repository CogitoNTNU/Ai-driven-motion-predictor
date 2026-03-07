/**
 * Chart data structure received from the backend via data-chart parts
 */
export interface ChartData {
  tool_name: string;
  chart_id: string;
  symbol: string;
  type: "line_chart" | "bar_chart" | "area_chart" | "pie_chart" | string;
  data: Array<{
    date: string;
    price: number;
  }>;
  metadata: {
    start_date: string;
    end_date: string;
    start_price: number;
    end_price: number;
    absolute_growth: number;
    percentage_growth: number;
    trading_days: number;
  };
}

/**
 * Text content part from the LLM
 */
export interface TextPart {
  type: "text";
  text: string;
}

/**
 * Custom data part for charts (from Data Stream Protocol)
 */
export interface DataChartPart {
  type: "data-chart";
  data: ChartData;
}

/**
 * AI SDK v5 Message Part Types
 */
export type MessagePart = TextPart | DataChartPart;

/**
 * AI SDK v5 Message structure
 */
export interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  parts: MessagePart[];
  createdAt?: number;
}

/**
 * Type guard to check if a part is a data-chart part
 */
export function isChartPart(part: MessagePart): part is DataChartPart {
  return part.type === "data-chart";
}

/**
 * Type guard to check if a part is a text part
 */
export function isTextPart(part: MessagePart): part is TextPart {
  return part.type === "text";
}

/**
 * Type guard specifically for stock growth chart parts
 */
export function isStockGrowthToolPart(part: MessagePart): part is DataChartPart {
  return part.type === "data-chart" && (part as any).data?.tool_name === "get_stock_growth";
}

/**
 * Legacy SSE event types (for backward compatibility)
 */
export type SSEEventType = "token" | "chart" | "done";

export interface TokenEvent {
  type: "token";
  content: string;
  role: string;
}

export interface ChartEvent {
  type: "chart";
  chart: ChartData;
}

export interface DoneEvent {
  type: "done";
  timestamp: string;
}

export type SSEEvent = TokenEvent | ChartEvent | DoneEvent;
