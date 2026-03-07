/**
 * Chart data structure received from the backend
 */
export interface ChartData {
  chart_id: string;
  symbol: string;
  type: "line_chart" | string;
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
 * Chat message structure
 */
export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  charts?: ChartData[];
}

/**
 * SSE event types from the backend
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
