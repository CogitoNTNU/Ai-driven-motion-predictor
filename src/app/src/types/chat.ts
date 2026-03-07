/**
 * Chart data structure received from the backend via tool invocation
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
 * AI SDK v5 Message Part Types
 * Tool invocations are the primary way charts are transferred from sub-agents
 */
export type MessagePart = TextPart | TextDeltaPart | ToolInvocationPart | FinishPart;

/**
 * Text content from the LLM (complete text part)
 */
export interface TextPart {
  type: "text";
  text: string;
}

/**
 * Text delta part during streaming (incremental text from the LLM)
 */
export interface TextDeltaPart {
  type: "text-delta";
  text: string;
}

/**
 * Tool invocation part for AI SDK v5
 * Charts are transferred as tool outputs from sub-agents
 */
export interface ToolInvocationPart {
  type: `tool-${string}`;
  toolCallId: string;
  toolName: string;
  state: "input-available" | "output-available" | "output-error";
  input?: Record<string, unknown>;
  output?: ChartData;
  errorText?: string;
}

/**
 * Specific tool type for get_stock_growth tool
 */
export type StockGrowthToolPart = ToolInvocationPart & {
  type: "tool-get_stock_growth";
  toolName: "get_stock_growth";
  output?: ChartData;
};

/**
 * Finish signal from the stream
 */
export interface FinishPart {
  type: "finish";
  finishReason: "stop" | "length" | "error";
  usage?: {
    promptTokens: number;
    completionTokens: number;
  };
}

/**
 * AI SDK v5 Message structure with parts
 */
export interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  parts: MessagePart[];
  createdAt?: number;
}

/**
 * Type guard to check if a part is a tool invocation with chart output
 */
export function isStockGrowthToolPart(part: MessagePart): part is StockGrowthToolPart {
  return part.type === "tool-get_stock_growth" && part.state === "output-available";
}

/**
 * Type guard to check if a part is a text part
 */
export function isTextPart(part: MessagePart): part is TextPart {
  return part.type === "text";
}

/**
 * Legacy SSE event types (for backward compatibility during migration)
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
