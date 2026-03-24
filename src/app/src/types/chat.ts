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
 * Tool call data structure (wrapped inside data field per AI SDK schema)
 */
export interface ToolCallData {
  toolCallId: string;
  toolName: string;
  input: {
    _agentName?: string;
    [key: string]: unknown;
  };
}

/**
 * Tool result data structure (wrapped inside data field per AI SDK schema)
 */
export interface ToolResultData {
  toolCallId: string;
  toolName: string;
  output: string;
}

/**
 * Tool call part - represents a tool invocation from an agent
 * Backend sends these as custom data-tool-call data parts
 * AI SDK schema: { type: "data-*", id?: string, data: unknown, transient?: boolean }
 */
export interface ToolCallPart {
  type: "data-tool-call";
  id?: string;
  data: ToolCallData;
}

/**
 * Tool result part - represents the result of a tool execution
 * Backend sends these as custom data-tool-result data parts
 * AI SDK schema: { type: "data-*", id?: string, data: unknown, transient?: boolean }
 */
export interface ToolResultPart {
  type: "data-tool-result";
  id?: string;
  data: ToolResultData;
}

/**
 * AI SDK v5 Message Part Types
 * Includes text, data-chart, and tool parts
 */
export type MessagePart = TextPart | DataChartPart | ToolCallPart | ToolResultPart;

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
  return part.type === "data-chart" && (part as DataChartPart).data?.tool_name === "get_stock_growth";
}

/**
 * Type guard to check if a part is a tool call part
 */
export function isToolCallPart(part: MessagePart): part is ToolCallPart {
  return part.type === "data-tool-call";
}

/**
 * Type guard to check if a part is a tool result part
 */
export function isToolResultPart(part: MessagePart): part is ToolResultPart {
  return part.type === "data-tool-result";
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
