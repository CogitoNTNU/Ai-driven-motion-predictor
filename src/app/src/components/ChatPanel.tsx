import { useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { ArrowUp, Square } from "lucide-react";
import { useChat } from "@/hooks/use-chat";
import { ChartRenderer } from "@/components/charts/ChartRenderer";
import { Streamdown } from "streamdown";
import { code } from "@streamdown/code";
import {
  isChartPart,
  isToolCallPart,
  isToolResultPart,
  type ChartData,
  type Message,
  type ToolCallPart,
  type ToolResultPart,
} from "@/types/chat";

function extractChartPairs(
  message: Message
): Array<{ stock?: ChartData; sentiment?: ChartData }> {
  if (!message.parts || message.parts.length === 0) return [];
  const charts = message.parts.filter(isChartPart).map((p) => p.data);
  if (charts.length === 0) return [];

  const bySymbol: Record<string, { stock?: ChartData; sentiment?: ChartData }> = {};
  for (const chart of charts) {
    if (
      chart.tool_name === "get_stock_growth" ||
      chart.tool_name === "get_current_price"
    ) {
      continue;
    }

    const symbol = chart.symbol;
    if (!bySymbol[symbol]) bySymbol[symbol] = {};
    if (chart.tool_name === "get_stock_news_sentiment") {
      bySymbol[symbol].sentiment = chart;
    } else {
      bySymbol[symbol].stock = chart;
    }
  }
  return Object.values(bySymbol);
}

function extractToolCalls(
  message: Message
): Array<ToolCallPart | ToolResultPart> {
  if (!message.parts || message.parts.length === 0) return [];
  return message.parts.filter(
    (part): part is ToolCallPart | ToolResultPart =>
      isToolCallPart(part) || isToolResultPart(part)
  );
}

function getToolCallSummary(toolCall: ToolCallPart | ToolResultPart): string {
  if (isToolCallPart(toolCall)) {
    const toolName = toolCall.data.toolName;
    const args = toolCall.data.input as Record<string, unknown>;
    if (toolName === "get_stock_growth") {
      return `Analyzing ${args.symbol || args.ticker || "stock"} stock growth`;
    }
    if (toolName === "get_current_price") {
      return `Fetching ${args.symbol || args.ticker || "stock"} price`;
    }
    if (toolName === "get_stock_news_sentiment") {
      return `Analyzing ${args.symbol || args.ticker || "stock"} sentiment`;
    }
    return toolName.replace(/_/g, " ");
  }
  if (isToolResultPart(toolCall)) {
    try {
      const data = JSON.parse(toolCall.data.output);
      const toolName = toolCall.data.toolName;
      if (toolName === "get_stock_growth" && data.percentage_growth !== undefined) {
        const sign = data.percentage_growth >= 0 ? "+" : "";
        return `${data.symbol || "Stock"}: ${sign}${data.percentage_growth.toFixed(1)}%`;
      }
      if (toolName === "get_current_price" && data.price !== undefined) {
        return `${data.symbol || "Stock"}: $${data.price.toFixed(2)}`;
      }
      if (
        toolName === "get_stock_news_sentiment" &&
        data.average_sentiment !== undefined
      ) {
        const sign = data.average_sentiment > 0 ? "+" : "";
        return `${data.symbol || "Stock"}: ${sign}${data.average_sentiment.toFixed(2)} sentiment`;
      }
    } catch {
      // Fall through
    }
    return "Complete";
  }
  return "";
}

interface CitationBadgeProps {
  index: number;
  toolCall: ToolCallPart | ToolResultPart;
}

function CitationBadge({ index, toolCall }: CitationBadgeProps) {
  const summary = getToolCallSummary(toolCall);
  const isComplete = isToolResultPart(toolCall);
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Badge
          variant="outline"
          className="mx-0.5 inline-flex h-5 min-w-[1.25rem] cursor-pointer items-center justify-center rounded-full border-[#4d4d4f] bg-[#2f2f2f] px-1.5 text-xs font-medium text-[#ececf1] transition-colors hover:border-[#10a37f] hover:bg-[#10a37f]/10 hover:text-[#10a37f]"
        >
          {index}
        </Badge>
      </TooltipTrigger>
      <TooltipContent
        side="top"
        className="max-w-xs border-[#4d4d4f] bg-[#2f2f2f] text-[#ececf1]"
      >
        <div className="space-y-1">
          <p className="text-xs text-[#9ca3af]">
            {isComplete ? "Source" : "Analyzing"}
          </p>
          <p className="text-sm font-medium">{summary}</p>
        </div>
      </TooltipContent>
    </Tooltip>
  );
}

interface ChatPanelProps {
  ticker: string;
  stockContext: string;
}

export function ChatPanel({ ticker, stockContext }: ChatPanelProps) {
  const { messages, input, handleInputChange, handleSubmit, isLoading, stop, error } =
    useChat({
      api: `${import.meta.env.VITE_API_URL ?? ""}/api/chat`,
      context: stockContext,
    });

  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  return (
    <TooltipProvider delayDuration={100}>
      <div className="flex flex-1 min-h-0 flex-col overflow-hidden rounded-xl border border-[#4d4d4f] bg-[#2f2f2f]">
        {/* Card header */}
        <div className="shrink-0 border-b border-[#4d4d4f] px-4 py-3">
          <h2 className="text-sm font-semibold text-white">Ask about {ticker}</h2>
        </div>

        {/* Scrollable message area */}
        <div className="flex-1 min-h-0 overflow-y-auto">
          {messages.length === 0 ? (
            <div className="flex h-full items-center justify-center p-6 text-center">
              <p className="text-sm text-[#9ca3af]">
                Ask anything about {ticker} — predictions, news, price history...
              </p>
            </div>
          ) : (
            <div className="flex flex-col">
              {messages.map((message) => {
                const chartPairs =
                  message.role === "assistant" ? extractChartPairs(message) : [];
                const toolCalls =
                  message.role === "assistant" ? extractToolCalls(message) : [];
                const isLastAssistantMessage =
                  message.role === "assistant" &&
                  message.id === messages[messages.length - 1]?.id &&
                  isLoading;

                if (message.role === "user") {
                  return (
                    <div key={message.id} className="px-3 py-2">
                      <div className="flex justify-end">
                        <div className="max-w-[80%] rounded-[16px] rounded-tr-sm bg-[#10a37f] px-4 py-2.5 text-white shadow-sm">
                          <p className="whitespace-pre-wrap text-sm leading-relaxed">
                            {message.content}
                          </p>
                        </div>
                      </div>
                    </div>
                  );
                }

                return (
                  <div
                    key={message.id}
                    className="border-b border-[#4d4d4f]/30 px-3 py-3 last:border-b-0"
                  >
                    <div className="flex gap-3">
                      <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-[#10a37f] text-xs font-medium text-white">
                        AI
                      </div>
                      <div className="min-w-0 flex-1 space-y-3">
                        <div className="prose prose-sm prose-invert max-w-none [&>*:first-child]:mt-0 [&>*:last-child]:mb-0">
                          <Streamdown
                            plugins={{ code }}
                            isAnimating={isLastAssistantMessage}
                            caret="block"
                            className="text-sm leading-relaxed text-[#ececf1] [&>*:first-child]:mt-0 [&>*:last-child]:mb-0"
                          >
                            {message.content || ""}
                          </Streamdown>
                        </div>

                        {toolCalls.length > 0 && (
                          <div className="flex items-center gap-1 text-xs">
                            <span className="text-[#6b7280]">Sources:</span>
                            {toolCalls.map((toolCall, idx) => (
                              <CitationBadge
                                key={`${toolCall.type}-${idx}`}
                                index={idx + 1}
                                toolCall={toolCall}
                              />
                            ))}
                          </div>
                        )}

                        {chartPairs.length > 0 && (
                          <div className="space-y-4 pt-1">
                            {chartPairs.map((pair, idx) => (
                              <div
                                key={idx}
                                className="rounded-xl border border-[#4d4d4f] bg-[#212121] p-3"
                              >
                                <ChartRenderer
                                  chart={pair.stock || pair.sentiment!}
                                  sentimentChart={pair.sentiment}
                                />
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
              <div ref={scrollRef} className="h-2" />
            </div>
          )}
        </div>

        {/* Error */}
        {error && (
          <div className="shrink-0 border-t border-[#4d4d4f] bg-red-900/10 px-3 py-2">
            <p className="text-xs text-red-400">
              {error.message || "Failed to send message"}
            </p>
          </div>
        )}

        {/* Input area */}
        <div className="shrink-0 border-t border-[#4d4d4f] p-3">
          <form
            onSubmit={handleSubmit}
            className="flex items-end gap-2"
          >
            <div className="relative flex-1">
              <Input
                placeholder={`Ask about ${ticker}...`}
                value={input || ""}
                onChange={handleInputChange}
                disabled={isLoading}
                className="min-h-[44px] w-full rounded-2xl border-[#4d4d4f] bg-[#212121] py-2.5 pl-4 pr-12 text-sm text-white placeholder:text-[#6b7280] focus:border-[#4d4d4f] focus:ring-0 focus-visible:ring-0 focus-visible:ring-offset-0"
              />
              <div className="absolute bottom-1.5 right-1.5">
                {isLoading ? (
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    onClick={stop}
                    className="h-8 w-8 rounded-full bg-[#404040] text-white hover:bg-[#4d4d4f]"
                  >
                    <Square className="h-3.5 w-3.5 fill-current" />
                  </Button>
                ) : (
                  <Button
                    type="submit"
                    size="icon"
                    disabled={!input?.trim()}
                    className="h-8 w-8 rounded-full bg-[#10a37f] text-white hover:bg-[#0d8c6d] disabled:opacity-40 disabled:hover:bg-[#10a37f]"
                  >
                    <ArrowUp className="h-3.5 w-3.5" />
                  </Button>
                )}
              </div>
            </div>
          </form>
        </div>
      </div>
    </TooltipProvider>
  );
}
