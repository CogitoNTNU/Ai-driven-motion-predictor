import { useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Send, Bot, User, Square, BarChart3 } from "lucide-react";
import { cn } from "@/lib/utils";
import { useChat } from "@/hooks/use-chat";
import { ChartRenderer } from "@/components/charts/ChartRenderer";
import { Streamdown } from "streamdown";
import { code } from "@streamdown/code";
import { isChartPart, type ChartData, type Message } from "@/types/chat";
import "streamdown/styles.css";

/**
 * Extract chart data from data-chart parts in a message.
 * Charts are sent from the backend via data-chart parts in the Data Stream Protocol.
 */
function extractChartsFromMessage(message: Message): ChartData[] {
  if (!message.parts || message.parts.length === 0) return [];

  return message.parts
    .filter(isChartPart)
    .map((part) => part.data);
}

export function Chat() {
  const { messages, input, handleInputChange, handleSubmit, isLoading, stop } =
    useChat({
      api: `${import.meta.env.VITE_API_URL ?? ""}/api/chat`,
    });
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  return (
    <div className="flex h-[calc(100vh-2rem)] flex-col gap-4 p-4">
      {/* Header */}
      <div className="flex items-center gap-2 border-b pb-4">
        <Bot className="h-6 w-6 text-primary" />
        <div>
          <h1 className="text-lg font-semibold">Stock Analysis Assistant</h1>
          <p className="text-sm text-muted-foreground">
            Ask about stock growth and performance
          </p>
        </div>
      </div>

      {/* Chat Messages */}
      <ScrollArea className="flex-1 pr-4">
        <div className="flex flex-col gap-4">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center py-12 text-center text-muted-foreground">
              <Bot className="mb-4 h-12 w-12 opacity-20" />
              <p className="text-lg font-medium">Welcome!</p>
              <p className="max-w-sm">
                Ask me about stock growth, like "What's Apple's growth from January
                1 to February 1, 2026?"
              </p>
            </div>
          )}

          {messages.map((message, index) => {
            // Extract charts from data-chart parts for assistant messages
            const charts = message.role === "assistant"
              ? extractChartsFromMessage(message)
              : [];
            
            // Check if this is the last assistant message and loading
            const isLastAssistantMessage = 
              message.role === "assistant" && 
              index === messages.length - 1 && 
              isLoading;

            return (
              <div key={message.id}>
                <div
                  className={cn(
                    "flex gap-3",
                    message.role === "user" ? "flex-row-reverse" : "flex-row"
                  )}
                >
                  <Avatar className="h-8 w-8">
                    <AvatarFallback
                      className={cn(
                        message.role === "user"
                          ? "bg-primary text-primary-foreground"
                          : "bg-muted"
                      )}
                    >
                      {message.role === "user" ? (
                        <User className="h-4 w-4" />
                      ) : (
                        <Bot className="h-4 w-4" />
                      )}
                    </AvatarFallback>
                  </Avatar>

                  <div
                    className={cn(
                      "max-w-[80%] rounded-lg px-4 py-2",
                      message.role === "user"
                        ? "bg-primary text-primary-foreground"
                        : "bg-muted"
                    )}
                  >
                    {message.role === "user" ? (
                      <div className="prose prose-sm dark:prose-invert max-w-none [&>*:first-child]:mt-0 [&>*:last-child]:mb-0">
                        <span>{message.content}</span>
                      </div>
                    ) : (
                      <div className="max-w-none text-sm">
                        <Streamdown
                          plugins={{ code }}
                          isAnimating={isLastAssistantMessage}
                          caret="block"
                          className="prose prose-sm dark:prose-invert [&>*:first-child]:mt-0 [&>*:last-child]:mb-0"
                        >
                          {message.content || (isLastAssistantMessage ? "" : "")}
                        </Streamdown>
                      </div>
                    )}
                  </div>
                </div>

                {/* Charts from data-chart parts - rendered at end of assistant messages */}
                {charts.length > 0 && (
                  <div className="ml-11 mt-4 space-y-4">
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <BarChart3 className="h-4 w-4" />
                      <span>Analysis Charts ({charts.length})</span>
                    </div>
                    <div className="space-y-4">
                      {charts.map((chart) => (
                        <ChartRenderer key={chart.chart_id} chart={chart} />
                      ))}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
          <div ref={scrollRef} />
        </div>
      </ScrollArea>

      {/* Input Area */}
      <form
        onSubmit={handleSubmit}
        className="flex items-center gap-2 border-t pt-4"
      >
        <Input
          placeholder="Ask about stock growth..."
          value={input || ""}
          onChange={handleInputChange}
          disabled={isLoading}
          className="flex-1"
        />
        {isLoading ? (
          <Button type="button" variant="outline" size="icon" onClick={stop}>
            <Square className="h-4 w-4" />
          </Button>
        ) : (
          <Button type="submit" size="icon" disabled={!input?.trim()}>
            <Send className="h-4 w-4" />
          </Button>
        )}
      </form>
    </div>
  );
}
