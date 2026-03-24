import {useRef, useEffect} from "react";
import {Button} from "@/components/ui/button";
import {Input} from "@/components/ui/input";
import {ScrollArea} from "@/components/ui/scroll-area";
import {Badge} from "@/components/ui/badge";
import {Tooltip, TooltipContent, TooltipProvider, TooltipTrigger} from "@/components/ui/tooltip";
import {ArrowUp, Square} from "lucide-react";
import {useChat} from "@/hooks/use-chat";
import {ChartRenderer} from "@/components/charts/ChartRenderer";
import {Streamdown} from "streamdown";
import {code} from "@streamdown/code";
import {isChartPart, isToolCallPart, isToolResultPart, type ChartData, type Message, type ToolCallPart, type ToolResultPart} from "@/types/chat";
import "streamdown/styles.css";

/**
 * Extract chart data from data-chart parts in a message.
 */
function extractChartsFromMessage(message: Message): ChartData[] {
    if (!message.parts || message.parts.length === 0) return [];
    return message.parts.filter(isChartPart).map((part) => part.data);
}

/**
 * Extract tool calls from message parts.
 */
function extractToolCalls(message: Message): Array<ToolCallPart | ToolResultPart> {
    if (!message.parts || message.parts.length === 0) return [];
    return message.parts.filter((part): part is ToolCallPart | ToolResultPart => 
        isToolCallPart(part) || isToolResultPart(part)
    );
}

/**
 * Get tool call summary for citation tooltip.
 */
function getToolCallSummary(toolCall: ToolCallPart | ToolResultPart): string {
    if (isToolCallPart(toolCall)) {
        const toolName = toolCall.data.toolName;
        const args = toolCall.data.input as Record<string, unknown>;
        
        if (toolName === "get_stock_growth") {
            const symbol = args.symbol || args.ticker || "stock";
            return `Analyzing ${symbol} stock growth`;
        }
        if (toolName === "get_current_price") {
            const symbol = args.symbol || args.ticker || "stock";
            return `Fetching ${symbol} price`;
        }
        if (toolName === "get_stock_news_sentiment") {
            const symbol = args.symbol || args.ticker || "stock";
            return `Analyzing ${symbol} sentiment`;
        }
        return toolName.replace(/_/g, " ");
    }
    
    if (isToolResultPart(toolCall)) {
        const toolName = toolCall.data.toolName;
        const output = toolCall.data.output;
        
        try {
            const data = JSON.parse(output);
            if (toolName === "get_stock_growth" && data.percentage_growth !== undefined) {
                const sign = data.percentage_growth >= 0 ? "+" : "";
                return `${data.symbol || "Stock"}: ${sign}${data.percentage_growth.toFixed(1)}%`;
            }
            if (toolName === "get_current_price" && data.price !== undefined) {
                return `${data.symbol || "Stock"}: $${data.price.toFixed(2)}`;
            }
            if (toolName === "get_stock_news_sentiment" && data.average_sentiment !== undefined) {
                const sign = data.average_sentiment > 0 ? "+" : "";
                return `${data.symbol || "Stock"}: ${sign}${data.average_sentiment.toFixed(2)} sentiment`;
            }
        } catch {
            // Fall through to default
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

interface CitationRowProps {
    toolCalls: Array<ToolCallPart | ToolResultPart>;
}

function CitationRow({ toolCalls }: CitationRowProps) {
    if (toolCalls.length === 0) return null;
    
    return (
        <div className="mt-4 flex items-center gap-1 text-sm">
            <span className="text-[#6b7280]">Sources:</span>
            {toolCalls.map((toolCall, idx) => (
                <CitationBadge 
                    key={`${toolCall.type}-${idx}`} 
                    index={idx + 1} 
                    toolCall={toolCall} 
                />
            ))}
        </div>
    );
}

export function Chat() {
    const {messages, input, handleInputChange, handleSubmit, isLoading, stop, error} = useChat({
        api: `${import.meta.env.VITE_API_URL ?? ""}/api/chat`,
    });
    
    if (error) {
        console.error("Chat error:", error);
    }
    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollIntoView({behavior: "smooth"});
        }
    }, [messages]);

    return (
        <TooltipProvider delayDuration={100}>
            <div className="flex h-screen flex-col bg-[#212121]">
                {/* Header */}
                <div className="fixed top-0 left-0 right-0 z-50 bg-[#212121]/95 backdrop-blur supports-[backdrop-filter]:bg-[#212121]/60">
                    <div className="flex h-14 items-center justify-center px-4">
                        <h1 className="text-sm font-medium text-white">Stock Analysis</h1>
                    </div>
                </div>

                {/* Chat Messages */}
                <ScrollArea className="flex-1 pt-14 pb-32">
                    <div className="mx-auto flex max-w-3xl flex-col">
                        {messages.length === 0 && (
                            <div className="flex min-h-[60vh] flex-col items-center justify-center px-4 text-center">
                                <h2 className="mb-2 text-3xl font-semibold text-white">
                                    What stock can I help you analyze?
                                </h2>
                                <p className="text-base text-[#9ca3af]">
                                    Ask about stock growth, like "What's Apple's growth from January to February 2026?"
                                </p>
                            </div>
                        )}

                        {messages.map((message) => {
                            const charts = message.role === "assistant" ? extractChartsFromMessage(message) : [];
                            const toolCalls = message.role === "assistant" ? extractToolCalls(message) : [];
                            const isLastAssistantMessage = message.role === "assistant" && message.id === messages[messages.length - 1]?.id && isLoading;

                            if (message.role === "user") {
                                return (
                                    <div key={message.id} className="px-4 py-3">
                                        <div className="mx-auto max-w-3xl">
                                            <div className="flex justify-end gap-4">
                                                <div className="max-w-[80%]">
                                                    <div className="rounded-[20px] rounded-tr-sm bg-[#10a37f] px-5 py-3 text-white shadow-sm">
                                                        <p className="whitespace-pre-wrap text-[15px] leading-relaxed">
                                                            {message.content}
                                                        </p>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                );
                            }

                            return (
                                <div key={message.id} className="border-b border-[#2f2f2f] bg-[#212121] px-4 py-6">
                                    <div className="mx-auto max-w-3xl">
                                        <div className="flex gap-4">
                                            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-[#10a37f] text-sm font-medium text-white">
                                                AI
                                            </div>
                                            <div className="min-w-0 flex-1 space-y-4">
                                                <div className="prose prose-base prose-invert max-w-none [&>*:first-child]:mt-0 [&>*:last-child]:mb-0">
                                                    <Streamdown
                                                        plugins={{code}}
                                                        isAnimating={isLastAssistantMessage}
                                                        caret="block"
                                                        className="text-[15px] leading-relaxed text-[#ececf1] [&>*:first-child]:mt-0 [&>*:last-child]:mb-0"
                                                    >
                                                        {message.content || ""}
                                                    </Streamdown>
                                                </div>

                                                {toolCalls.length > 0 && (
                                                    <CitationRow toolCalls={toolCalls} />
                                                )}

                                                {charts.length > 0 && (
                                                    <div className="space-y-4 pt-2">
                                                        {charts.map((chart) => (
                                                            <ChartRenderer key={chart.chart_id} chart={chart}/>
                                                        ))}
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            );
                        })}
                        <div ref={scrollRef} className="h-8"/>
                    </div>
                </ScrollArea>

                {/* Error Display */}
                {error && (
                    <div className="fixed bottom-24 left-1/2 z-50 w-[calc(100%-2rem)] max-w-3xl -translate-x-1/2">
                        <div className="rounded-lg border border-red-800 bg-red-900/20 p-4 text-red-200">
                            <p className="text-sm font-medium">Error: {error.message || "Failed to send message"}</p>
                        </div>
                    </div>
                )}

                {/* Fixed Input Area */}
                <div className="fixed bottom-0 left-0 right-0 z-50">
                    {/* Gradient fade */}
                    <div className="pointer-events-none h-16 bg-gradient-to-t from-[#212121] via-[#212121]/90 to-transparent" />
                    
                    {/* Input container */}
                    <div className="bg-[#212121] px-4 pb-4">
                        <form
                            onSubmit={handleSubmit}
                            className="mx-auto flex max-w-3xl items-end gap-2"
                        >
                            <div className="relative flex-1">
                                <Input
                                    placeholder="Message Stock Analysis Assistant..."
                                    value={input || ""}
                                    onChange={handleInputChange}
                                    disabled={isLoading}
                                    className="min-h-[52px] w-full rounded-[26px] border-[#4d4d4f] bg-[#2f2f2f] py-3 pl-5 pr-14 text-[15px] text-white placeholder:text-[#6b7280] focus:border-[#4d4d4f] focus:ring-0 focus-visible:ring-0 focus-visible:ring-offset-0"
                                />
                                <div className="absolute bottom-1.5 right-1.5">
                                    {isLoading ? (
                                        <Button 
                                            type="button" 
                                            variant="ghost" 
                                            size="icon" 
                                            onClick={stop}
                                            className="h-9 w-9 rounded-full bg-[#404040] text-white hover:bg-[#4d4d4f]"
                                        >
                                            <Square className="h-4 w-4 fill-current"/>
                                        </Button>
                                    ) : (
                                        <Button 
                                            type="submit" 
                                            size="icon" 
                                            disabled={!input?.trim()}
                                            className="h-9 w-9 rounded-full bg-[#10a37f] text-white hover:bg-[#0d8c6d] disabled:opacity-40 disabled:hover:bg-[#10a37f]"
                                        >
                                            <ArrowUp className="h-4 w-4" />
                                        </Button>
                                    )}
                                </div>
                            </div>
                        </form>
                        <div className="pt-2 text-center">
                            <p className="text-xs text-[#6b7280]">
                                AI can make mistakes. Check important info.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </TooltipProvider>
    );
}
