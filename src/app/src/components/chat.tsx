import {useRef, useEffect} from "react";
import {Button} from "@/components/ui/button";
import {Input} from "@/components/ui/input";
import {ScrollArea} from "@/components/ui/scroll-area";
import {Send, Square} from "lucide-react";
import {useChat} from "@/hooks/use-chat";
import {ChartRenderer} from "@/components/charts/ChartRenderer";
import {Streamdown} from "streamdown";
import {code} from "@streamdown/code";
import {isChartPart, type ChartData, type Message} from "@/types/chat";
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
        <div className="flex h-screen flex-col bg-[#212121]">
            {/* Header */}
            <div className="sticky top-0 z-10 bg-[#212121]/95 backdrop-blur supports-[backdrop-filter]:bg-[#212121]/60">
                <div className="flex h-14 items-center justify-center px-4">
                    <h1 className="text-sm font-medium text-white">Stock Analysis</h1>
                </div>
            </div>

            {/* Chat Messages */}
            <ScrollArea className="flex-1">
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

                    {messages.map((message, index) => {
                        const charts = message.role === "assistant" ? extractChartsFromMessage(message) : [];
                        const isLastAssistantMessage = message.role === "assistant" && index === messages.length - 1 && isLoading;

                        if (message.role === "user") {
                            return (
                                <div key={message.id} className="px-4 py-6">
                                    <div className="mx-auto max-w-3xl">
                                        <div className="flex gap-4">
                                            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-[#10a37f] text-sm font-medium text-white">
                                                You
                                            </div>
                                            <div className="min-w-0 flex-1">
                                                <p className="whitespace-pre-wrap text-[15px] leading-relaxed text-white">
                                                    {message.content}
                                                </p>
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
                    <div ref={scrollRef}/>
                </div>
            </ScrollArea>

            {/* Error Display */}
            {error && (
                <div className="mx-4 mb-2 rounded-lg border border-red-800 bg-red-900/20 p-4 text-red-200">
                    <p className="text-sm font-medium">Error: {error.message || "Failed to send message"}</p>
                </div>
            )}

            {/* Input Area */}
            <div className="border-t border-[#2f2f2f] bg-[#212121]">
                <form
                    onSubmit={handleSubmit}
                    className="mx-auto flex max-w-3xl items-end gap-2 p-4"
                >
                    <div className="relative flex-1">
                        <Input
                            placeholder="Message Stock Analysis Assistant..."
                            value={input || ""}
                            onChange={handleInputChange}
                            disabled={isLoading}
                            className="min-h-[52px] w-full rounded-3xl border-[#4d4d4f] bg-[#2f2f2f] pr-12 text-[15px] text-white placeholder:text-[#9ca3af] focus:border-[#10a37f] focus:ring-[#10a37f]"
                        />
                        <div className="absolute bottom-1 right-1">
                            {isLoading ? (
                                <Button 
                                    type="button" 
                                    variant="ghost" 
                                    size="icon" 
                                    onClick={stop}
                                    className="h-9 w-9 rounded-full text-[#9ca3af] hover:bg-[#4d4d4f] hover:text-white"
                                >
                                    <Square className="h-4 w-4 fill-current"/>
                                </Button>
                            ) : (
                                <Button 
                                    type="submit" 
                                    size="icon" 
                                    disabled={!input?.trim()}
                                    className="h-9 w-9 rounded-full bg-[#10a37f] text-white hover:bg-[#0d8c6d] disabled:opacity-40"
                                >
                                    <Send className="h-4 w-4"/>
                                </Button>
                            )}
                        </div>
                    </div>
                </form>
                <div className="pb-3 text-center">
                    <p className="text-xs text-[#6b7280]">
                        AI can make mistakes. Check important info.
                    </p>
                </div>
            </div>
        </div>
    );
}
