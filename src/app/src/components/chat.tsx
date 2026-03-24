import {useRef, useEffect} from "react";
import {Button} from "@/components/ui/button";
import {Input} from "@/components/ui/input";
import {ScrollArea} from "@/components/ui/scroll-area";
import {Avatar, AvatarFallback} from "@/components/ui/avatar";
import {Badge} from "@/components/ui/badge";
import {Card} from "@/components/ui/card";
import {Send, Bot, User, Square, BarChart3, Shield} from "lucide-react";
import {useChat} from "@/hooks/use-chat";
import {ChartRenderer} from "@/components/charts/ChartRenderer";
import {ToolCallRenderer} from "@/components/tool-call";
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
    
    // Log errors for debugging
    if (error) {
        console.error("Chat error:", error);
    }
    const scrollRef = useRef<HTMLDivElement>(null);

    // Auto-scroll to bottom when new messages arrive
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollIntoView({behavior: "smooth"});
        }
    }, [messages]);

    return (<div className="flex h-screen flex-col bg-background">
        {/* Header */}
        <div
            className="sticky top-0 z-10 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
            <div className="container max-w-4xl mx-auto flex h-16 items-center gap-2 px-4">
                <Bot className="h-6 w-6 text-primary"/>
                <div>
                    <h1 className="text-lg font-semibold">Stock Analysis Assistant</h1>
                    <p className="text-sm text-muted-foreground">
                        Ask about stock growth and performance
                    </p>
                </div>
            </div>
        </div>

        {/* Chat Messages */}
        <ScrollArea className="flex-1">
            <div className="container max-w-4xl mx-auto flex flex-col gap-6 p-4">
                {messages.length === 0 && (<div
                    className="flex flex-col items-center justify-center py-20 text-center text-muted-foreground">
                    <Bot className="mb-4 h-16 w-16 opacity-20"/>
                    <h2 className="text-xl font-semibold text-foreground mb-2">Welcome!</h2>
                    <p className="max-w-md text-base">
                        Ask me about stock growth, like "What's Apple's growth from January
                        1 to February 1, 2026?"
                    </p>
                </div>)}

                {messages.map((message, index) => {
                    // Extract charts from data-chart parts for assistant messages
                    const charts = message.role === "assistant" ? extractChartsFromMessage(message) : [];

                    // Check if message has tool call or tool result parts
                    const hasToolCalls = message.role === "assistant" && message.parts ? message.parts.some((p) => p.type === "data-tool-call" || p.type === "data-tool-result") : false;

                    // Check if this is the last assistant message and loading
                    const isLastAssistantMessage = message.role === "assistant" && index === messages.length - 1 && isLoading;

                    // Determine agent type for avatar
                    const getAgentIcon = () => {
                        if (message.role === "user") return <User className="h-4 w-4"/>;
                        if (hasToolCalls) return <Shield className="h-4 w-4"/>;
                        return <Bot className="h-4 w-4"/>;
                    };

                    // Determine agent background color
                    const getAgentBgColor = () => {
                        if (message.role === "user") return "bg-primary text-primary-foreground";
                        if (hasToolCalls) return "bg-purple-500 text-white";
                        return "bg-blue-500 text-white";
                    };

                    const getAgentBadge = () => {
                        if (message.role === "user") return null;
                        if (hasToolCalls) return <Badge variant="secondary" className="ml-2 text-xs">Agent</Badge>;
                        return <Badge variant="secondary" className="ml-2 text-xs">AI</Badge>;
                    };

                    return (<Card key={message.id} className="shadow-sm">
                        <div className="flex gap-4 p-6">
                            <Avatar className="h-10 w-10 shrink-0">
                                <AvatarFallback className={getAgentBgColor()}>
                                    {getAgentIcon()}
                                </AvatarFallback>
                            </Avatar>

                            <div className="flex-1 space-y-3">
                                <div className="flex items-center gap-2">
                      <span className="font-semibold text-base">
                        {message.role === "user" ? "You" : "Assistant"}
                      </span>
                                    {getAgentBadge()}
                                </div>

                                <div
                                    className="prose prose-base dark:prose-invert max-w-none [&>*:first-child]:mt-0 [&>*:last-child]:mb-0">
                                    {message.role === "user" ? (<p className="text-base">{message.content}</p>) : (
                                        <Streamdown
                                            plugins={{code}}
                                            isAnimating={isLastAssistantMessage}
                                            caret="block"
                                            className="[&>*:first-child]:mt-0 [&>*:last-child]:mb-0"
                                        >
                                            {message.content || (isLastAssistantMessage ? "" : "")}
                                        </Streamdown>)}
                                </div>

                                {hasToolCalls && (<div className="mt-4">
                                    <ToolCallRenderer parts={message.parts || []}/>
                                </div>)}

                                {charts.length > 0 && (<div className="mt-4 space-y-4">
                                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                                        <BarChart3 className="h-4 w-4"/>
                                        <span
                                            className="font-medium">Analysis Charts ({charts.length})</span>
                                    </div>
                                    <div className="space-y-4">
                                        {charts.map((chart) => (<ChartRenderer key={chart.chart_id} chart={chart}/>))}
                                    </div>
                                </div>)}
                            </div>
                        </div>
                    </Card>);
                })}
                <div ref={scrollRef}/>
            </div>
        </ScrollArea>

        {/* Error Display */}
        {error && (
            <div className="sticky bottom-20 mx-4 mb-2 p-4 bg-red-50 border border-red-200 rounded-lg text-red-800">
                <p className="font-medium">Error: {error.message || "Failed to send message"}</p>
            </div>
        )}

        {/* Input Area */}
        <div
            className="sticky bottom-0 border-t bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
            <form
                onSubmit={handleSubmit}
                className="container max-w-4xl mx-auto flex items-center gap-2 p-4"
            >
                <Input
                    placeholder="Ask about stock growth..."
                    value={input || ""}
                    onChange={handleInputChange}
                    disabled={isLoading}
                    className="flex-1 h-11"
                />
                {isLoading ? (<Button type="button" variant="outline" size="icon" onClick={stop} className="h-11 w-11">
                    <Square className="h-4 w-4"/>
                </Button>) : (<Button type="submit" size="icon" disabled={!input?.trim()} className="h-11 w-11">
                    <Send className="h-4 w-4"/>
                </Button>)}
            </form>
        </div>
    </div>);
}
