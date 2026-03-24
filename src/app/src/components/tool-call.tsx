import {useState} from "react";
import {Wrench, CheckCircle, AlertCircle, Loader2, ChevronDown, ChevronUp, Bot, TrendingUp, Newspaper} from "lucide-react";
import {cn} from "@/lib/utils";
import type {MessagePart, ToolCallPart, ToolResultPart} from "@/types/chat";
import {isToolCallPart, isToolResultPart} from "@/types/chat";

interface ToolCallRendererProps {
    parts: MessagePart[];
}

interface ToolCallState {
    toolCallId: string;
    toolName: string;
    agentName?: string;
    toolArgs: Record<string, unknown>;
    status: "calling" | "in-progress" | "complete" | "error";
    output?: string;
    error?: string;
}

// Parse tool input from AI SDK format
function parseToolInput(input: unknown): { args: Record<string, unknown>; agentName: string } {
    if (typeof input !== "object" || input === null) {
        return { args: {}, agentName: "assistant" };
    }
    
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const inputObj = input as any;
    const agentName = inputObj._agentName || "assistant";
    // Remove _agentName from display args
    const { _agentName, ...args } = inputObj;
    return { args, agentName };
}

// Get agent display info
function getAgentInfo(agentName: string | undefined) {
    const name = agentName || "assistant";
    
    if (name.includes("stock_analyst")) {
        return {
            displayName: "Stock Analyst",
            icon: <TrendingUp className="h-3 w-3 text-white"/>,
            color: "bg-blue-500",
            description: "Analyzes stock prices and growth"
        };
    }
    
    if (name.includes("sentiment_analyst")) {
        return {
            displayName: "Sentiment Analyst",
            icon: <Newspaper className="h-3 w-3 text-white"/>,
            color: "bg-purple-500",
            description: "Analyzes news sentiment"
        };
    }
    
    if (name.includes("supervisor")) {
        return {
            displayName: "Supervisor",
            icon: <Bot className="h-3 w-3 text-white"/>,
            color: "bg-indigo-500",
            description: "Coordinates analysis"
        };
    }
    
    return {
        displayName: "Assistant",
        icon: <Bot className="h-3 w-3 text-white"/>,
        color: "bg-gray-500",
        description: "General assistant"
    };
}

// Get tool action description
function getToolAction(toolName: string, args: Record<string, unknown>): string {
    const getStringArg = (key: string): string => {
        const val = args[key];
        return typeof val === "string" ? val : "unknown";
    };

    switch (toolName) {
        case "get_stock_growth": {
            const symbol = getStringArg("symbol") || getStringArg("ticker");
            const start = getStringArg("start_date") || "?";
            const end = getStringArg("end_date") || "?";
            return `Analyzing ${symbol} stock growth from ${start} to ${end}`;
        }
        
        case "get_current_price": {
            const priceSymbol = getStringArg("symbol") || getStringArg("ticker");
            return `Fetching current price for ${priceSymbol}`;
        }
        
        case "get_stock_news_sentiment": {
            const sentimentSymbol = getStringArg("symbol") || getStringArg("ticker");
            return `Analyzing news sentiment for ${sentimentSymbol}`;
        }
        
        default:
            return `Executing ${toolName}`;
    }
}

// Get short result summary from output
function getResultSummary(toolName: string, output: unknown): string {
    if (!output) return "No result";
    
    try {
        const outputStr = typeof output === "string" ? output : JSON.stringify(output);
        const data = JSON.parse(outputStr);
        
        switch (toolName) {
            case "get_stock_growth": {
                if (data.percentage_growth !== undefined) {
                    const growth: number = data.percentage_growth;
                    const symbol: string = data.symbol || "Stock";
                    const emoji = growth >= 0 ? "📈" : "📉";
                    return `${emoji} ${symbol}: ${growth > 0 ? "+" : ""}${growth.toFixed(2)}%`;
                }
                break;
            }
            
            case "get_current_price": {
                if (data.price !== undefined) {
                    const symbol: string = data.symbol || "Stock";
                    return `💵 ${symbol}: $${data.price.toFixed(2)}`;
                }
                break;
            }
            
            case "get_stock_news_sentiment": {
                if (data.average_sentiment !== undefined) {
                    const symbol: string = data.symbol || "Stock";
                    const sentiment: number = data.average_sentiment;
                    let emoji = "😐";
                    if (sentiment > 0.3) emoji = "😊";
                    else if (sentiment < -0.3) emoji = "😟";
                    return `${emoji} ${symbol}: ${sentiment > 0 ? "+" : ""}${sentiment.toFixed(2)} sentiment`;
                }
                break;
            }
        }
    } catch {
        // If parsing fails, return string representation
    }
    
    const strResult = typeof output === "string" ? output : JSON.stringify(output);
    return strResult.length > 60 ? strResult.substring(0, 60) + "..." : strResult;
}

function extractToolCalls(parts: MessagePart[]): ToolCallState[] {
    const toolCallMap = new Map<string, ToolCallState>();

    // First pass: collect all tool calls
    parts.forEach((part) => {
        if (isToolCallPart(part)) {
            const { args, agentName } = parseToolInput(part.input);
            
            toolCallMap.set(part.toolCallId, {
                toolCallId: part.toolCallId,
                toolName: part.toolName,
                agentName: agentName,
                toolArgs: args,
                status: "calling", // Will be updated if we find a result
            });
        }
    });

    // Second pass: update with results
    parts.forEach((part) => {
        if (isToolResultPart(part)) {
            const existing = toolCallMap.get(part.toolCallId);
            if (existing) {
                existing.status = "complete";
                existing.output = part.output;
            } else {
                // Result without a call (shouldn't happen, but handle it)
                toolCallMap.set(part.toolCallId, {
                    toolCallId: part.toolCallId,
                    toolName: part.toolName,
                    agentName: "assistant",
                    toolArgs: {},
                    status: "complete",
                    output: part.output,
                });
            }
        }
    });

    // Filter out internal transfer tools
    return Array.from(toolCallMap.values()).filter(
        (tc) => !tc.toolName.includes("transfer_back_to_supervisor")
    );
}

export function ToolCallRenderer({parts}: ToolCallRendererProps) {
    const toolCalls = extractToolCalls(parts);

    if (toolCalls.length === 0) return null;

    return (
        <div className="mt-3 space-y-2">
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <Wrench className="h-3 w-3"/>
                <span>Agent Activity ({toolCalls.length})</span>
            </div>

            <div className="space-y-2">
                {toolCalls.map((toolCall) => (
                    <ToolCallCard key={toolCall.toolCallId} toolCall={toolCall}/>
                ))}
            </div>
        </div>
    );
}

function ToolCallCard({toolCall}: { toolCall: ToolCallState }) {
    const [isExpanded, setIsExpanded] = useState(false);
    const isComplete = toolCall.status === "complete";
    const isInProgress = toolCall.status === "calling" || toolCall.status === "in-progress";
    const isError = toolCall.status === "error";

    const agentInfo = getAgentInfo(toolCall.agentName);
    const actionDescription = getToolAction(toolCall.toolName, toolCall.toolArgs);
    const resultSummary = isComplete ? getResultSummary(toolCall.toolName, toolCall.output) : null;

    return (
        <div
            className={cn(
                "rounded-lg border overflow-hidden transition-all",
                isInProgress && "border-blue-200 bg-blue-50/50 dark:bg-blue-950/20",
                isComplete && "border-green-200 bg-green-50/50 dark:bg-green-950/20",
                isError && "border-red-200 bg-red-50/50 dark:bg-red-950/20",
                isExpanded && "shadow-md"
            )}
        >
            {/* Header - Always visible */}
            <button
                onClick={() => isComplete && setIsExpanded(!isExpanded)}
                disabled={!isComplete}
                className={cn(
                    "w-full flex items-center gap-3 p-3 text-left",
                    isComplete && "cursor-pointer hover:bg-black/5 dark:hover:bg-white/5"
                )}
            >
                {/* Status Icon */}
                <div
                    className={cn(
                        "flex h-6 w-6 shrink-0 items-center justify-center rounded-full",
                        agentInfo.color,
                        isComplete && "bg-green-500"
                    )}
                >
                    {isInProgress && <Loader2 className="h-3 w-3 animate-spin text-white"/>}
                    {isComplete && <CheckCircle className="h-3 w-3 text-white"/>}
                    {isError && <AlertCircle className="h-3 w-3 text-white"/>}
                </div>

                {/* Main Info */}
                <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                        <span className="font-medium text-sm text-foreground">{agentInfo.displayName}</span>
                        <span className="text-xs text-muted-foreground">•</span>
                        <span className="text-xs text-muted-foreground">{actionDescription}</span>
                    </div>
                    
                    {resultSummary && (
                        <div className="text-xs text-green-600 dark:text-green-400 mt-0.5 font-medium">
                            {resultSummary}
                        </div>
                    )}
                </div>

                {/* Expand Icon */}
                {isComplete && (
                    <div className="shrink-0 text-muted-foreground">
                        {isExpanded ? <ChevronUp className="h-4 w-4"/> : <ChevronDown className="h-4 w-4"/>}
                    </div>
                )}
            </button>

            {/* Expanded Details */}
            {isExpanded && isComplete && (
                <div className="border-t bg-muted/30 px-3 py-3 space-y-3">
                    {/* Agent Info */}
                    <div className="space-y-1">
                        <div className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Agent</div>
                        <div className="flex items-center gap-2">
                            <div className={cn("h-4 w-4 rounded-full flex items-center justify-center", agentInfo.color)}>
                                {agentInfo.icon}
                            </div>
                            <div>
                                <div className="text-sm font-medium">{agentInfo.displayName}</div>
                                <div className="text-xs text-muted-foreground">{agentInfo.description}</div>
                                <div className="text-xs text-muted-foreground font-mono">{toolCall.agentName}</div>
                            </div>
                        </div>
                    </div>

                    {/* Tool Info */}
                    <div className="space-y-1">
                        <div className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Tool</div>
                        <div className="text-sm font-mono bg-muted rounded px-2 py-1">{toolCall.toolName}</div>
                    </div>

                    {/* Arguments */}
                    <div className="space-y-1">
                        <div className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Parameters</div>
                        <pre className="text-xs bg-muted rounded p-2 overflow-x-auto">
                            {JSON.stringify(toolCall.toolArgs, null, 2)}
                        </pre>
                    </div>

                    {/* Output */}
                    <div className="space-y-1">
                        <div className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Output</div>
                        <pre className="text-xs bg-green-50 dark:bg-green-950/30 border border-green-200 dark:border-green-800 rounded p-2 overflow-x-auto">
                            {toolCall.output || "No output"}
                        </pre>
                    </div>
                </div>
            )}
        </div>
    );
}
