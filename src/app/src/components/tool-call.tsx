import {useState} from "react";
import {Wrench, CheckCircle, AlertCircle, Loader2, ChevronDown, ChevronUp, Bot, TrendingUp, Newspaper, Terminal} from "lucide-react";
import {cn} from "@/lib/utils";
import type {MessagePart} from "@/types/chat";
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
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const { _agentName: _, ...args } = inputObj;
    return { args, agentName };
}

// Get agent display info
function getAgentInfo(agentName: string | undefined) {
    const name = agentName || "assistant";
    
    if (name.includes("stock_analyst")) {
        return {
            displayName: "Stock Analyst",
            shortName: "Analyst",
            icon: TrendingUp,
            gradient: "from-blue-500 to-cyan-500",
            bgColor: "bg-blue-500/10",
            textColor: "text-blue-600 dark:text-blue-400",
            borderColor: "border-blue-200 dark:border-blue-800",
        };
    }
    
    if (name.includes("sentiment_analyst")) {
        return {
            displayName: "Sentiment Analyst",
            shortName: "Sentiment",
            icon: Newspaper,
            gradient: "from-purple-500 to-pink-500",
            bgColor: "bg-purple-500/10",
            textColor: "text-purple-600 dark:text-purple-400",
            borderColor: "border-purple-200 dark:border-purple-800",
        };
    }
    
    if (name.includes("supervisor")) {
        return {
            displayName: "Supervisor",
            shortName: "Supervisor",
            icon: Bot,
            gradient: "from-indigo-500 to-violet-500",
            bgColor: "bg-indigo-500/10",
            textColor: "text-indigo-600 dark:text-indigo-400",
            borderColor: "border-indigo-200 dark:border-indigo-800",
        };
    }
    
    return {
        displayName: "Assistant",
        shortName: "Assistant",
        icon: Bot,
        gradient: "from-slate-500 to-gray-500",
        bgColor: "bg-slate-500/10",
        textColor: "text-slate-600 dark:text-slate-400",
        borderColor: "border-slate-200 dark:border-slate-800",
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
            return `Analyzing ${symbol} stock growth`;
        }
        
        case "get_current_price": {
            const priceSymbol = getStringArg("symbol") || getStringArg("ticker");
            return `Fetching ${priceSymbol} price`;
        }
        
        case "get_stock_news_sentiment": {
            const sentimentSymbol = getStringArg("symbol") || getStringArg("ticker");
            return `Analyzing ${sentimentSymbol} sentiment`;
        }
        
        default:
            return toolName.replace(/_/g, " ");
    }
}

// Get short result summary from output
function getResultSummary(toolName: string, output: unknown): { text: string; isError: boolean } {
    if (!output) return { text: "Completed", isError: false };
    
    // Check if output contains error
    const outputStr = typeof output === "string" ? output : JSON.stringify(output);
    if (outputStr.toLowerCase().includes("error") || outputStr.toLowerCase().includes("fail")) {
        return { 
            text: outputStr.length > 80 ? outputStr.substring(0, 80) + "..." : outputStr, 
            isError: true 
        };
    }
    
    try {
        const data = JSON.parse(outputStr);
        
        switch (toolName) {
            case "get_stock_growth": {
                if (data.percentage_growth !== undefined) {
                    const growth: number = data.percentage_growth;
                    const symbol: string = data.symbol || "Stock";
                    const sign = growth >= 0 ? "+" : "";
                    return { 
                        text: `${symbol} ${sign}${growth.toFixed(1)}%`, 
                        isError: false 
                    };
                }
                break;
            }
            
            case "get_current_price": {
                if (data.price !== undefined) {
                    const symbol: string = data.symbol || "Stock";
                    return { 
                        text: `${symbol} $${data.price.toFixed(2)}`, 
                        isError: false 
                    };
                }
                break;
            }
            
            case "get_stock_news_sentiment": {
                if (data.average_sentiment !== undefined) {
                    const symbol: string = data.symbol || "Stock";
                    const sentiment: number = data.average_sentiment;
                    const sign = sentiment > 0 ? "+" : "";
                    return { 
                        text: `${symbol} ${sign}${sentiment.toFixed(2)} sentiment`, 
                        isError: false 
                    };
                }
                break;
            }
        }
    } catch {
        // If parsing fails, return string representation
    }
    
    return { 
        text: outputStr.length > 60 ? outputStr.substring(0, 60) + "..." : outputStr, 
        isError: false 
    };
}

function extractToolCalls(parts: MessagePart[]): ToolCallState[] {
    const toolCallMap = new Map<string, ToolCallState>();

    // First pass: collect all tool calls
    parts.forEach((part) => {
        if (isToolCallPart(part)) {
            const { args, agentName } = parseToolInput(part.data.input);
            
            toolCallMap.set(part.data.toolCallId, {
                toolCallId: part.data.toolCallId,
                toolName: part.data.toolName,
                agentName: agentName,
                toolArgs: args,
                status: "calling",
            });
        }
    });

    // Second pass: update with results
    parts.forEach((part) => {
        if (isToolResultPart(part)) {
            const existing = toolCallMap.get(part.data.toolCallId);
            const outputStr = part.data.output;
            const hasError = outputStr.toLowerCase().includes("error") || 
                           outputStr.toLowerCase().includes("fail");
            
            if (existing) {
                existing.status = hasError ? "error" : "complete";
                existing.output = outputStr;
            } else {
                toolCallMap.set(part.data.toolCallId, {
                    toolCallId: part.data.toolCallId,
                    toolName: part.data.toolName,
                    agentName: "assistant",
                    toolArgs: {},
                    status: hasError ? "error" : "complete",
                    output: outputStr,
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
        <div className="mt-4 space-y-2">
            <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground/70 uppercase tracking-wider">
                <Wrench className="h-3 w-3"/>
                <span>Tools ({toolCalls.length})</span>
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
    const resultSummary = isComplete || isError ? getResultSummary(toolCall.toolName, toolCall.output) : null;
    const AgentIcon = agentInfo.icon;

    return (
        <div
            className={cn(
                "group rounded-xl border bg-card overflow-hidden transition-all duration-200",
                "hover:shadow-sm",
                isInProgress && agentInfo.borderColor,
                isComplete && "border-green-200 dark:border-green-800",
                isError && "border-red-200 dark:border-red-800"
            )}
        >
            {/* Header */}
            <button
                onClick={() => (isComplete || isError) && setIsExpanded(!isExpanded)}
                disabled={isInProgress}
                className={cn(
                    "w-full flex items-center gap-3 p-3 text-left",
                    (isComplete || isError) && "cursor-pointer hover:bg-muted/50"
                )}
            >
                {/* Status Icon */}
                <div
                    className={cn(
                        "flex h-8 w-8 shrink-0 items-center justify-center rounded-lg transition-colors",
                        isInProgress && agentInfo.bgColor,
                        isComplete && "bg-green-500/10",
                        isError && "bg-red-500/10"
                    )}
                >
                    {isInProgress && <Loader2 className={cn("h-4 w-4 animate-spin", agentInfo.textColor)}/>}
                    {isComplete && <CheckCircle className="h-4 w-4 text-green-600 dark:text-green-400"/>}
                    {isError && <AlertCircle className="h-4 w-4 text-red-600 dark:text-red-400"/>}
                </div>

                {/* Main Info */}
                <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                        <span className="font-medium text-sm text-foreground truncate">
                            {actionDescription}
                        </span>
                    </div>
                    
                    {resultSummary && (
                        <div className={cn(
                            "text-xs mt-0.5 font-medium truncate",
                            resultSummary.isError ? "text-red-600 dark:text-red-400" : "text-green-600 dark:text-green-400"
                        )}>
                            {resultSummary.text}
                        </div>
                    )}
                </div>

                {/* Expand Icon */}
                {(isComplete || isError) && (
                    <div className="shrink-0 text-muted-foreground/60 group-hover:text-muted-foreground transition-colors">
                        {isExpanded ? <ChevronUp className="h-4 w-4"/> : <ChevronDown className="h-4 w-4"/>}
                    </div>
                )}
            </button>

            {/* Expanded Details */}
            {isExpanded && (isComplete || isError) && (
                <div className="border-t bg-muted/30">
                    <div className="p-4 space-y-4">
                        {/* Agent Badge */}
                        <div className="flex items-center gap-2">
                            <div className={cn(
                                "flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium",
                                agentInfo.bgColor,
                                agentInfo.textColor
                            )}>
                                <AgentIcon className="h-3 w-3" />
                                <span>{agentInfo.displayName}</span>
                            </div>
                            <div className="text-xs text-muted-foreground font-mono">
                                {toolCall.agentName}
                            </div>
                        </div>

                        {/* Tool Name */}
                        <div className="space-y-1.5">
                            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                                <Terminal className="h-3 w-3" />
                                <span>Tool</span>
                            </div>
                            <code className="block text-xs font-mono bg-background border rounded-md px-3 py-2">
                                {toolCall.toolName}
                            </code>
                        </div>

                        {/* Parameters */}
                        {Object.keys(toolCall.toolArgs).length > 0 && (
                            <div className="space-y-1.5">
                                <div className="text-xs text-muted-foreground">Parameters</div>
                                <pre className="text-xs bg-background border rounded-md p-3 overflow-x-auto">
                                    {JSON.stringify(toolCall.toolArgs, null, 2)}
                                </pre>
                            </div>
                        )}

                        {/* Output */}
                        <div className="space-y-1.5">
                            <div className="text-xs text-muted-foreground">Output</div>
                            <pre className={cn(
                                "text-xs rounded-md p-3 overflow-x-auto max-h-60",
                                isError 
                                    ? "bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800 text-red-900 dark:text-red-200" 
                                    : "bg-green-50 dark:bg-green-950/20 border border-green-200 dark:border-green-800 text-green-900 dark:text-green-200"
                            )}>
                                {toolCall.output || "No output"}
                            </pre>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
