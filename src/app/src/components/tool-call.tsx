import { Wrench, CheckCircle, AlertCircle, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import type { MessagePart } from "@/types/chat";
import { isToolCallPart, isToolCallResultPart } from "@/types/chat";

interface ToolCallRendererProps {
  parts: MessagePart[];
}

interface ToolCallState {
  toolCallId: string;
  toolName: string;
  agentName?: string;
  toolArgs: string[];
  status: "calling" | "in-progress" | "complete" | "error";
  result?: string | object;
  error?: string;
}

function extractToolCalls(parts: MessagePart[]): ToolCallState[] {
  const toolCallMap = new Map<string, ToolCallState>();

  parts.forEach((part) => {
    if (isToolCallPart(part)) {
      const existing = toolCallMap.get(part.toolCallId);
      if (existing) {
        existing.status = "in-progress";
      } else {
        toolCallMap.set(part.toolCallId, {
          toolCallId: part.toolCallId,
          toolName: part.toolName,
          agentName: part.agentName ?? "assistant",
          toolArgs: Array.isArray(part.toolArgs) ? part.toolArgs : [],
          status: "calling",
        });
      }
    } else if (isToolCallResultPart(part)) {
      const existing = toolCallMap.get(part.toolCallId);
      if (existing) {
        existing.status = "complete";
        existing.result = part.result;
      } else {
        toolCallMap.set(part.toolCallId, {
          toolCallId: part.toolCallId,
          toolName: part.toolName,
          agentName: "unknown",
          toolArgs: [],
          status: "complete",
          result: part.result,
        });
      }
    }
  });

  return Array.from(toolCallMap.values());
}

export function ToolCallRenderer({ parts }: ToolCallRendererProps) {
  const toolCalls = extractToolCalls(parts);

  if (toolCalls.length === 0) return null;

  return (
    <div className="mt-3 space-y-2">
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <Wrench className="h-3 w-3" />
        <span>Tool calls ({toolCalls.length})</span>
      </div>

      <div className="space-y-1.5">
        {toolCalls.map((toolCall) => (
          <ToolCallCard key={toolCall.toolCallId} toolCall={toolCall} />
        ))}
      </div>
    </div>
  );
}

function ToolCallCard({ toolCall }: { toolCall: ToolCallState }) {
  const isComplete = toolCall.status === "complete";
  const isInProgress = toolCall.status === "calling" || toolCall.status === "in-progress";
  const isError = toolCall.status === "error";

  const agentName = toolCall.agentName ?? "assistant";
  const agentColor =
    agentName.toLowerCase().includes("supervisor")
      ? "bg-purple-500"
      : "bg-blue-500";

  return (
    <div
      className={cn(
        "flex items-start gap-2 rounded-md border p-2.5 text-sm",
        isInProgress && "border-blue-200 bg-blue-50/50 dark:bg-blue-950/20",
        isComplete && "border-green-200 bg-green-50/50 dark:bg-green-950/20",
        isError && "border-red-200 bg-red-50/50 dark:bg-red-950/20"
      )}
    >
      <div
        className={cn(
          "mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full",
          agentColor,
          isComplete && "bg-green-500"
        )}
      >
        {isInProgress && <Loader2 className="h-3 w-3 animate-spin text-white" />}
        {isComplete && <CheckCircle className="h-3 w-3 text-white" />}
        {isError && <AlertCircle className="h-3 w-3 text-white" />}
      </div>

      <div className="flex min-w-0 flex-1 flex-col gap-1">
        <div className="flex items-center gap-2 overflow-hidden">
          <span className="font-medium text-foreground">{toolCall.toolName}</span>
          <span className="text-xs text-muted-foreground">by {agentName}</span>
        </div>

        {toolCall.toolArgs.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {toolCall.toolArgs.map((arg, index) => (
              <code
                key={`${arg}-${index}`}
                className="rounded bg-muted px-1.5 py-0.5 text-xs font-mono text-muted-foreground"
              >
                {arg}
              </code>
            ))}
          </div>
        )}

        {isComplete && toolCall.status === "complete" && (
          <div className="mt-1 rounded bg-muted/50 px-2 py-1 text-xs text-muted-foreground">
            Completed
          </div>
        )}
      </div>
    </div>
  );
}