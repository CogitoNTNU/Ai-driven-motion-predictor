import { useChat as useAIChat } from "@ai-sdk/react";
import type { UIMessage } from "ai";
import type { Message, MessagePart } from "@/types/chat";

interface UseChatOptions {
  api: string;
}

/**
 * Transform AI SDK UIMessages to our custom Message format.
 * The AI SDK v5 handles tool invocations as message parts automatically.
 */
function transformMessages(aiMessages: UIMessage[]): Message[] {
  return aiMessages.map((msg) => ({
    id: msg.id,
    role: msg.role as "user" | "assistant" | "system",
    content: typeof msg.content === "string" ? msg.content : "",
    parts: (msg.parts || []) as MessagePart[],
    createdAt: msg.createdAt?.getTime() ?? Date.now(),
  }));
}

export function useChat(options: UseChatOptions) {
  const {
    messages: aiMessages,
    input,
    handleInputChange,
    handleSubmit,
    isLoading,
    append,
    stop,
  } = useAIChat({
    api: options.api,
    // Enable experimental features for tool invocations if needed
    experimental_throttle: 50,
  });

  // Transform AI SDK messages to our custom format with proper parts
  const messages = transformMessages(aiMessages);

  return {
    messages,
    input: input ?? "",
    handleInputChange,
    handleSubmit,
    isLoading,
    append: async (message: Parameters<typeof append>[0]) => {
      await append(message);
    },
    stop,
  };
}