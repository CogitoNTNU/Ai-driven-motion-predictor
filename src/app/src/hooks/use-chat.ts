import { useChat as useAIChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { useState } from "react";
import type { UIMessage } from "ai";
import type { Message, MessagePart } from "@/types/chat";

interface UseChatOptions {
  api: string;
}

function transformMessages(aiMessages: UIMessage[]): Message[] {
  return aiMessages.map((msg) => {
    // Accumulate text from both "text" and "text-delta" parts during streaming
    // AI SDK v5 streams text in text-delta parts and accumulates them into a text part
    const textContent = msg.parts
      ?.filter((p) => {
        // Accept both the typed "text" parts and streaming "text-delta" parts
        // (text-delta is sent during streaming but may not be in type definitions)
        const partType = (p as any).type;
        return partType === "text" || partType === "text-delta";
      })
      .map((p) => {
        // Handle both TextPart and TextDeltaPart types
        const part = p as any;
        return part.text || "";
      })
      .join("") || "";
    
    return {
      id: msg.id,
      role: msg.role as "user" | "assistant" | "system",
      content: textContent,
      parts: (msg.parts || []) as MessagePart[],
      createdAt: Date.now(),
    };
  });
}

export function useChat(options: UseChatOptions) {
  const [input, setInput] = useState("");

  const {
    messages: aiMessages,
    sendMessage,
    status,
    stop,
  } = useAIChat({
    transport: new DefaultChatTransport({
      api: options.api,
    }),
  });

  const messages = transformMessages(aiMessages);

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim()) return;
    
    await sendMessage({ text: input });
    setInput("");
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInput(e.target.value);
  };

  const isLoading = status === "submitted" || status === "streaming";

  return {
    messages,
    input,
    handleInputChange,
    handleSubmit,
    isLoading,
    append: async (message: Parameters<typeof sendMessage>[0]) => {
      await sendMessage(message);
    },
    stop,
  };
}
