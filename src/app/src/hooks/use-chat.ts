import { useState, useCallback, useRef } from "react";
import type { Message, ChartData, SSEEvent } from "@/types/chat";

interface UseChatOptions {
  api?: string;
}

export function useChat(options: UseChatOptions = {}) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const chartsBufferRef = useRef<Map<string, ChartData>>(new Map());

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      setInput(e.target.value);
    },
    []
  );

  const append = useCallback(
    async (message: { role: "user" | "assistant"; content: string }) => {
      const newMessage: Message = {
        id: Date.now().toString(),
        ...message,
      };

      setMessages((prev) => [...prev, newMessage]);

      if (message.role === "user") {
        setIsLoading(true);
        const assistantMessageId = (Date.now() + 1).toString();

        // Clear charts buffer for new request
        chartsBufferRef.current.clear();

        setMessages((prev) => [
          ...prev,
          { id: assistantMessageId, role: "assistant", content: "", charts: [] },
        ]);

        try {
          const abortController = new AbortController();
          abortControllerRef.current = abortController;

          const response = await fetch(options.api || "/api/chat", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              messages: [...messages, newMessage].map((m) => ({
                role: m.role,
                content: m.content,
              })),
            }),
            signal: abortController.signal,
          });

          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }

          const reader = response.body?.getReader();
          if (!reader) {
            throw new Error("No response body");
          }

          const decoder = new TextDecoder();
          let buffer = "";

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            for (const line of lines) {
              if (line.startsWith("data: ")) {
                const data = line.slice(6);
                if (data === "[DONE]") continue;
                try {
                  const parsed: SSEEvent = JSON.parse(data);

                  if (parsed.type === "token" && parsed.content) {
                    setMessages((prev) =>
                      prev.map((m) =>
                        m.id === assistantMessageId
                          ? { ...m, content: m.content + parsed.content }
                          : m
                      )
                    );
                  } else if (parsed.type === "chart" && parsed.chart) {
                    // Store chart in buffer
                    chartsBufferRef.current.set(
                      parsed.chart.chart_id,
                      parsed.chart
                    );
                    // Update message with current charts
                    setMessages((prev) =>
                      prev.map((m) =>
                        m.id === assistantMessageId
                          ? {
                              ...m,
                              charts: Array.from(chartsBufferRef.current.values()),
                            }
                          : m
                      )
                    );
                  }
                } catch {
                  // Skip invalid JSON
                }
              }
            }
          }
        } catch (error) {
          if (error instanceof Error && error.name === "AbortError") {
            console.log("Request aborted");
          } else {
            console.error("Error streaming chat:", error);
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMessageId
                  ? {
                      ...m,
                      content:
                        "Sorry, there was an error processing your request.",
                    }
                  : m
              )
            );
          }
        } finally {
          setIsLoading(false);
          abortControllerRef.current = null;
          // chartsBufferRef is already cleared at the start of each new request
          // (line above the assistant message creation), so clearing here again
          // is redundant and risks a race condition in concurrent React renders.
        }
      }
    },
    [messages, options.api]
  );

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      if (!input.trim() || isLoading) return;

      const userMessage = input.trim();
      setInput("");
      await append({ role: "user", content: userMessage });
    },
    [input, isLoading, append]
  );

  const stop = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setIsLoading(false);
    }
  }, []);

  return {
    messages,
    input,
    handleInputChange,
    handleSubmit,
    isLoading,
    append,
    stop,
  };
}
