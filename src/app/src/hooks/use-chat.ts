import {useChat as useAIChat} from "@ai-sdk/react";
import {DefaultChatTransport} from "ai";
import {useState} from "react";
import type {UIMessage} from "ai";
import type {Message, MessagePart, TextPart} from "@/types/chat";

interface UseChatOptions {
    api: string;
}

function transformMessages(aiMessages: UIMessage[]): Message[] {
    return aiMessages.map((msg) => {
        // Extract text content from text parts
        const textContent = msg.parts
            ?.filter((p) => {
                const partType = (p as MessagePart).type;
                return partType === "text";
            })
            .map((p) => {
                const part = p as TextPart;
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
        messages: aiMessages, sendMessage, status, stop,
    } = useAIChat({
        transport: new DefaultChatTransport({
            api: options.api,
        }),
    });

    const messages = transformMessages(aiMessages);

    const handleSubmit = async (e?: React.FormEvent) => {
        e?.preventDefault();
        if (!input.trim()) return;

        const text = input;
        setInput("");
        await sendMessage({text});
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
