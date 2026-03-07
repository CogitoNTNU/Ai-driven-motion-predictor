"""FastAPI application with streaming agent responses using AI SDK v5 format."""

import os
import json
import uuid
from typing import AsyncGenerator
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import Union

from agents.graph import get_graph, AgentState
from langchain_core.messages import (
    AIMessage,
    ToolMessage,
    HumanMessage,
    SystemMessage,
    AnyMessage,
)

# Load environment variables
load_dotenv()


class TextPart(BaseModel):
    """Text part in AI SDK v5 UIMessage format."""

    type: str = Field(default="text", description="Part type - should be 'text'")
    text: str = Field(description="Text content")


class ChatMessage(BaseModel):
    """Chat message model supporting AI SDK v5 UIMessage format.

    Accepts messages with either:
    - 'content' field (legacy format)
    - 'parts' field (AI SDK v5 UIMessage format)
    """

    role: str = Field(description="Message role: 'user', 'assistant', or 'system'")
    content: Union[str, None] = Field(
        default=None, description="Text content (legacy format)"
    )
    parts: Union[list[dict], None] = Field(
        default=None, description="Message parts in AI SDK v5 format"
    )

    def get_text_content(self) -> str:
        """Extract text content from either format."""
        # If content field is provided, use it (legacy format)
        if self.content is not None:
            return self.content

        # If parts field is provided, extract text from first text part (AI SDK v5 format)
        if self.parts:
            for part in self.parts:
                if isinstance(part, dict) and part.get("type") == "text":
                    return part.get("text", "")
                elif hasattr(part, "type") and str(getattr(part, "type")) == "text":
                    return str(getattr(part, "text", ""))

        return ""


class ChatRequest(BaseModel):
    """Chat request model supporting AI SDK v5 UIMessage format."""

    messages: list[ChatMessage] = Field(
        description="List of messages in AI SDK v5 UIMessage format or legacy format"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup
    print("🚀 Starting up...")
    # Pre-initialize the graph
    get_graph()
    print("✅ Agent graph initialized")
    yield
    # Shutdown
    print("👋 Shutting down...")


app = FastAPI(
    title="Stock Analysis Agent API",
    description="FastAPI server with LangGraph agents for stock analysis",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def convert_dict_to_message(msg_dict: dict) -> AnyMessage:
    """Convert a dict message to proper LangChain message object."""
    role = msg_dict.get("role", "user")
    content = msg_dict.get("content", "")

    if role == "user":
        return HumanMessage(content=content)
    elif role == "assistant":
        return AIMessage(content=content)
    elif role == "system":
        return SystemMessage(content=content)
    else:
        # Default to HumanMessage for unknown roles
        return HumanMessage(content=content)


async def stream_agent_response(messages: list[dict]) -> AsyncGenerator[str, None]:
    """Stream agent responses using Vercel AI SDK Data Stream Protocol.

    Uses the Server-Sent Events (SSE) format with proper Data Stream Protocol headers.
    Text content is streamed using text-start/text-delta/text-end pattern.
    Charts are sent as data-chart parts after text completes.
    Tool calls are streamed as tool-call and tool-result parts.

    Stream format: SSE with "data: " prefix and JSON payload
    - text-start: Begins a text block with unique ID
    - text-delta: Incremental text content
    - text-end: Completes the text block
    - tool-call: Tool invocation with name, args, and agent name
    - tool-result: Tool execution result
    - data-chart: Chart data as custom data part
    - finish: Message completion signal
    """
    graph = get_graph()

    # Prepare the input state - convert dict messages to LangChain message objects
    langchain_messages = [convert_dict_to_message(msg) for msg in messages]
    input_state: AgentState = {"messages": langchain_messages, "charts": []}

    # Track tool calls and their outputs
    tool_calls: dict = {}
    streamed_tool_outputs: set = set()
    text_chunks_sent: set = set()

    # Generate unique IDs for stream parts
    text_id = str(uuid.uuid4())
    has_text = False

    # Stream with 'messages' mode to get LLM tokens and tool results
    async for chunk in graph.astream(input_state, stream_mode="messages"):
        message_chunk, metadata = chunk

        # PRIORITY 1: Stream text from AI messages
        if isinstance(message_chunk, AIMessage):
            content = message_chunk.content
            if content and isinstance(content, str) and content.strip():
                chunk_id = id(message_chunk)
                if chunk_id not in text_chunks_sent:
                    text_chunks_sent.add(chunk_id)

                    # Start text block if not already started
                    if not has_text:
                        has_text = True
                        start_event = {"type": "text-start", "id": text_id}
                        yield f"data: {json.dumps(start_event)}\n\n"

                    # Send text content in delta format
                    delta_event = {
                        "type": "text-delta",
                        "id": text_id,
                        "delta": content,
                    }
                    yield f"data: {json.dumps(delta_event)}\n\n"

            # PRIORITY 2: Stream tool calls as they are initiated
            if hasattr(message_chunk, "tool_calls") and message_chunk.tool_calls:
                for tc in message_chunk.tool_calls:
                    tool_id = tc.get("id") or str(uuid.uuid4())
                    tool_name = tc.get("name", "unknown")
                    tool_args = tc.get("args", {})

                    # Store tool call for result matching
                    tool_calls[tool_id] = {
                        "name": tool_name,
                        "arguments": tool_args,
                    }

                    # Extract only parameter names (not full values) to reduce verbosity
                    param_names = list(tool_args.keys()) if tool_args else []

                    # Try to determine agent name from metadata
                    # Look for 'lanes' or 'agent' in metadata
                    agent_name = "assistant"
                    if isinstance(metadata, dict):
                        if "lanes" in metadata and isinstance(metadata["lanes"], list):
                            lanes = metadata["lanes"]
                            agent_name = lanes[-1] if lanes else "assistant"
                        elif "agent" in metadata:
                            agent_name = str(metadata["agent"])
                        elif metadata.get("langgraph_node"):
                            agent_name = str(metadata["langgraph_node"])

                    # Stream tool-call event with data- prefix for Data Stream Protocol
                    tool_call_event = {
                        "type": "data-tool-call",
                        "data": {
                            "toolCallId": tool_id,
                            "toolName": tool_name,
                            "toolArgs": param_names,  # Just parameter names
                            "agentName": agent_name,
                        },
                    }
                    yield f"data: {json.dumps(tool_call_event)}\n\n"

        # PRIORITY 3: Stream tool outputs (charts and other results)
        if isinstance(message_chunk, ToolMessage):
            tool_id = getattr(message_chunk, "tool_call_id", None) or str(uuid.uuid4())
            tool_content = message_chunk.content

            if tool_id not in streamed_tool_outputs:
                streamed_tool_outputs.add(tool_id)

                # Get tool call info for the result
                tool_call_info = tool_calls.get(tool_id, {"name": "unknown"})
                tool_name = tool_call_info.get("name", "unknown")

                # Stream tool-result event with data- prefix for Data Stream Protocol
                tool_result_event = {
                    "type": "data-tool-result",
                    "data": {
                        "toolCallId": tool_id,
                        "toolName": tool_name,
                        "result": tool_content,
                    },
                }
                yield f"data: {json.dumps(tool_result_event)}\n\n"

                # Get artifact if available (from @tool(response_format="content_and_artifact"))
                artifact = getattr(message_chunk, "artifact", None)

                if artifact and isinstance(artifact, dict) and artifact.get("chart_id"):
                    # End text block before sending chart
                    if has_text:
                        end_event = {"type": "text-end", "id": text_id}
                        yield f"data: {json.dumps(end_event)}\n\n"
                        has_text = False

                    # Send chart data as custom data part
                    chart_event = {
                        "type": "data-chart",
                        "data": artifact,
                    }
                    yield f"data: {json.dumps(chart_event)}\n\n"

    # End text block if still open
    if has_text:
        end_event = {"type": "text-end", "id": text_id}
        yield f"data: {json.dumps(end_event)}\n\n"

    # Send finish signal
    finish_event = {
        "type": "finish",
        "finishReason": "stop",
    }
    yield f"data: {json.dumps(finish_event)}\n\n"

    # Send stream termination marker
    yield "data: [DONE]\n\n"


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Stock Analysis Agent API",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Chat endpoint that streams agent responses in AI SDK UIMessageStream format.

    Accepts a list of messages and returns a streaming response with
    message parts compatible with @ai-sdk/react's useChat hook.

    Stream Format (AI SDK UIMessageStream):
    - type: "text-delta" - Incremental text content from the LLM
    - type: "data-chart" - Chart data parts for visualization
    - type: "finish" - Message completion with usage statistics

    This format is compatible with Vercel AI SDK's useChat hook in the frontend.
    """
    try:
        # Convert messages to the format expected by LangGraph
        # Supports both AI SDK v5 UIMessage format (with parts) and legacy format (with content)
        messages = [
            {"role": msg.role, "content": msg.get_text_content()}
            for msg in request.messages
        ]

        # Return streaming response with Data Stream Protocol header
        return StreamingResponse(
            stream_agent_response(messages),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
                "x-vercel-ai-ui-message-stream": "v1",  # Data Stream Protocol header
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/sync")
async def chat_sync(request: ChatRequest):
    """
    Synchronous chat endpoint (non-streaming) for testing.
    """
    try:
        graph = get_graph()
        messages = [
            {"role": msg.role, "content": msg.get_text_content()}
            for msg in request.messages
        ]

        # Convert to LangChain message objects
        langchain_messages = [convert_dict_to_message(msg) for msg in messages]

        # Invoke synchronously with custom state
        input_state: AgentState = {"messages": langchain_messages, "charts": []}
        result = await graph.ainvoke(input_state)

        # Extract the final message
        final_messages = result.get("messages", [])
        charts = result.get("charts", [])

        if final_messages:
            last_message = final_messages[-1]
            return {
                "response": last_message.content,
                "role": last_message.type
                if hasattr(last_message, "type")
                else "assistant",
                "charts": charts,
            }

        return {"response": "No response generated", "role": "assistant", "charts": []}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))

    uvicorn.run(app, host=host, port=port)
