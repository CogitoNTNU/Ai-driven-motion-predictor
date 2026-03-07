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
from pydantic import BaseModel
from dotenv import load_dotenv

from agents.graph import get_graph, AgentState
from langchain_core.messages import AIMessageChunk, ToolMessage

# Load environment variables
load_dotenv()


class ChatMessage(BaseModel):
    """Chat message model."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """Chat request model."""

    messages: list[ChatMessage]


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


async def stream_agent_response(messages: list[dict]) -> AsyncGenerator[str, None]:
    """Stream agent responses in AI SDK v5 UI Message Stream format.

    Uses tool invocation parts to transfer chart data from sub-agents to frontend.
    Charts are sent as tool outputs that the frontend renders at the end of messages.
    """
    graph = get_graph()

    # Prepare the input state
    input_state: AgentState = {"messages": messages, "charts": []}

    # Track tool calls and their outputs
    tool_calls: dict = {}
    streamed_tool_outputs: set = set()

    # Stream with 'messages' mode to get LLM tokens and tool results
    async for chunk in graph.astream(input_state, stream_mode="messages"):
        message_chunk, metadata = chunk

        # Track tool call starts (input-available state)
        if isinstance(message_chunk, AIMessageChunk):
            # Check for tool calls in the message
            if hasattr(message_chunk, "tool_calls") and message_chunk.tool_calls:
                for tc in message_chunk.tool_calls:
                    tool_id = tc.get("id") or str(uuid.uuid4())
                    tool_calls[tool_id] = {
                        "name": tc.get("name", "unknown"),
                        "arguments": tc.get("args", {}),
                    }

                    # Send tool invocation start (input-available)
                    tool_data = {
                        "type": f"tool-{tc.get('name', 'unknown')}",
                        "toolCallId": tool_id,
                        "toolName": tc.get("name", "unknown"),
                        "state": "input-available",
                        "input": tc.get("args", {}),
                    }
                    yield f"{json.dumps(tool_data)}\n"

        # ToolMessage: extract chart artifact and send as tool output
        if isinstance(message_chunk, ToolMessage):
            tool_id = getattr(message_chunk, "tool_call_id", None) or str(uuid.uuid4())

            # Get artifact if available (from @tool(response_format="content_and_artifact"))
            artifact = getattr(message_chunk, "artifact", None)

            if artifact and isinstance(artifact, dict) and artifact.get("chart_id"):
                if tool_id not in streamed_tool_outputs:
                    streamed_tool_outputs.add(tool_id)

                    # Send tool invocation with output (output-available)
                    tool_data = {
                        "type": f"tool-{artifact.get('tool_name', 'get_stock_growth')}",
                        "toolCallId": tool_id,
                        "toolName": artifact.get("tool_name", "get_stock_growth"),
                        "state": "output-available",
                        "output": artifact,
                    }
                    yield f"{json.dumps(tool_data)}\n"

        # Stream text from AI messages
        if isinstance(message_chunk, AIMessageChunk) and message_chunk.content:
            # Send as AI SDK v5 text part
            text_data = {
                "type": "text",
                "text": message_chunk.content,
            }
            yield f"{json.dumps(text_data)}\n"

    # Send finish signal
    finish_data = {
        "type": "finish",
        "finishReason": "stop",
        "usage": {
            "promptTokens": 0,
            "completionTokens": 0,
        },
    }
    yield f"{json.dumps(finish_data)}\n"


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
        messages = [
            {"role": msg.role, "content": msg.content} for msg in request.messages
        ]

        # Return streaming response
        return StreamingResponse(
            stream_agent_response(messages),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
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
            {"role": msg.role, "content": msg.content} for msg in request.messages
        ]

        # Invoke synchronously with custom state
        input_state: AgentState = {"messages": messages, "charts": []}
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
