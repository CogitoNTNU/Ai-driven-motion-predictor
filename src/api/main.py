"""FastAPI application with streaming agent responses."""

import os
import json
from typing import AsyncGenerator
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from agents.graph import get_graph, AgentState

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
    """Stream agent responses token by token with chart data."""
    graph = get_graph()

    # Prepare the input state with empty charts array
    input_state: AgentState = {"messages": messages, "charts": []}

    # Track charts already sent to avoid duplicates
    streamed_charts: set = set()

    # Stream with 'messages' mode to get LLM tokens and tool results
    async for chunk in graph.astream(input_state, stream_mode="messages"):
        message_chunk, metadata = chunk

        # ToolMessage: extract chart artifact placed there by the @tool decorator
        if hasattr(message_chunk, "artifact") and message_chunk.artifact:
            artifact = message_chunk.artifact
            if isinstance(artifact, dict) and artifact.get("chart_id"):
                chart_id = artifact["chart_id"]
                if chart_id not in streamed_charts:
                    streamed_charts.add(chart_id)
                    yield f"data: {json.dumps({'type': 'chart', 'chart': artifact})}\n\n"

        # Stream text tokens from AI messages only (skip tool/human messages)
        if (
            hasattr(message_chunk, "content")
            and message_chunk.content
            and hasattr(message_chunk, "type")
            and message_chunk.type == "AIMessageChunk"
        ):
            data = {
                "type": "token",
                "content": message_chunk.content,
                "role": "assistant",
            }
            yield f"data: {json.dumps(data)}\n\n"

    # Send completion signal
    done_data = {
        "type": "done",
        "timestamp": datetime.now().isoformat(),
    }
    yield f"data: {json.dumps(done_data)}\n\n"


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
    Chat endpoint that streams agent responses.

    Accepts a list of messages and returns a streaming response with
    tokens from the agent conversation.

    Stream Events:
    - type: "token" - Text content from the LLM
    - type: "chart" - Chart data for visualization
    - type: "done" - Stream completion signal
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
