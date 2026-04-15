"""FastAPI application with streaming agent responses using AI SDK v5 format."""

# Standard library imports (must be first for ruff E402)
import json
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import AsyncGenerator, Optional, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src directory to Python path for Kaare module
# main.py is in src/api/, so go up one level to reach src/
src_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(src_path))

# Third-party imports (after path manipulation - noqa required for E402)
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from langchain_core.messages import (  # noqa: E402
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import BaseModel, Field  # noqa: E402

# Load environment variables before other imports that might need them
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

# Local imports (after path manipulation - noqa required for E402)
from agents.graph import AgentState, get_graph  # noqa: E402
import yfinance as yf  # noqa: E402
from Kaare import KaareClient  # noqa: E402
from Kaare.db.connection import get_shared_pool, close_shared_pool  # noqa: E402
from Kaare.db import migrations  # noqa: E402

# In-memory sentiment cache: ticker -> (cached_at, result_dict)
_sentiment_cache: dict[str, tuple[datetime, dict]] = {}
_SENTIMENT_TTL = timedelta(hours=1)


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
    context: Optional[str] = Field(
        default=None,
        description="Optional stock context (ticker info) prepended as system message",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    print("🚀 Starting up...")

    get_graph()
    print("✅ Agent graph initialized")

    # Shared DB pool — created once, reused by all REST endpoints
    try:
        pool = await get_shared_pool()
        await migrations.run_migrations(pool)
        app.state.db_pool = pool
        print("✅ Database pool initialized")
    except Exception as exc:
        print(
            f"⚠️  Database pool init failed (sentiment endpoints will be degraded): {exc}"
        )
        app.state.db_pool = None

    yield

    # Shutdown
    await close_shared_pool()
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


async def stream_agent_response(
    messages: list[dict], context: Optional[str] = None
) -> AsyncGenerator[str, None]:
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

    # Prepend context as system message if provided
    if context:
        messages = [{"role": "system", "content": context}] + messages

    # Prepare the input state - convert dict messages to LangChain message objects
    langchain_messages = [convert_dict_to_message(msg) for msg in messages]
    input_state: AgentState = {"messages": langchain_messages, "charts": []}

    # Track tool calls and their outputs
    tool_calls: dict = {}
    streamed_tool_outputs: set = set()
    text_chunks_sent: set = set()

    # Generate unique IDs for stream parts
    text_id = str(uuid.uuid4())
    message_id = str(uuid.uuid4())
    has_text = False

    # Send start event with message ID (AI SDK v5 format)
    start_event = {"type": "start", "messageId": message_id}
    yield f"data: {json.dumps(start_event, ensure_ascii=False)}\n\n"

    # Stream with multiple modes to get both LLM tokens and node updates
    # This is necessary for supervisor/sub-agent patterns to track when sub-agents are invoked
    # stream_mode=["messages", "updates"] gives us:
    # - "messages": LLM token chunks from both supervisor and sub-agents
    # - "updates": Node state updates showing which agent/node is running
    async for chunk in graph.astream(input_state, stream_mode=["messages", "updates"]):
        message_chunk = None
        metadata = {}

        # Handle tuple format when using multiple stream modes: (mode, data)
        if isinstance(chunk, tuple) and len(chunk) == 2:
            mode, data = chunk
            if mode == "messages":
                message_chunk, metadata = data
            elif mode == "updates":
                # Node update - contains state changes from a specific node (agent)
                # This helps us track which sub-agent is currently running
                for node_name, node_state in data.items():
                    if node_name and node_state and isinstance(node_state, dict):
                        # Track current agent from node name
                        if "messages" in node_state and node_state["messages"]:
                            # Process messages from node update
                            last_msg = node_state["messages"][-1]
                            message_chunk = last_msg
                            metadata = {"langgraph_node": node_name}
                            break  # Process first valid message only
        else:
            # Single mode streaming (backward compatibility)
            message_chunk, metadata = chunk

        # Skip if no message chunk to process
        if message_chunk is None:
            continue

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
                        yield f"data: {json.dumps(start_event, ensure_ascii=False)}\n\n"

                    # Send text content in delta format
                    delta_event = {
                        "type": "text-delta",
                        "id": text_id,
                        "delta": content,
                    }
                    yield f"data: {json.dumps(delta_event, ensure_ascii=False)}\n\n"

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

                    # Extract full tool arguments for better frontend visibility
                    # Send the full args object so frontend can show what the tool is doing
                    tool_args_data = tool_args if tool_args else {}

                    # Try to determine agent name from metadata
                    # LangGraph metadata contains langgraph_node which is the agent name
                    agent_name = "assistant"
                    if isinstance(metadata, dict):
                        # Primary: use langgraph_node which contains the actual agent/node name
                        if metadata.get("langgraph_node"):
                            agent_name = str(metadata["langgraph_node"])
                        # Fallback: check lanes array
                        elif "lanes" in metadata and isinstance(
                            metadata["lanes"], list
                        ):
                            lanes = metadata["lanes"]
                            agent_name = lanes[-1] if lanes else "assistant"
                        # Legacy fallback
                        elif "agent" in metadata:
                            agent_name = str(metadata["agent"])

                    # Stream tool call as custom data part
                    # AI SDK accepts custom data-* events that can contain arbitrary data
                    # Frontend will handle these via message.parts with type "data-tool-call"
                    # Schema: { type: "data-*", id?: string, data: unknown, transient?: boolean }
                    tool_call_event = {
                        "type": "data-tool-call",  # Custom data part type
                        "id": tool_id,  # Optional ID for persistence in message parts
                        "data": {
                            "toolCallId": tool_id,
                            "toolName": tool_name,
                            "input": tool_args_data,
                            "_agentName": agent_name,  # Include agent name for frontend display
                        },
                    }
                    yield f"data: {json.dumps(tool_call_event, ensure_ascii=False)}\n\n"

        # PRIORITY 3: Stream tool outputs (charts and other results)
        if isinstance(message_chunk, ToolMessage):
            tool_id = getattr(message_chunk, "tool_call_id", None) or str(uuid.uuid4())
            tool_content = message_chunk.content

            if tool_id not in streamed_tool_outputs:
                streamed_tool_outputs.add(tool_id)

                # Get tool call info for the result
                tool_call_info = tool_calls.get(tool_id, {"name": "unknown"})
                tool_name = tool_call_info.get("name", "unknown")

                # Stream tool result as custom data part
                # AI SDK accepts custom data-* events that can contain arbitrary data
                # Frontend will handle these via message.parts with type "data-tool-result"
                # Schema: { type: "data-*", id?: string, data: unknown, transient?: boolean }
                output_data = (
                    tool_content
                    if isinstance(tool_content, str)
                    else json.dumps(tool_content, ensure_ascii=False)
                )
                tool_result_event = {
                    "type": "data-tool-result",  # Custom data part type
                    "id": tool_id,  # Optional ID for persistence in message parts
                    "data": {
                        "toolCallId": tool_id,
                        "toolName": tool_name,
                        "output": output_data,
                    },
                }
                yield f"data: {json.dumps(tool_result_event, ensure_ascii=False)}\n\n"

                # Get artifact if available (from @tool(response_format="content_and_artifact"))
                artifact = getattr(message_chunk, "artifact", None)

                if artifact and isinstance(artifact, dict) and artifact.get("chart_id"):
                    # End text block before sending chart
                    if has_text:
                        end_event = {"type": "text-end", "id": text_id}
                        yield f"data: {json.dumps(end_event, ensure_ascii=False)}\n\n"
                        has_text = False

                    # Send chart data as custom data part
                    chart_event = {
                        "type": "data-chart",
                        "data": artifact,
                    }
                    yield f"data: {json.dumps(chart_event, ensure_ascii=False)}\n\n"

    # End text block if still open
    if has_text:
        end_event = {"type": "text-end", "id": text_id}
        yield f"data: {json.dumps(end_event, ensure_ascii=False)}\n\n"

    # Send finish signal (AI SDK v5 format)
    # Schema: { type: "finish", finishReason?: string, messageMetadata?: unknown }
    # Note: usage is NOT part of the standard schema - use messageMetadata if needed
    finish_event = {
        "type": "finish",
        "finishReason": "stop",
    }
    yield f"data: {json.dumps(finish_event, ensure_ascii=False)}\n\n"

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
            stream_agent_response(messages, context=request.context),
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


@app.get("/api/search")
async def search_tickers(q: str = ""):
    """Search for ticker symbols with autocomplete."""
    if not q or len(q.strip()) < 1:
        return []
    try:
        search = yf.Search(q.strip(), news_count=0, max_results=8)
        quotes = search.quotes or []
        results = []
        for quote in quotes:
            symbol = quote.get("symbol", "")
            name = quote.get("longname") or quote.get("shortname") or symbol
            q_type = quote.get("quoteType", "")
            if symbol and q_type in ("EQUITY", "ETF"):
                results.append({"symbol": symbol, "name": name, "type": q_type})
        return results[:8]
    except Exception:
        return []


@app.get("/api/stock/{ticker}/summary")
async def get_stock_summary(ticker: str):
    """Get current price, company name, and daily change for a ticker."""
    try:
        t = yf.Ticker(ticker.upper())
        info = t.fast_info
        hist = t.history(period="2d")

        current_price = float(getattr(info, "last_price", None) or 0)
        prev_close = float(getattr(info, "previous_close", None) or 0)

        if not current_price and not hist.empty:
            current_price = float(hist["Close"].iloc[-1])
        if not prev_close and len(hist) >= 2:
            prev_close = float(hist["Close"].iloc[-2])

        change = current_price - prev_close if prev_close else 0
        change_pct = (change / prev_close * 100) if prev_close else 0

        full_info = t.info
        name = full_info.get("longName") or full_info.get("shortName") or ticker.upper()
        currency = full_info.get("currency", "USD")

        return {
            "symbol": ticker.upper(),
            "name": name,
            "price": round(current_price, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "currency": currency,
        }
    except Exception as e:
        raise HTTPException(
            status_code=404, detail=f"Could not fetch data for {ticker}: {str(e)}"
        )


@app.get("/api/stock/{ticker}/history")
async def get_stock_history(ticker: str, range: str = "1M"):
    """Get historical price data for a ticker. range: 1W, 1M, 3M, 1Y"""
    range_map = {"1W": "5d", "1M": "1mo", "3M": "3mo", "1Y": "1y"}
    period = range_map.get(range.upper(), "1mo")
    try:
        t = yf.Ticker(ticker.upper())
        hist = t.history(period=period)

        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data for {ticker}")

        data = [
            {"date": idx.strftime("%Y-%m-%d"), "price": round(float(row["Close"]), 2)}
            for idx, row in hist.iterrows()
        ]

        start_price = float(hist["Close"].iloc[0])
        end_price = float(hist["Close"].iloc[-1])
        pct_growth = (
            ((end_price - start_price) / start_price * 100) if start_price else 0
        )

        return {
            "symbol": ticker.upper(),
            "range": range.upper(),
            "data": data,
            "metadata": {
                "start_date": hist.index[0].strftime("%Y-%m-%d"),
                "end_date": hist.index[-1].strftime("%Y-%m-%d"),
                "start_price": round(start_price, 2),
                "end_price": round(end_price, 2),
                "percentage_growth": round(pct_growth, 2),
                "trading_days": len(hist),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stock/{ticker}/predictions")
async def get_stock_predictions(ticker: str):
    """Report the deployed prediction state for a ticker.

    The app does not currently expose live model forecasts, so this endpoint
    returns the latest market price together with an explicit unavailable state
    instead of fabricated prediction data.
    """
    try:
        symbol = ticker.upper()
        stock = yf.Ticker(symbol)
        info = stock.fast_info
        hist = stock.history(period="2d")

        current_price = float(getattr(info, "last_price", None) or 0)
        if not current_price and not hist.empty:
            current_price = float(hist["Close"].iloc[-1])

        return {
            "symbol": symbol,
            "status": "unavailable",
            "message": "No deployed prediction models are currently available for this ticker.",
            "current_price": round(current_price, 2) if current_price else None,
            "models": [],
            "ensemble": None,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Could not fetch prediction status for {ticker}: {str(exc)}",
        )


@app.get("/api/stock/{ticker}/sentiment/stream")
async def stream_stock_sentiment(ticker: str, request: Request):
    """Stream FinBERT sentiment analysis article-by-article as an SSE feed.

    Each scored article emits an ``article`` event. When all articles are done a
    ``done`` event is emitted with the final aggregated score.  Already-scored
    articles (DB cache) are emitted immediately; only genuinely new articles
    trigger FinBERT and arrive with a slight delay.
    """
    from Kaare.db import repository as repo
    from datetime import date

    symbol = ticker.upper()

    async def generate() -> AsyncGenerator[str, None]:
        import time

        today = date.today()
        start = today - timedelta(days=30)

        pool = getattr(request.app.state, "db_pool", None)
        client = KaareClient(pool=pool)
        if pool is None:
            await client.initialize()

        try:
            t_stream_start = time.perf_counter()

            # --- Phase 1: Finnhub fetch ---
            t0 = time.perf_counter()
            articles = await client._finnhub.fetch_raw_news(symbol, start, today)
            t_finnhub = time.perf_counter() - t0
            logger.info(
                "[%s] Finnhub fetch: %.2fs — %d articles",
                symbol,
                t_finnhub,
                len(articles),
            )

            if not articles:
                yield f"data: {json.dumps({'type': 'done', 'avg_score': None, 'label': 'Unavailable', 'article_count': 0})}\n\n"
                return

            # --- Phase 2: DB insert + cache lookup ---
            t0 = time.perf_counter()
            ids = await repo.insert_raw_news_returning_ids(client._db, articles)
            existing_scores = await repo.get_article_sentiment_by_ids(client._db, ids)
            t_db = time.perf_counter() - t0
            logger.info(
                "[%s] DB insert+cache lookup: %.2fs — %d/%d cached",
                symbol,
                t_db,
                len(existing_scores),
                len(articles),
            )

            total = len(articles)
            all_net_scores: list[float] = []
            current = 0

            # Emit cached articles immediately
            for art_id, article in zip(ids, articles):
                if art_id in existing_scores:
                    score = existing_scores[art_id]
                    all_net_scores.append(score["net_score"])
                    current += 1
                    headline = article.text[:120].split(".")[0]
                    yield f"data: {json.dumps({'type': 'article', 'headline': headline, 'net_score': round(score['net_score'], 4), 'label': _net_to_label(score['net_score']), 'from_cache': True, 'current': current, 'total': total})}\n\n"

            # Score unscored articles one-by-one and stream each result
            unscored_pairs = [
                (i, a) for i, a in zip(ids, articles) if i not in existing_scores
            ]

            if unscored_pairs:
                unscored_ids = [i for i, _ in unscored_pairs]
                unscored_articles = [a for _, a in unscored_pairs]
                new_scores: list[dict] = []

                # --- Phase 3: FinBERT inference ---
                t0 = time.perf_counter()
                async for score in client._analyzer.score_articles_stream(
                    [a.text for a in unscored_articles]
                ):
                    idx = len(new_scores)
                    new_scores.append(score)
                    all_net_scores.append(score["net_score"])
                    current += 1
                    headline = unscored_articles[idx].text[:120].split(".")[0]
                    yield f"data: {json.dumps({'type': 'article', 'headline': headline, 'net_score': round(score['net_score'], 4), 'label': _net_to_label(score['net_score']), 'from_cache': False, 'current': current, 'total': total})}\n\n"

                t_finbert = time.perf_counter() - t0
                n = len(unscored_pairs)
                logger.info(
                    "[%s] FinBERT inference: %.2fs total — %d articles, %.3fs/article",
                    symbol,
                    t_finbert,
                    n,
                    t_finbert / n,
                )

                await repo.insert_article_sentiment_batch(
                    client._db, list(zip(unscored_ids, new_scores))
                )
                await repo.aggregate_daily_ticker_sentiment(client._db)

            avg = sum(all_net_scores) / len(all_net_scores) if all_net_scores else 0.0
            response = {
                "symbol": symbol,
                "avg_score": round(avg, 4),
                "label": _net_to_label(avg),
                "article_count": total,
            }
            _sentiment_cache[symbol] = (datetime.now(timezone.utc), response)
            logger.info(
                "[%s] Stream complete: %.2fs total",
                symbol,
                time.perf_counter() - t_stream_start,
            )
            yield f"data: {json.dumps({'type': 'done', **response})}\n\n"

        except Exception as exc:
            logger.exception("Sentiment stream failed for %s: %s", symbol, exc)
            yield f"data: {json.dumps({'type': 'done', 'avg_score': None, 'label': 'Unavailable', 'article_count': 0})}\n\n"
        finally:
            if pool is None:
                await client.close()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _net_to_label(score: float) -> str:
    if score >= 0.5:
        return "Very Positive"
    if score >= 0.1:
        return "Positive"
    if score > -0.1:
        return "Neutral"
    if score > -0.5:
        return "Negative"
    return "Very Negative"


@app.get("/api/stock/{ticker}/sentiment")
async def get_stock_sentiment_summary(ticker: str, request: Request):
    """Get FinBERT news sentiment summary for a ticker.

    Results are cached in-process for 1 hour per ticker to avoid
    re-running the full Finnhub + FinBERT pipeline on every page load.
    """
    from datetime import date

    symbol = ticker.upper()

    # Return cached result if still fresh
    now = datetime.now(timezone.utc)
    cached = _sentiment_cache.get(symbol)
    if cached is not None:
        cached_at, cached_result = cached
        if now - cached_at < _SENTIMENT_TTL:
            return cached_result

    today = date.today()
    start = today - timedelta(days=30)

    try:
        # Use the shared pool if available (avoids per-request pool creation)
        pool = getattr(request.app.state, "db_pool", None)
        client = KaareClient(pool=pool)
        if pool is None:
            await client.initialize()
        try:
            result = await client.get_stock_news_sentiment(symbol, start, today)
        finally:
            if pool is None:
                await client.close()

        score = float(result.avg_score)
        response = {
            "symbol": symbol,
            "avg_score": round(score, 4),
            "label": _net_to_label(score),
            "article_count": result.article_count,
        }
        _sentiment_cache[symbol] = (now, response)  # now is timezone-aware
        return response

    except Exception as exc:
        logger.exception("Sentiment endpoint failed for %s: %s", symbol, exc)
        return {
            "symbol": symbol,
            "avg_score": None,
            "label": "Unavailable",
            "article_count": 0,
        }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))

    uvicorn.run(app, host=host, port=port)
