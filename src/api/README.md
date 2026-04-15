# Backend - AI-Driven Motion Predictor API

FastAPI server providing stock analysis capabilities through LangGraph agents. Streams AI responses using the Vercel AI SDK Data Stream Protocol for real-time chat interactions.

## Tech Stack

- **Framework**: FastAPI 0.135.1+ with Uvicorn 0.30.0+
- **Python**: 3.12 (specified in `pyproject.toml`: `requires-python = ">=3.12,<3.13"`)
- **AI/ML**: LangGraph 0.2.0+, LangChain 0.3.0+, LangChain-OpenAI 0.2.0+
- **Supervisor Pattern**: LangGraph-Supervisor 0.0.15+ for multi-agent orchestration
- **Environment**: python-dotenv 1.0.0+
- **Financial Data**: yfinance 0.2.0+

## Prerequisites

- **Python**: Version 3.12
- **uv**: Modern Python package manager (install via `curl -LsSf https://astral.sh/uv/install.sh | sh` or see [docs.astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/))

## Installation

1. **Navigate to the API directory**:

   ```sh
   cd src/api
   ```

1. **Create and activate virtual environment** (optional, uv handles this automatically):

   ```sh
   uv venv
   source .venv/bin/activate  # Linux/Mac
   # or: .venv\Scripts\activate  # Windows
   ```

1. **Install dependencies**:

   ```sh
   uv sync
   ```

   Or install in editable mode:

   ```sh
   uv pip install -e .
   ```

## Environment Variables

The backend requires environment variables from the root `.env` file. Copy the example:

```sh
cp ../../.env.example ../../.env
```

Required variables (from root `.env.example`):

```env
# PostgreSQL / ParadeDB
DB_USER=admin
DB_PASSWORD=changeme
DB_NAME=motion_predictor
DB_PORT=5432

# API Keys
ALPHAVANTAGE_API_KEY=your_api_key_here
FINNHUB_API_KEY=api_key_here
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx  # HuggingFace token for FinBERT

# LLM Configuration
OPENAI_BASE_URL=https://api.openai.com/v1  # or your custom endpoint
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

# Model Configuration
MODEL_NAME=gpt-4o-mini  # or your preferred model

# Server Configuration
PORT=8000
HOST=0.0.0.0

# Frontend URL (for CORS in production)
VITE_API_URL=http://localhost:8000
```

## Running the Server

### Development Mode (with auto-reload)

```sh
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Or directly:

```sh
uv run main.py
```

### Production Mode

```sh
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

The server will be available at `http://localhost:8000`.

## API Endpoints

### Health Check

- **GET** `/` - Root endpoint returning API info
- **GET** `/health` - Health check with timestamp

### Chat Endpoints

#### Streaming Chat (Primary)

- **POST** `/api/chat`
- Content-Type: `application/json`
- Accepts: AI SDK v5 UIMessage format or legacy format
- Returns: `text/event-stream` with Data Stream Protocol

**Request body:**

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Analyze AAPL stock"
    }
  ]
}
```

**Response format** (SSE with `data: ` prefix):

- `type: "start"` - Message start with ID
- `type: "text-start"` - Begin text block
- `type: "text-delta"` - Incremental text content
- `type: "text-end"` - End text block
- `type: "data-tool-call"` - Tool invocation
- `type: "data-tool-result"` - Tool result
- `type: "data-chart"` - Chart visualization data
- `type: "finish"` - Message completion
- `[DONE]` - Stream termination

#### Synchronous Chat (Testing)

- **POST** `/api/chat/sync`
- Returns complete response as JSON (non-streaming)

## Project Structure

```
src/api/
├── main.py              # FastAPI application entry point
├── pyproject.toml       # Python dependencies and metadata
├── uv.lock             # Locked dependency versions
├── agents/             # LangGraph agent implementations
│   ├── __init__.py
│   ├── graph.py        # Main agent graph definition
│   ├── nodes.py        # Agent node implementations
│   └── tools.py        # Tool definitions
└── prompts/            # System prompts for agents
    ├── __init__.py
    ├── supervisor.py   # Supervisor agent prompts
    ├── researcher.py   # Researcher agent prompts
    └── analyst.py      # Analyst agent prompts
```

## Key Dependencies

From `pyproject.toml`:

**Core Framework:**

- `fastapi[standard]>=0.135.1` - Web framework with standard extras
- `uvicorn>=0.30.0` - ASGI server
- `python-dotenv>=1.0.0` - Environment variable loading

**AI/ML Stack:**

- `langchain>=0.3.0` - LLM framework
- `langgraph>=0.2.0` - Agent workflow framework
- `langgraph-supervisor>=0.0.15` - Multi-agent supervisor pattern
- `langchain-openai>=0.2.0` - OpenAI integration

**Data Sources:**

- `yfinance>=0.2.0` - Yahoo Finance data
- `asyncpg>=0.31.0` - Async PostgreSQL driver
- `torch>=2.0.0` - PyTorch for ML models
- `transformers>=4.40.0` - HuggingFace transformers
- `datasets>=2.19` - HuggingFace datasets
- `finnhub-python>=1.4.19` - Finnhub API client

## Architecture

The backend uses a **supervisor pattern** with LangGraph:

1. **Supervisor Agent**: Orchestrates the conversation and delegates tasks
1. **Researcher Agent**: Gathers stock market data and financial information
1. **Analyst Agent**: Performs technical analysis and generates insights

Agents communicate through a shared state containing:

- `messages`: Conversation history (LangChain message objects)
- `charts`: Generated chart data for visualization

### Streaming Implementation

The streaming endpoint uses LangGraph's `astream` with `stream_mode=["messages", "updates"]` to:

- Stream LLM token chunks in real-time
- Track which agent/node is currently active
- Capture tool calls and their results
- Emit chart data when available

## CORS Configuration

Currently configured for development with open CORS:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Development Notes

### Adding New Tools

Tools are defined in `agents/tools.py` using LangChain's `@tool` decorator:

```python
from langchain_core.tools import tool


@tool(response_format="content_and_artifact")
def my_tool(param: str) -> tuple[str, dict]:
    """Tool description for the LLM."""
    result = do_something(param)
    artifact = {"chart_id": "chart_123", "data": result}
    return str(result), artifact
```

### Modifying Agent Behavior

Update prompts in the `prompts/` directory. Agents are instantiated in `agents/graph.py`.

### Testing the API

Test with curl:

```sh
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

Or use the sync endpoint for easier debugging:

```sh
curl -X POST http://localhost:8000/api/chat/sync \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

## Troubleshooting

**Module not found errors:**
Ensure you're running from within the `src/api` directory and using `uv run`.

**Environment variables not loading:**
The `.env` file should be in the project root (two levels up from `src/api`). Verify `load_dotenv()` is finding it.

**LLM API errors:**
Check that `OPENAI_API_KEY` and `OPENAI_BASE_URL` are correctly set in your `.env` file.

**Database connection errors:**
Verify PostgreSQL is running and `DB_*` variables are correct. The database is optional for basic chat functionality.
