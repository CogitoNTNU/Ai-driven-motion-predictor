# AI-Driven Motion Predictor

A full-stack application for AI-powered stock analysis featuring a React frontend chat interface and a FastAPI backend
with LangGraph agents.

<div align="center">

![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/CogitoNTNU/AI-driven-motion-predictor/ci.yml)
![GitHub top language](https://img.shields.io/github/languages/top/CogitoNTNU/AI-driven-motion-predictor)
![GitHub language count](https://img.shields.io/github/languages/count/CogitoNTNU/AI-driven-motion-predictor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Project Version](https://img.shields.io/badge/version-0.0.1-blue)](https://img.shields.io/badge/version-0.0.1-blue)

<img src="docs/images/cogito_white.svg" width="25%" alt="Cogito Project Logo" style="display: block; margin-left: auto; margin-right: auto;">
</div>

<details>
<summary><b>📋 Table of contents </b></summary>

- [AI-Driven Motion Predictor](#ai-driven-motion-predictor)
  - [Project Structure](#project-structure)
  - [Description](#description)
  - [Prerequisites](#prerequisites)
  - [Quick Start](#quick-start)
  - [Frontend](#frontend)
  - [Backend](#backend)
  - [Environment Variables](#environment-variables)
  - [Documentation](#documentation)
  - [Testing](#testing)
  - [Team](#team)
  - [License](#license)

</details>

## Project Structure

```
AI-driven-motion-predictor/
├── src/
│   ├── app/                 # React frontend (see src/app/README.md)
│   │   ├── package.json     # Node.js dependencies: React 19, Vite, Tailwind CSS
│   │   └── ...
│   ├── api/                 # FastAPI backend (see src/api/README.md)
│   │   ├── pyproject.toml   # Python dependencies: FastAPI, LangGraph, LangChain
│   │   ├── main.py          # FastAPI application entry point
│   │   └── ...
│   └── Kaare/               # Core ML/stock analysis library
├── pyproject.toml           # Root Python project configuration
├── requirements.txt         # Legacy pip requirements
├── .env.example             # Environment variable template
└── README.md                # This file
```

## Description

<!-- TODO: Provide a brief overview of what this project does and its key features. Please add pictures or videos of the application -->

This project provides an AI-powered stock analysis platform with:

- **Frontend**: React 19 + TypeScript + Vite chat interface using shadcn/ui components and Vercel AI SDK for streaming
  responses
- **Backend**: FastAPI server with LangGraph multi-agent architecture (supervisor pattern) for stock market analysis
- **Core ML**: PyTorch, Transformers, and financial data integrations (yfinance, Finnhub)

## Prerequisites

### Required

- **Git**: Version control [Download Git](https://git-scm.com/downloads)
- **Python**: 3.12 or 3.13 [Download Python](https://www.python.org/downloads/)
- **uv**: Modern Python package manager [Install UV](https://docs.astral.sh/uv/getting-started/installation/)
  ```sh
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Node.js**: 20 or higher [Download Node.js](https://nodejs.org/)
- **pnpm**: Package manager for Node.js
  ```sh
  npm install -g pnpm
  # or
  curl -fsSL https://get.pnpm.io/install.sh | sh -
  ```

### Optional

- **Docker**: For containerized development [Download Docker](https://www.docker.com/products/docker-desktop)
- **PostgreSQL/ParadeDB**: For persistent data storage (can use Docker)

## Quick Start

### 1. Clone the Repository

```sh
git clone https://github.com/CogitoNTNU/AI-driven-motion-predictor.git
cd AI-driven-motion-predictor
```

### 2. Configure Environment Variables

Copy the example environment file:

```sh
cp .env.example .env
```

Edit `.env` with your API keys:

```env
# Required for LLM functionality
OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=https://api.openai.com/v1

# Required for stock data
FINNHUB_API_KEY=your-finnhub-key
ALPHAVANTAGE_API_KEY=your-alphavantage-key

# Required for ML models
HF_TOKEN=hf-your-huggingface-token

# Optional: Database (uses defaults if not set)
DB_USER=admin
DB_PASSWORD=changeme
DB_NAME=motion_predictor
```

### 3. Install Dependencies

**Root Python project (for core ML library):**

```sh
uv sync
```

**Backend API:**

```sh
cd src/api
uv sync
cd ../..
```

**Frontend:**

```sh
cd src/app
pnpm install
cd ../..
```

### 4. Start Development Servers

**Terminal 1 - Backend:**

```sh
cd src/api
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**

```sh
cd src/app
pnpm dev
```

The application will be available at:

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs (Swagger UI)

## Frontend

The frontend is a React 19 + TypeScript application built with Vite and styled with Tailwind CSS 4 and shadcn/ui
components.

**Key technologies** (from `src/app/package.json`):

- React 19.2.4 with TypeScript 5.9.3
- Vite 7.3.1 build tool
- Tailwind CSS 4.2.1
- Vercel AI SDK React 3.0.118 for streaming chat
- Recharts 3.8.0 for data visualization
- shadcn/ui component library with Radix UI primitives

**Quick commands:**

```sh
cd src/app

# Install dependencies
pnpm install

# Start development server
pnpm dev

# Build for production
pnpm build

# Format code
pnpm format

# Type checking
pnpm typecheck
```

**Detailed documentation:** See [src/app/README.md](src/app/README.md)

## Backend

The backend is a FastAPI server providing stock analysis through LangGraph agents with streaming responses.

**Key technologies** (from `src/api/pyproject.toml`):

- FastAPI 0.135.1 with Uvicorn 0.30.0
- Python 3.12 or 3.13
- LangGraph 0.2.0+ for agent workflows
- LangChain 0.3.0+ and LangChain-OpenAI 0.2.0+
- LangGraph-Supervisor 0.0.15+ for multi-agent orchestration
- yfinance 0.2.0+ for financial data

**Quick commands:**

```sh
cd src/api

# Install dependencies
uv sync

# Run development server with hot reload
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or run directly
uv run main.py
```

**API Endpoints:**

- `GET /` - API info
- `GET /health` - Health check
- `POST /api/chat` - Streaming chat endpoint (AI SDK format)
- `POST /api/chat/sync` - Synchronous chat endpoint

**Detailed documentation:** See [src/api/README.md](src/api/README.md)

## Environment Variables

The application uses environment variables from `.env` in the project root. Key variables:

| Variable               | Required | Description                                       |
| ---------------------- | -------- | ------------------------------------------------- |
| `OPENAI_API_KEY`       | Yes      | OpenAI API key for LLM                            |
| `OPENAI_BASE_URL`      | Yes      | OpenAI API base URL                               |
| `MODEL_NAME`           | No       | Model to use (default: gpt-4o-mini)               |
| `FINNHUB_API_KEY`      | Yes\*    | Finnhub API for stock data                        |
| `ALPHAVANTAGE_API_KEY` | Yes\*    | Alpha Vantage API for stock data                  |
| `HF_TOKEN`             | Yes      | HuggingFace token for FinBERT models              |
| `DB_USER`              | No       | PostgreSQL username                               |
| `DB_PASSWORD`          | No       | PostgreSQL password                               |
| `DB_NAME`              | No       | PostgreSQL database name                          |
| `DB_PORT`              | No       | PostgreSQL port (default: 5432)                   |
| `VITE_API_URL`         | No       | Frontend API URL (default: http://localhost:8000) |

\* At least one stock data provider required

See `.env.example` for the complete list.

## Documentation

To build and preview the documentation site locally:

```sh
# Install docs dependencies (from root pyproject.toml docs group)
uv sync --group docs

# Build and serve
uv run mkdocs build
uv run mkdocs serve
```

The documentation will be available at [http://127.0.0.1:8000/](http://127.0.0.1:8000/).

## Testing

Run the test suite from the project root:

```sh
uv run pytest --doctest-modules --cov=src --cov-report=html
```

View the coverage report at `htmlcov/index.html`.

### Set up pre-commit hooks (development):

```sh
uv run pre-commit install
```

## Team

This project would not have been possible without the hard work and dedication of all of the contributors. Thank you for
the time and effort you have put into making this project a reality.

<table align="center">
    <tr>
        <!--
        <td align="center">
            <a href="https://github.com/NAME_OF_MEMBER">
              <img src="https://github.com/NAME_OF_MEMBER.png?size=100" width="100px;" alt="NAME OF MEMBER"/><br />
              <sub><b>NAME OF MEMBER</b></sub>
            </a>
        </td>
        -->
    </tr>
</table>

![Group picture](docs/img/team.png)

## License

Distributed under the MIT License. See `LICENSE` for more information.
