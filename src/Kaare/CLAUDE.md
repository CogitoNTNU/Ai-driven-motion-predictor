# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install dependencies:**
```bash
uv sync
```

**Run tests:**
```bash
uv run pytest --doctest-modules --cov=src --cov-report=html
```

**Run a single test:**
```bash
uv run pytest tests/test_example.py::test_name
```

**Linting (via pre-commit):**
```bash
uv run pre-commit run --all-files
```

**Documentation:**
```bash
uv run mkdocs serve   # preview at http://127.0.0.1:8000/
uv run mkdocs build
```

**Start database services:**
```bash
docker compose up -d  # PostgreSQL on port 5432, pgAdmin on port 5050
```

## Architecture

This is an AI-driven stock motion prediction project. The core flow is:

1. **Data ingestion** — `src/yfinanceLoader.py` fetches historical price data from Yahoo Finance using the `yfinance` library and converts it to NumPy arrays.
2. **Storage** — A PostgreSQL database (via Docker) stores OHLCV market data. SQL query patterns are in `queries/example_queries.sql` (uses DuckDB syntax).
3. **Modeling** — Jupyter notebooks in `src/Practice/` and `src/tasks.ipynb` contain exploratory analysis and model development (scikit-learn).
4. **Module** — `src/Kaare/` is the primary module directory for this branch's work.

## Key conventions

- **Package manager**: `uv` (not pip). Always use `uv run` to execute scripts/tools.
- **Python version**: 3.12+
- **Docstrings**: Google style (enforced by mkdocstrings).
- **Commits**: Conventional commits format (enforced by Commitizen via pre-commit).
- **Code style**: Ruff for formatting and linting; nbQA for notebooks.
- **Environment**: Copy `.env.example` to `.env` and fill in credentials before starting Docker services.
