# AI-Driven Motion Predictor

AI-Driven Motion Predictor is a full-stack stock analysis workspace with:

- a React dashboard for price history, sentiment, and chat-driven analysis
- a FastAPI backend that streams tool-backed responses to the frontend
- Kaare data and FinBERT sentiment tooling for stock and news pipelines

## Local Development

1. Sync the root Python environment with `uv sync --group dev`.
1. Sync the API project with `uv sync --project src/api`.
1. Install frontend dependencies with `pnpm install --dir src/app`.
1. Start optional local database services with `docker compose up -d`.

## Validation

- `uv run pre-commit run --all-files`
- `uv run pytest tests`
- `uv run --project src/api python -m unittest discover -s tests`
- `pnpm --dir src/app test`
- `pnpm --dir src/app build`

## More Information

- Root setup and architecture: `README.md`
- Frontend details: `src/app/README.md`
- API details: `src/api/README.md`
