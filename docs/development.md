# Development

## Prerequisites

- Python 3.11+ and [`uv`](https://github.com/astral-sh/uv)
- Node + [`bun`](https://bun.sh) (or npm) for the frontend
- Docker (for Postgres+pgvector)
- [Ollama](https://ollama.com) with `nomic-embed-text` and `llama3.1:8b`
- For voice: `OPENAI_API_KEY` and LiveKit credentials (Cloud or local server)

## Backend

```bash
cd server
docker compose up -d postgres          # Postgres+pgvector on host port 5433
cp .env.example .env                   # then edit (see docs/configuration.md)
uv sync                                # install deps (incl. dev: ragas, langchain-ollama)

# Run the API (auto-ingests data/*.pdf on first start)
uv run uvicorn src.main:app --reload   # http://localhost:8000  · docs at /docs
```

Manual ingest / re-ingest:
```bash
uv run python -m src.ingest            # idempotent
uv run python -m src.ingest --force    # re-ingest (replaces existing)
```

Smoke-test retrieval + cache from the shell:
```bash
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"query":"Where does the Weimaraner come from?","top_k":8}' | jq '{answer, cached}'
```

## Frontend

```bash
cd client
bun install        # or npm install
bun dev            # http://localhost:3000
```
Set `NEXT_PUBLIC_RAG_API_URL` in `client/.env.local` if the API isn't on
`localhost:8000`.

## Voice agent

Easiest: talk in your terminal (no frontend/LiveKit room needed):
```bash
cd server
uv run python -m src.livekit_agent download-files   # one-time (Silero VAD, etc.)
uv run python -m src.livekit_agent console
```

As a worker for a real room / the frontend:
```bash
uv run python -m src.livekit_agent dev     # hot-reload; registers with LiveKit in .env
```
Requires `OPENAI_API_KEY` (STT/TTS) and LiveKit credentials. The browser gets a
room+token from `POST /api/voice/session`.

## Evaluation

```bash
cd server
uv run python -m evals.run_eval --limit 3   # quick
uv run python -m evals.run_eval             # full (slow on a local 8B judge)
```
See [evaluation.md](evaluation.md).

## Useful checks

```bash
# What's ingested
curl -s localhost:8000/documents | jq

# Inspect the DB directly
psql postgresql://postgres:postgres@localhost:5433/ragdb -c \
  "select metadata->>'breed', count(*) from chunks where metadata ? 'breed' group by 1 order by 2 desc limit 10;"

# Clear the answer cache
curl -X DELETE localhost:8000/cache
```

## Troubleshooting

- **DB connection refused** → is the `rag-postgres` container up? Is `DATABASE_URL`
  on port **5433** (not 5432)? `docker compose ps`, `docker compose logs postgres`.
- **Embedding/dimension errors** → the schema is `vector(768)` for
  `nomic-embed-text`; a different model needs a schema change.
- **Empty / bad answers** → confirm the PDF ingested (`/documents`) and Ollama has
  the models (`ollama list`).
- **Voice silent / errors** → check `OPENAI_API_KEY` and LiveKit creds; the agent
  logs whether it served a cached answer or fell back to the LLM.
- **`import ragas` fails** → the eval deps pin the langchain 0.3.x line; run
  `uv sync` so the lockfile is applied.
