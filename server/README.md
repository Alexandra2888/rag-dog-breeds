# Dog Breed RAG — Backend

FastAPI service that powers the [Dog Breed RAG Assistant](../README.md): PDF
ingestion, hybrid retrieval, answer generation, a shared answer cache, a LiveKit
voice agent, and a Ragas eval suite. Retrieval and generation run on local
**Ollama**; storage is **Postgres + pgvector**.

```
PDF → breed-aware chunking → Ollama embeddings → pgvector
                                                     │
question → hybrid retrieval (vector + FTS + trigram + breed-label, RRF) → Ollama LLM → answer
                                                     │
                                          shared answer cache (Postgres)
                                                     │
                              text: FastAPI /query   ·   voice: LiveKit agent
```

For the full design (and **why**), see [`../docs/`](../docs/README.md) —
especially [rag-pipeline](../docs/rag-pipeline.md) and
[design-decisions](../docs/design-decisions.md).

## Prerequisites

- Python 3.11+ and [`uv`](https://github.com/astral-sh/uv)
- Docker (Postgres + pgvector)
- [Ollama](https://ollama.com) with `nomic-embed-text` and `llama3.1:8b`
- Voice only: `OPENAI_API_KEY` (STT/TTS) and LiveKit credentials

```bash
ollama pull nomic-embed-text
ollama pull llama3.1:8b
```

## Setup

```bash
docker compose up -d postgres          # Postgres+pgvector on host port 5433
cp .env.example .env                   # edit as needed (see docs/configuration.md)
uv sync                                # install deps (incl. dev: ragas, langchain-ollama)
uv run uvicorn src.main:app --reload   # http://localhost:8000 · docs at /docs
```

The PDF in `data/` is **auto-ingested** on first startup (idempotent). Manual:

```bash
uv run python -m src.ingest            # ingest new PDFs in data/
uv run python -m src.ingest --force    # re-ingest (replaces existing)
```

## Endpoints (summary)

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/query` | RAG answer (+ cache, `cached` flag) |
| POST | `/search` | Hybrid retrieval, no answer |
| POST | `/ingest` · `/ingest/upload` | Ingest a PDF (path / upload) |
| GET | `/documents` | List ingested docs |
| DELETE | `/documents/{id}` | Delete a doc + chunks |
| DELETE | `/cache` | Clear the answer cache |
| POST | `/api/voice/session` | Mint a LiveKit room + token |
| DELETE | `/api/voice/session/{room}` | Tear down a voice room |

Full request/response shapes: [`../docs/api-reference.md`](../docs/api-reference.md).
Interactive: `/docs` (Swagger), `/redoc`.

## Voice agent

Built on **livekit-agents 1.x**. OpenAI handles STT (`gpt-4o-transcribe`) + TTS;
the LLM and RAG run locally. Each turn is answered through the **cached**
`RAGService.query`, spoken via `session.say()`, skipping the LLM on a cache hit
(with a safe fallback). Talk to it in your terminal:

```bash
uv run python -m src.livekit_agent download-files   # one-time
uv run python -m src.livekit_agent console          # speak via mic/speakers
uv run python -m src.livekit_agent dev              # run as a worker for a room/frontend
```

Requires `OPENAI_API_KEY` and LiveKit creds. See
[`../docs/architecture.md`](../docs/architecture.md) and
[`../docs/caching.md`](../docs/caching.md).

## Evaluation

[Ragas](https://docs.ragas.io) metrics + a deterministic retrieval check, judged
by local Ollama (no API cost):

```bash
uv run python -m evals.run_eval --limit 3   # quick smoke test
uv run python -m evals.run_eval             # full suite (slow on a local judge)
```

Details: [`evals/README.md`](evals/README.md) and
[`../docs/evaluation.md`](../docs/evaluation.md).

## Project structure

```
server/
├── docker-compose.yml      # Postgres + API + LiveKit server + agent
├── Dockerfile
├── pyproject.toml          # uv deps (runtime + dev: ragas, langchain-ollama)
├── data/                   # PDF knowledge base
├── evals/                  # Ragas eval suite (golden.jsonl, run_eval.py)
└── src/
    ├── main.py             # FastAPI app + routes + startup auto-ingest
    ├── config.py           # env settings
    ├── pdf_processor.py    # breed-aware chunking
    ├── embeddings.py       # Ollama embeddings (nomic task prefixes)
    ├── database.py         # pgvector ops, hybrid search (RRF), answer cache
    ├── rag_service.py      # retrieval + generation + cache orchestration
    ├── models.py           # Pydantic request/response models
    ├── ingest.py           # auto-ingest CLI + startup hook
    ├── livekit_agent.py    # voice agent (cached RAG, STT/TTS, fallback)
    └── livekit_server.py   # agent worker entrypoint
```

## Configuration & troubleshooting

See [`../docs/configuration.md`](../docs/configuration.md) (env vars) and
[`../docs/development.md`](../docs/development.md) (troubleshooting). Key gotchas:

- `DATABASE_URL` uses host port **5433** (Docker maps `5433:5432`).
- DB schema is `vector(768)` for `nomic-embed-text`; a different-dimension model
  needs a schema change.
- `import ragas` requires the pinned langchain 0.3.x line — run `uv sync`.

## License

MIT
