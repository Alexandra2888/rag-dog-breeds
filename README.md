# Dog Breed RAG Assistant

A full-stack, **local-first** Retrieval-Augmented Generation (RAG) app that answers
questions about dog breeds — by **text or voice** — grounded in a single source
book (*The Complete Dog Breed Book*). Retrieval and generation run on local
[Ollama](https://ollama.com) models; only speech-to-text / text-to-speech use a
cloud provider.

```
                          ┌──────────────────────────────┐
                          │      Next.js client (3000)     │
                          │   text chat  │   voice chat    │
                          └───────┬───────────────┬────────┘
                          REST    │               │ WebRTC (LiveKit)
                                  ▼               ▼
                    ┌─────────────────────┐  ┌─────────────────────┐
                    │  FastAPI API (8000)  │  │  LiveKit voice agent │
                    │  /query  /search     │  │  STT→RAG→LLM→TTS     │
                    │  /api/voice/session  │  └──────────┬──────────┘
                    └──────────┬──────────┘             │
                               │   shared RAGService + answer cache
                               ▼                         ▼
            ┌───────────────────────────┐   ┌────────────────────────┐
            │ Postgres + pgvector (5433) │   │   Ollama (11434)        │
            │ chunks · query_cache       │   │ nomic-embed-text · LLM  │
            └───────────────────────────┘   └────────────────────────┘
```

## Highlights

- **Hybrid retrieval** — Reciprocal Rank Fusion over four signals: dense vectors,
  full-text keywords, fuzzy trigrams (typo/STT tolerance), and a breed-label
  match. Near-perfect per-breed recall (top-1 39/40, top-5 40/40 on the eval set).
- **Breed-aware chunking** — each breed entry becomes its own chunk by detecting
  the book's stats info box, so a chunk's embedding represents a single breed.
- **Text + voice, one knowledge base** — both modes share the same retrieval,
  generation, and cache.
- **Shared answer cache** — repeated questions are served from Postgres (no
  embedding, search, or LLM call); ~9000× faster on a hit. Shared across the
  text and voice processes.
- **Evaluation suite** — [Ragas](https://docs.ragas.io) metrics + a deterministic
  retrieval check, scored by a local Ollama judge (no API cost).

## Repository layout

| Path | What it is |
|---|---|
| [`server/`](server/README.md) | FastAPI backend, RAG pipeline, LiveKit voice agent, evals |
| [`client/`](client/README.md) | Next.js frontend (text + voice chat UI) |
| [`docs/`](docs/README.md) | Architecture, pipeline, API, caching, evaluation, config |

## Quick start

```bash
# 1. Backend infra (Postgres+pgvector) and models
cd server
docker compose up -d postgres
ollama pull nomic-embed-text && ollama pull llama3.1:8b

# 2. Install + run the API (auto-ingests the PDF in data/ on first start)
uv sync
uv run uvicorn src.main:app --reload   # http://localhost:8000  (docs at /docs)

# 3. Frontend
cd ../client
bun install   # or npm install
bun dev        # http://localhost:3000
```

For voice, set `OPENAI_API_KEY` and LiveKit credentials — see
[`docs/development.md`](docs/development.md) and [`server/README.md`](server/README.md).

## Documentation

Start at [`docs/README.md`](docs/README.md). Key reads:

- [Architecture](docs/architecture.md) — components and data flow (text + voice)
- [RAG pipeline](docs/rag-pipeline.md) — chunking, embeddings, hybrid search, generation
- [Design decisions](docs/design-decisions.md) — **why** it's built this way (interview-ready)
- [API reference](docs/api-reference.md) — every endpoint
- [Caching](docs/caching.md) — the shared answer cache
- [Evaluation](docs/evaluation.md) — running and extending the Ragas suite
- [Configuration](docs/configuration.md) — environment variables
- [Development](docs/development.md) — local setup, voice console, troubleshooting
- [Deployment](docs/deployment.md) — Vercel + Fly.io + managed Postgres

## License

MIT
