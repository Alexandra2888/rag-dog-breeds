# Dog Breed RAG Assistant

A full-stack Retrieval-Augmented Generation (RAG) app that answers questions
about dog breeds — by **text or voice** — grounded in a single source book
(*The Complete Dog Breed Book*). The retriever is a **fine-tuned
`bge-base-en-v1.5`** embedder trained on the corpus (recall@5 **0.80 → 0.84**
vs off-the-shelf bge). Both embeddings and generation are **pluggable**: run
fully local on [Ollama](https://ollama.com), or point at cloud providers. The
live demo runs on Vercel + Fly.io + Neon Postgres with OpenAI chat.

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
            ┌───────────────────────────┐   ┌────────────────────────────┐
            │ Postgres + pgvector (5433) │   │ fine-tuned bge embedder    │
            │ chunks · query_cache       │   │ (768-dim) · chat LLM       │
            │                            │   │ Ollama local │ or OpenAI   │
            └───────────────────────────┘   └────────────────────────────┘
```

## Highlights

- **Fine-tuned retriever** — `bge-base-en-v1.5` fine-tuned on ~1,193 synthetic
  dog-breed query→passage pairs (MultipleNegativesRankingLoss). On a held-out,
  judge-free eval it lifts recall@5 **0.80 → 0.84**, recall@3 0.73 → 0.79, and
  MRR 0.67 → 0.72 vs off-the-shelf bge — and far above the prior
  nomic-embed-text (recall@5 0.32). 768-dim, so it drops into the existing
  pgvector schema. Selected via `ST_MODEL_PATH`; see [fine-tuning.md](docs/fine-tuning.md).
- **Pluggable providers** — embeddings (local fine-tuned bge · Jina/Gemini
  cloud · nomic on Ollama) and chat (Ollama `llama3.x` · OpenAI `gpt-4o-mini`,
  Gemini, Anthropic) each swap via env vars. Prod uses fine-tuned bge + OpenAI.
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
  retrieval check, scored by a local Ollama judge (no API cost), plus an
  **adversarial / edge-case** suite (jailbreaks, prompt-injection, out-of-scope)
  scored per category by a deterministic refusal metric. See
  [evaluation.md](docs/evaluation.md).
- **Structured logging + online eval** — structlog with a per-request `request_id`
  threaded through text and voice (JSON logs in prod), plus **production quality
  monitoring**: per-query signals + a sampled LLM judge + thumbs feedback on real
  traffic, with drift tracked in MLflow. See [observability.md](docs/observability.md).

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

# Embeddings: prod uses the fine-tuned bge model, selected via ST_MODEL_PATH.
# A fresh clone either trains it (uv run python -m finetune.train, saved to
# finetune/models/bge-base-dogbreeds) or falls back to nomic (local) / Jina
# (cloud) if ST_MODEL_PATH is unset. Chat defaults to Ollama llama3.x locally.

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
- [Fine-tuning](docs/fine-tuning.md) — synthetic pairs → bge fine-tune (MNRL) → recall@k/MRR, MLflow
- [Design decisions](docs/design-decisions.md) — **why** it's built this way (interview-ready)
- [API reference](docs/api-reference.md) — every endpoint
- [Caching](docs/caching.md) — the shared answer cache
- [Evaluation](docs/evaluation.md) — running and extending the Ragas suite
- [Observability](docs/observability.md) — structured logging and per-request `request_id`
- [Configuration](docs/configuration.md) — environment variables
- [Development](docs/development.md) — local setup, voice console, troubleshooting
- [Deployment](docs/deployment.md) — Vercel + Fly.io + managed Postgres

## License

MIT
