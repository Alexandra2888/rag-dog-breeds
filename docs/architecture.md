# Architecture

## Components

| Component | Tech | Port | Role |
|---|---|---|---|
| Frontend | Next.js (App Router, React 19, Tailwind v4) | 3000 | Text + voice chat UI |
| API | FastAPI (Python 3.11+, `uv`) | 8000 | RAG query/search, ingestion, LiveKit token minting, cache admin |
| Voice agent | livekit-agents 1.x | — | Real-time spoken Q&A (worker process) |
| Database | Postgres + pgvector (`pgvector/pgvector:pg16`) | 5433→5432 | Chunks, embeddings, answer cache |
| Embeddings | Default: local **fine-tuned sentence-transformers** `bge-base-dogbreeds` (via `ST_MODEL_PATH`). Alternatives: Ollama `nomic-embed-text` (local) or OpenAI-compatible / Jina (cloud) | — | Priority `ST_MODEL_PATH` > `INFERENCE_PROVIDER=openai` > `ollama`; all 768-dim. See [rag-pipeline.md](rag-pipeline.md#3-embeddings-embeddingspy), [fine-tuning.md](fine-tuning.md) |
| Chat / generation | Pluggable: Ollama (local, llama3.x) or any OpenAI-compatible API (prod: **OpenAI `gpt-4o-mini`**) | 11434 (Ollama) | Switched by `INFERENCE_PROVIDER`; see [deployment.md](deployment.md) |
| Speech | OpenAI STT (`gpt-4o-transcribe`) + TTS | — | Voice only |
| Realtime transport | LiveKit server (or LiveKit Cloud) | 7880/7881 | WebRTC media for voice |

The text path and the voice path are **separate processes** but share the same
`RAGService`, Postgres, Ollama, and — crucially — the same answer cache.

## Text data flow

```
Browser ──POST /query──► FastAPI ──► RAGService.query(mode="text")
                                        │
                       ┌── cache hit ───┴── cache miss ──┐
                       │                                  │
              return cached answer            embed query (fine-tuned bge, query-instruction prefix)
              (no LLM, ~3ms)                  hybrid search (pgvector + FTS + trgm + breed)
                                              generate answer (configured chat model; prod OpenAI gpt-4o-mini)
                                              store in query_cache
                                        │
                                        ▼
                          { answer, chunks[], cached }  ──► rendered in chat
```

See [rag-pipeline.md](rag-pipeline.md) for retrieval internals and
[caching.md](caching.md) for the cache.

## Voice data flow

```
Mic ─► LiveKit room ─► Voice agent (AgentSession)
                          │  on_user_turn_completed(user_text)
                          ▼
                 RAGService.query(mode="voice")   ← shared cache with text
                          │
              ┌── answer ─┴── failure ──┐
              ▼                          ▼
   session.say(answer)        inject context → default Ollama LLM reply
   raise StopResponse()       (fallback path, never leaves user without a reply)
              │
              ▼
   OpenAI TTS ─► speakers
```

The browser obtains a LiveKit room + token from FastAPI (`POST /api/voice/session`),
then connects directly to the LiveKit server. The agent joins the room
automatically and answers. STT and TTS use OpenAI; the LLM and retrieval are local.

## Why this shape

- **Local-first (still)**: the whole stack runs on one machine with Ollama +
  local models, so there's no per-query API cost and no data egress for the core
  RAG. The retriever's default is now a **fine-tuned bge model loaded locally**
  from `ST_MODEL_PATH` — no external embedding call either. The same image also
  deploys to the cloud (Vercel → Fly self-hosted bge API → Neon → OpenAI); see
  [deployment.md](deployment.md).
- **Shared cache across processes**: because text (FastAPI) and voice (LiveKit
  worker) run separately, the cache must be external — Postgres — so a question
  asked in one mode benefits the other and survives restarts. See
  [caching.md](caching.md).
- **One book, structured facts**: the corpus is a single breed catalogue with
  labeled fields (Origin / Weight / Height / Life span). The pipeline and evals
  lean on that structure (breed-aware chunking, breed-label retrieval, reference-
  based eval questions).

## Observability

Structured logging is wired through **structlog**: all stdlib logging is routed
through a single JSON formatter, a `request_id` contextvar is bound by a FastAPI
middleware and auto-attached to every log line across modules, and each request
emits an access line. This gives correlated, machine-parseable logs in
production without per-module wiring.

On top of that, **online eval** records per-query quality signals on real traffic
(retrieval-score stats + deterministic refusal detection on every query, a sampled
reference-free LLM judge on a fraction) to the `online_eval` table via a
background task — no added request latency — plus a `POST /feedback` thumbs
signal. A cron-able CLI rolls windows up into an `online-eval` MLflow experiment so
quality **drift** is a trend line. See [observability.md](observability.md).

## Process / deployment topology

`docker-compose.yml` defines four services: `postgres`, `rag-api`,
`livekit-server`, `livekit-agent`. For local dev you typically run only
`postgres` in Docker and run the API + agent + frontend on the host for
hot-reload. See [deployment.md](deployment.md) and [development.md](development.md).
