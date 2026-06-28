# Architecture

## Components

| Component | Tech | Port | Role |
|---|---|---|---|
| Frontend | Next.js (App Router, React 19, Tailwind v4) | 3000 | Text + voice chat UI |
| API | FastAPI (Python 3.11+, `uv`) | 8000 | RAG query/search, ingestion, LiveKit token minting, cache admin |
| Voice agent | livekit-agents 1.x | — | Real-time spoken Q&A (worker process) |
| Database | Postgres + pgvector (`pgvector/pgvector:pg16`) | 5433→5432 | Chunks, embeddings, answer cache |
| LLM + embeddings | Pluggable: Ollama (local) or any OpenAI-compatible API (Gemini in the free cloud deploy) | 11434 (Ollama) | Switched by `INFERENCE_PROVIDER`; see [deployment.md](deployment.md) |
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
              return cached answer            embed query (nomic, search_query)
              (no LLM, ~3ms)                  hybrid search (pgvector + FTS + trgm + breed)
                                              generate answer (Ollama llama3.1)
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

- **Local-first**: retrieval + generation never leave the machine, so there's no
  per-query API cost and no data egress for the core RAG. Only speech I/O is cloud.
- **Shared cache across processes**: because text (FastAPI) and voice (LiveKit
  worker) run separately, the cache must be external — Postgres — so a question
  asked in one mode benefits the other and survives restarts. See
  [caching.md](caching.md).
- **One book, structured facts**: the corpus is a single breed catalogue with
  labeled fields (Origin / Weight / Height / Life span). The pipeline and evals
  lean on that structure (breed-aware chunking, breed-label retrieval, reference-
  based eval questions).

## Process / deployment topology

`docker-compose.yml` defines four services: `postgres`, `rag-api`,
`livekit-server`, `livekit-agent`. For local dev you typically run only
`postgres` in Docker and run the API + agent + frontend on the host for
hot-reload. See [deployment.md](deployment.md) and [development.md](development.md).
