# Configuration

## Backend (`server/.env`)

Loaded by `src/config.py` (pydantic-settings). Defaults shown; the committed
`.env` overrides some (e.g. DB port `5433`, chat model `llama3.1:8b`).

| Variable | Default | Notes |
|---|---|---|
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/ragdb` | Use **5433** locally to match the Docker port mapping (`5433:5432`); a Neon/Supabase URL in the cloud |
| `INFERENCE_PROVIDER` | `ollama` | `ollama` (local) or `openai` (any OpenAI-compatible API, e.g. Gemini) |
| `INFERENCE_BASE_URL` | `""` | OpenAI-compatible endpoint, e.g. `https://generativelanguage.googleapis.com/v1beta/openai/` |
| `INFERENCE_API_KEY` | `""` | Provider key (Gemini key for the free deploy) |
| `INFERENCE_CHAT_MODEL` | `gemini-2.5-flash` | Chat model (provider `openai`) |
| `INFERENCE_EMBEDDING_MODEL` | `gemini-embedding-001` | Embedding model; free deploy uses `jina-embeddings-v2-base-en` |
| `INFERENCE_EMBEDDING_DIM` | `768` | Requested dimension; `0` to omit (Jina v2 is fixed 768) |
| `INFERENCE_EMBEDDING_BASE_URL` | `""` | Separate embeddings endpoint (e.g. `https://api.jina.ai/v1`); falls back to `INFERENCE_BASE_URL` |
| `INFERENCE_EMBEDDING_API_KEY` | `""` | Separate embeddings key (Jina); falls back to `INFERENCE_API_KEY` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API (provider `ollama`) |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | 768-dim; uses asymmetric task prefixes |
| `OLLAMA_CHAT_MODEL` | `llama3.2` | Committed `.env` uses `llama3.1:8b` |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins; set to your Vercel URL in prod |
| `CHUNK_SIZE` | `1000` | Fallback size chunking only |
| `CHUNK_OVERLAP` | `200` | Fallback size chunking only |
| `API_HOST` | `0.0.0.0` | |
| `API_PORT` | `8000` | |
| `LIVEKIT_URL` | `ws://localhost:7880` | LiveKit Cloud uses `wss://...livekit.cloud` |
| `LIVEKIT_API_KEY` | `""` | Required for voice token minting |
| `LIVEKIT_API_SECRET` | `""` | Required for voice token minting |
| `LIVEKIT_AGENT_PORT` | `8080` | |
| `OPENAI_API_KEY` | `""` | Required for voice STT + TTS |

Notes:
- The DB schema is `vector(768)`. All providers produce 768-dim vectors
  (`nomic-embed-text` natively; `jina-embeddings-v2-base-en` natively;
  `gemini-embedding-001` via `INFERENCE_EMBEDDING_DIM=768`). Switching the
  embedding model means re-ingesting (new vector space).
- **Local dev** uses Ollama (`INFERENCE_PROVIDER=ollama`, default). **Free cloud
  deploy** uses **Gemini for chat + Jina for embeddings** (`INFERENCE_PROVIDER=openai`
  plus `INFERENCE_EMBEDDING_*`) — see [deployment.md](deployment.md). Keep prod
  values in `server/.env.prod` (gitignored).
- Voice needs `OPENAI_API_KEY` (STT/TTS) **and** LiveKit credentials. Text chat
  needs neither.

## Frontend (`client/.env.local`)

| Variable | Default | Notes |
|---|---|---|
| `NEXT_PUBLIC_RAG_API_URL` | `http://localhost:8000` | Browser → FastAPI base URL |
| `RAG_API_URL` | `http://localhost:8000` | Server-side fallback (SSR) |

> Anything prefixed `NEXT_PUBLIC_` is bundled into client JS and visible to users.
> Never put secrets (API keys/secrets) behind that prefix. The legacy
> `lib/livekit-config.ts` exposes such fields and should not be populated; voice
> tokens are minted server-side via `POST /api/voice/session`.

## Ollama models

```bash
ollama pull nomic-embed-text
ollama pull llama3.1:8b   # or your OLLAMA_CHAT_MODEL
```
