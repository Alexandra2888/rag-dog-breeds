# Configuration

## Backend (`server/.env`)

Loaded by `src/config.py` (pydantic-settings). Defaults shown; the committed
`.env` overrides some (e.g. DB port `5433`, chat model `llama3.1:8b`).

| Variable | Default | Notes |
|---|---|---|
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/ragdb` | Use **5433** to match the Docker port mapping (`5433:5432`) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | 768-dim; uses asymmetric task prefixes |
| `OLLAMA_CHAT_MODEL` | `llama3.2` | Committed `.env` uses `llama3.1:8b` |
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
- The DB schema hardcodes `vector(768)` to match `nomic-embed-text`. Changing the
  embedding model to a different dimension requires updating the schema.
- Voice needs `OPENAI_API_KEY` **and** LiveKit credentials. Text chat needs
  neither.

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
