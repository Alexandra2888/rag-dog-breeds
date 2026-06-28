# Deployment

## Docker Compose stack

`server/docker-compose.yml` defines four services:

| Service | Image / build | Ports | Notes |
|---|---|---|---|
| `postgres` | `pgvector/pgvector:pg16` | `5433:5432` | Persistent volume `postgres_data` |
| `rag-api` | local `Dockerfile` | `8000:8000` | FastAPI; mounts `./data` read-only |
| `livekit-server` | `livekit/livekit-server:latest` | `7880/7881`, `50000-50100/udp` | Optional self-hosted realtime transport |
| `livekit-agent` | local `Dockerfile` | — | Voice worker |

```bash
cd server
docker compose up -d                 # full stack
docker compose up -d postgres        # just the DB (typical for local dev)
docker compose ps
docker compose logs -f rag-api
```

Containers reach the host's Ollama via `host.docker.internal` (mapped through
`host-gateway`). Set `OLLAMA_BASE_URL=http://host.docker.internal:11434` in the
container environment if needed.

## First run

On API startup, PDFs in `data/` are auto-ingested (idempotent). To pre-seed
before starting the agent, run `uv run python -m src.ingest`.

## Production checklist

- **CORS**: the API currently allows all origins (`*`) with credentials — restrict
  `allow_origins` to your frontend origin.
- **Ingestion**: disable or path-restrict `POST /ingest` (it reads arbitrary
  server paths); prefer `POST /ingest/upload`.
- **Auth**: endpoints are unauthenticated, including LiveKit token minting
  (`/api/voice/session`) — add an API key / session check before exposing.
- **Secrets**: keep `OPENAI_API_KEY`, LiveKit secret, and DB credentials in the
  environment, never in `NEXT_PUBLIC_*`.
- **Models**: pin `OLLAMA_CHAT_MODEL` / `OLLAMA_EMBEDDING_MODEL`; the DB schema is
  `vector(768)` for `nomic-embed-text`.
- **Frontend**: `bun run build && bun run start` (or deploy to Vercel) with
  `NEXT_PUBLIC_RAG_API_URL` pointing at the API.

These security items are intentionally relaxed for a local single-user demo; see
the notes in [api-reference.md](api-reference.md).
