# Deployment

Two supported targets:

- **Local / self-hosted** — Docker Compose (below).
- **Cloud** — **Frontend on Vercel**, **Backend on Fly.io** (Option A: Ollama on
  Fly), **Postgres on Supabase/Neon**, **voice via LiveKit Cloud + OpenAI**.

---

## Cloud: Vercel (frontend) + Fly.io (backend)

### Topology

```
Vercel (Next.js)  ──►  Fly: dog-breed-rag-api  ──►  Supabase/Neon Postgres (pgvector)
                                  │  └────────────►  Fly: dog-breed-rag-ollama (LLM+embeddings)
Browser ──WebRTC──► LiveKit Cloud ◄── Fly: dog-breed-rag-agent ──► OpenAI (STT/TTS)
```

Three Fly apps share one Docker image (`server/Dockerfile`) via the configs in
`server/`: [`fly.api.toml`](../server/fly.api.toml),
[`fly.agent.toml`](../server/fly.agent.toml),
[`fly.ollama.toml`](../server/fly.ollama.toml).

### 1. Database (Supabase or Neon)

Create a Postgres database and copy its connection string. `pgvector` and
`pg_trgm` are enabled automatically on first boot (the app runs
`CREATE EXTENSION IF NOT EXISTS …`). Note the DB is `vector(768)` for
`nomic-embed-text`.

### 2. Ollama app (LLM + embeddings)

```bash
cd server
fly apps create dog-breed-rag-ollama
fly volumes create ollama_models --size 20 -a dog-breed-rag-ollama -r fra
fly deploy -c fly.ollama.toml
# Pull the models into the volume (once):
fly ssh console -a dog-breed-rag-ollama -C "ollama pull nomic-embed-text"
fly ssh console -a dog-breed-rag-ollama -C "ollama pull llama3.1:8b"
```

CPU works but is slow on an 8B model; for good latency use a Fly **GPU** machine
(needs GPU access, ~$1–3/hr) — see the comments in `fly.ollama.toml`. The answer
cache hides repeated questions either way.

### 3. API app (FastAPI)

```bash
fly apps create dog-breed-rag-api
fly secrets set -c fly.api.toml \
  DATABASE_URL="postgresql://...supabase/neon..." \
  LIVEKIT_URL="wss://<project>.livekit.cloud" \
  LIVEKIT_API_KEY="..." LIVEKIT_API_SECRET="..."
# Edit ALLOWED_ORIGINS in fly.api.toml to your Vercel domain, then:
fly deploy -c fly.api.toml
```

On first boot the API auto-ingests the bundled PDF into the DB (idempotent).

### 4. Voice agent app

```bash
fly apps create dog-breed-rag-agent
fly secrets set -c fly.agent.toml \
  DATABASE_URL="postgresql://...same as API..." \
  LIVEKIT_URL="wss://<project>.livekit.cloud" \
  LIVEKIT_API_KEY="..." LIVEKIT_API_SECRET="..." \
  OPENAI_API_KEY="sk-..."
fly deploy -c fly.agent.toml
fly scale count 1 -c fly.agent.toml      # keep the worker always on
```

### 5. Frontend (Vercel)

Import the repo in Vercel with **root directory = `client/`** (Next.js
auto-detected). Set env var:

```
NEXT_PUBLIC_RAG_API_URL = https://dog-breed-rag-api.fly.dev
```

Then redeploy. Voice works from the Vercel-hosted site because the browser
connects to LiveKit Cloud directly using a token minted by the API.

### Networking notes

- The API/agent reach Ollama privately at `dog-breed-rag-ollama.internal:11434`
  (Fly 6PN) — no public Ollama port. The machine must stay running for `.internal`
  to resolve; for scale-to-zero use a Flycast service (`.flycast`).
- Put all three Fly apps in the **same region** and org for low-latency private
  networking.
- Single-app alternative: you can collapse API+agent into one Fly app with two
  `[processes]` groups instead of two apps.

### Pre-prod checklist

- [ ] `ALLOWED_ORIGINS` set to the Vercel domain (no `*`).
- [ ] Secrets via `fly secrets` (never in `[env]` or `NEXT_PUBLIC_*`).
- [ ] Add auth to `/api/voice/session` (token-mint abuse) and disable/restrict
      `/ingest` (reads arbitrary server paths) — see [api-reference.md](api-reference.md).
- [ ] Ollama models pulled into the volume.
- [ ] API `min_machines_running = 1` so startup ingest runs and avoids cold starts.

---

## Local: Docker Compose

`server/docker-compose.yml` defines four services:

| Service | Image / build | Ports | Notes |
|---|---|---|---|
| `postgres` | `pgvector/pgvector:pg16` | `5433:5432` | Persistent volume `postgres_data` |
| `rag-api` | local `Dockerfile` | `8000:8000` | FastAPI; mounts `./data` |
| `livekit-server` | `livekit/livekit-server:latest` | `7880/7881`, `50000-50100/udp` | Optional self-hosted transport |
| `livekit-agent` | local `Dockerfile` | — | Voice worker |

```bash
cd server
docker compose up -d                 # full stack
docker compose up -d postgres        # just the DB (typical for local dev)
docker compose logs -f rag-api
```

Containers reach the host's Ollama via `host.docker.internal` — set
`OLLAMA_BASE_URL=http://host.docker.internal:11434` in the container env if needed.
On API startup, PDFs in `data/` are auto-ingested (idempotent).
