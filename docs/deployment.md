# Deployment

Two supported targets:

- **Local / self-hosted** — Docker Compose (below).
- **Cloud** — **Frontend on Vercel**, **Backend on Fly.io** (Option A: Ollama on
  Fly), **Postgres on Supabase/Neon**, **voice via LiveKit Cloud + OpenAI**.

---

## Cloud: Vercel (frontend) + Fly.io (backend)

### Topology

```
Vercel (Next.js)  ──►  Fly app "dog-breed-rag"  ──►  Supabase/Neon Postgres (pgvector)
                        ├─ process: api          ──►  Fly: dog-breed-rag-ollama (LLM+embeddings)
                        └─ process: agent  ──► LiveKit Cloud ◄──Browser ;  agent ──► OpenAI (STT/TTS)
```

**Two** Fly apps:
- [`fly.toml`](../server/fly.toml) — one app running both the **API** and the
  **voice agent** as separate `[processes]` groups from `server/Dockerfile`.
- [`fly.ollama.toml`](../server/fly.ollama.toml) — the Ollama server (separate
  app: different image + GPU/volume).

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

### 3. API + agent app (one app, two processes)

```bash
fly apps create dog-breed-rag
fly secrets set \
  DATABASE_URL="postgresql://...supabase/neon..." \
  LIVEKIT_URL="wss://<project>.livekit.cloud" \
  LIVEKIT_API_KEY="..." LIVEKIT_API_SECRET="..." \
  OPENAI_API_KEY="sk-..."
# Edit ALLOWED_ORIGINS in fly.toml to your Vercel domain, then:
fly deploy
fly scale count api=1 agent=1     # one machine per process group
```

The `api` process serves HTTP and on first boot auto-ingests the bundled PDF
(idempotent). The `agent` process is a worker with no inbound ports; keep it at
count ≥ 1 so voice can be answered. Both processes share the app's secrets.

### 5. Frontend (Vercel)

Import the repo in Vercel with **root directory = `client/`** (Next.js
auto-detected). Set env var:

```
NEXT_PUBLIC_RAG_API_URL = https://dog-breed-rag.fly.dev
```

Then redeploy. Voice works from the Vercel-hosted site because the browser
connects to LiveKit Cloud directly using a token minted by the API.

### Networking notes

- The API/agent reach Ollama privately at `dog-breed-rag-ollama.internal:11434`
  (Fly 6PN) — no public Ollama port. The machine must stay running for `.internal`
  to resolve; for scale-to-zero use a Flycast service (`.flycast`).
- Put both Fly apps in the **same region** and org for low-latency private
  networking.
- Split alternative: you can instead run the API and agent as two separate Fly
  apps (one `fly.toml` each) if you want to scale or deploy them independently.

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
