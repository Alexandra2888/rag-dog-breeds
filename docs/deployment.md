# Deployment

Three targets:

- **Free cloud (recommended)** — Vercel + Render + Neon + Gemini (below).
- **Local / self-hosted** — Docker Compose.
- **Paid cloud** — Fly.io with self-hosted Ollama (for fully-local inference).

The key idea: **don't self-host the 8B model** (that's the expensive part). Set
`INFERENCE_PROVIDER=openai` and point at a free OpenAI-compatible API (Google
Gemini), so the backend is a lightweight web service that fits free tiers. Local
dev still uses Ollama (`INFERENCE_PROVIDER=ollama`, the default).

---

## Free cloud: Vercel + Render + Neon + Gemini

```
Vercel (Next.js)  ──►  Render (FastAPI, free)  ──►  Neon Postgres (pgvector)
                                  ├──►  Gemini  (chat — gemini-2.5-flash)
                                  └──►  Jina    (embeddings — 768-dim)
```

### 1. API keys (two free keys)
- **Gemini** (chat) — <https://aistudio.google.com/apikey>. Model `gemini-2.5-flash`.
- **Jina** (embeddings) — <https://jina.ai/embeddings>. Model
  `jina-embeddings-v2-base-en` (fixed **768-dim**, matches the DB schema).

**Why two providers:** Gemini's free *embedding* tier is capped at **1000/day** —
too tight for a 768-chunk ingest plus live query traffic. Jina's free tier is far
more generous, so embeddings run on Jina while chat stays on Gemini. The split is
config-driven: `INFERENCE_*` for chat, `INFERENCE_EMBEDDING_*` for embeddings
(the latter falls back to the former if unset).

### 2. Neon database
Create a free project at <https://neon.tech> and copy the connection string. The
**pooled** endpoint works (verified — `CREATE EXTENSION` succeeds on it); the
**direct** host (same string without `-pooler`) also works. It already includes
`?sslmode=require`. `pgvector` and `pg_trgm` are created automatically on first
connect.
> (Supabase also works if you have a free slot — use its **Session pooler** URI,
> port 5432, not the Transaction pooler on 6543.)

### 3. Ingest the book once (into Neon)
Put the prod values in `server/.env.prod` (gitignored) and run locally so Render
doesn't re-embed on every cold start. With Jina this takes ~1 min for ~745 chunks:
```bash
cd server
INFERENCE_PROVIDER=openai \
INFERENCE_API_KEY="<gemini-key>" \
INFERENCE_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/" \
INFERENCE_EMBEDDING_BASE_URL="https://api.jina.ai/v1" \
INFERENCE_EMBEDDING_API_KEY="<jina-key>" \
INFERENCE_EMBEDDING_MODEL="jina-embeddings-v2-base-en" \
INFERENCE_EMBEDDING_DIM=0 \
DATABASE_URL="<neon-connection-url>" \
uv run python -m src.ingest --force
```

### 4. Backend on Render (free)
Push to GitHub, then Render → **New + → Blueprint** and pick the repo (uses
[`render.yaml`](../render.yaml)). Set the `sync: false` secrets in the dashboard:
`INFERENCE_API_KEY` (Gemini), `INFERENCE_EMBEDDING_API_KEY` (Jina), `DATABASE_URL`
(Neon), `ALLOWED_ORIGINS` (your Vercel URL), and optionally `LIVEKIT_URL`,
`LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` (voice). Free service sleeps after ~15 min
idle (cold start ~50s).

### 5. Frontend on Vercel
Import the repo, **root directory = `client/`**, set
`NEXT_PUBLIC_RAG_API_URL=https://<your-render-app>.onrender.com`, deploy.

### Voice (near-free, not 100%)
Voice's LLM now uses Gemini too, but it still needs: a LiveKit Cloud project
(free tier), **paid** STT/TTS (OpenAI, or Groq Whisper for cheaper STT), and an
**always-on agent worker** — Render's free tier only runs (sleeping) web
services, so the worker needs a small paid host (Render background worker ~$7/mo)
or a free-with-idle-sleep host (Hugging Face Space). Deploy text first; add voice
when you've picked a worker host.

---

## Paid cloud: Vercel + Fly.io (self-hosted Ollama)

Use this only if you want inference to stay fully local (no third-party model
API). It requires a Fly GPU/CPU machine for Ollama — not free.

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
