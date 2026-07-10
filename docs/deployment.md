# Deployment

The live setup:

- **Text chat (free)** — Vercel + Render + Neon + Gemini (chat) + Jina (embeddings).
- **Text chat (fine-tuned)** — Vercel + **Fly** (FastAPI self-hosting the fine-tuned
  **bge** embedder) + Neon + Gemini/Anthropic (chat). See
  [bge in production](#bge-in-production--self-hosted-api-on-fly) and
  [fine-tuning.md](fine-tuning.md).
- **Voice (always-on)** — a LiveKit agent worker on Fly.io (small cost) + LiveKit
  Cloud + OpenAI STT/TTS.
- **Local / self-hosted** — Docker Compose.

The key idea: **don't self-host the 8B model** (that's the expensive part). Set
`INFERENCE_PROVIDER=openai` and point at hosted free APIs, so the backend is a
lightweight web service that fits free tiers. Local dev still uses Ollama
(`INFERENCE_PROVIDER=ollama`, the default).

---

## Text chat — free cloud (Vercel + Render + Neon + Gemini + Jina)

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
connect. (Supabase also works — use its **Session pooler** URI on port 5432.)

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
(Neon), `ALLOWED_ORIGINS` (your Vercel URL), and (for voice) `LIVEKIT_URL`,
`LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`. Free service sleeps after ~15 min idle
(cold start ~50s); the connection pool revalidates on wake, and Neon wakes in ~1s.

### 5. Frontend on Vercel
Import the repo, **Root Directory = `client`**, set
`NEXT_PUBLIC_RAG_API_URL=https://<your-render-app>.onrender.com`, deploy. Then set
Render's `ALLOWED_ORIGINS` to your Vercel URL and redeploy.

---

## Voice — always-on agent worker on Fly.io

The browser connects to LiveKit Cloud (token minted by the Render API) fine on its
own, but voice stays "connecting" until an **agent worker** joins the room. The
worker is long-lived and connects *out* to LiveKit Cloud (no inbound ports), so it
needs an always-on host — Render's free tier sleeps, so we use Fly.io (~$2–3/mo).
Config: [`fly.agent.toml`](../server/fly.agent.toml).

```bash
cd server
fly apps create dog-breed-rag-agent
fly secrets set -a dog-breed-rag-agent \
  INFERENCE_API_KEY="<gemini>" \
  INFERENCE_EMBEDDING_API_KEY="<jina>" \
  DATABASE_URL="<neon>" \
  LIVEKIT_API_KEY="<lk-key>" LIVEKIT_API_SECRET="<lk-secret>" \
  OPENAI_API_KEY="<openai>"
fly deploy -c fly.agent.toml
fly scale count 1 -c fly.agent.toml     # keep exactly one running, always on
```

Notes:
- Non-secret env (provider URLs, models, `LIVEKIT_URL`) is in `fly.agent.toml`.
- It reuses the **same Neon DB** as the API, so voice answers from the same data.
- Speech (OpenAI STT/TTS) is usage-billed — a few cents per conversation.
- If the machine OOMs loading the Silero VAD, bump RAM: `fly scale memory 1024 -c fly.agent.toml`.
- Watch logs: `fly logs -a dog-breed-rag-agent`.
- **Cheaper alternative:** run the worker locally during a demo
  (`uv run python -m src.livekit_agent dev`) — it dials out to LiveKit Cloud and
  auto-joins rooms; $0 hosting, only STT/TTS usage.

---

## bge in production — self-hosted API on Fly

The free stack embeds with Jina's **API** (near-zero RAM). To serve the **fine-tuned
bge** model instead, the API must embed **in-process** — which means shipping torch +
the model and running on a real machine. This trades the free tier for a fast,
cold-start-free demo that keeps the retrieval win (recall@5 0.80 → 0.84) live.
Config: [`fly.toml`](../server/fly.toml). Same image serves the voice agent
([`fly.agent.toml`](../server/fly.agent.toml)).

**The one hard rule:** one Neon DB holds one vector space. The API, the voice agent,
and the Neon corpus must **all** be bge (or all Jina) — never mixed, or retrieval
silently breaks. The `ST_MODEL_PATH` env var is the single switch: set it, and
`EmbeddingGenerator` uses the local bge model for all embeddings (chat path untouched).

### What ships in the image
- `Dockerfile` bakes in **CPU-only** torch (`--index-url .../whl/cpu`, avoids the
  multi-GB CUDA build), `sentence-transformers`, and the model at
  `finetune/models/bge-base-dogbreeds` (~440 MB). `HF_HUB_OFFLINE=1` so it never
  calls HuggingFace at runtime. Image ≈ 1.5–2 GB.
- The model is **not in git** (gitignored) — `fly deploy` uploads it from your local
  build context, so it must exist at `server/finetune/models/bge-base-dogbreeds/`
  (produced by [`finetune/train.py`](../server/finetune/train.py)).

### Cutover (order matters — minimizes the broken window)

```bash
cd server

# 1. Create + configure the API app (secrets, not committed)
fly launch --no-deploy --copy-config --name dog-breed-rag-api
fly secrets set -a dog-breed-rag-api \
  DATABASE_URL="<neon>" \
  INFERENCE_API_KEY="<chat key: Gemini or Anthropic>" \
  ALLOWED_ORIGINS="https://<your-vercel-app>.vercel.app"

# 2. Deploy the API (builds torch+model image). /health passes even before the DB
#    is bge — nothing points at it yet, so retrieval mismatch is invisible for now.
fly deploy
fly scale memory 2048

# 3. Flip Neon to bge (run LOCALLY — fast, and avoids the Fly box re-embedding on
#    cold start). This is when the OLD Render/Jina site goes stale — cut over promptly.
ST_MODEL_PATH=finetune/models/bge-base-dogbreeds \
  DATABASE_URL="<neon>" uv run python -m src.ingest --force

# 4. Point the frontend at the Fly API and redeploy Vercel
#    NEXT_PUBLIC_RAG_API_URL=https://dog-breed-rag-api.fly.dev

# 5. Switch the voice agent to bge too (shares Neon). Bump RAM if it OOMs.
fly deploy -c fly.agent.toml
fly scale memory 3072 -c fly.agent.toml    # bge + Silero VAD; 2 GB can be tight
```

### Verify
```bash
curl -s https://dog-breed-rag-api.fly.dev/health
curl -s https://dog-breed-rag-api.fly.dev/query \
  -H 'content-type: application/json' -d '{"query":"Where is the Akita from?"}'
```
Expect the Akita chunk retrieved and a grounded answer. `fly logs -a dog-breed-rag-api`
shows JSON access lines with `request_id` (structured logging is on via `LOG_JSON=true`).

### Cost / trade-off
An always-on 2 GB `shared-cpu-1x` machine is ~$5–10/mo (off the free tier). Worth it
for a reliable live demo; if cost matters more than serving bge in the cloud, keep
Jina-in-prod and run bge locally (see [fine-tuning.md](fine-tuning.md)) — the eval
numbers tell the story without hosting the model.

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

---

## Pre-prod checklist

- [ ] `ALLOWED_ORIGINS` set to your Vercel domain (not `*`).
- [ ] All secrets via the platform (Render dashboard / `fly secrets`) — never in
      git or `NEXT_PUBLIC_*`.
- [ ] Add auth to `/api/voice/session` (token-mint abuse) and disable/restrict
      `/ingest` (reads arbitrary server paths) — see [api-reference.md](api-reference.md).
- [ ] Book ingested once into Neon (so Render/Fly don't re-embed on cold start).
- [ ] Voice agent kept at `count 1` on Fly so it can always answer.
