# Dog Breed RAG — Backend

FastAPI service that powers the [Dog Breed RAG Assistant](../README.md): PDF
ingestion, hybrid retrieval, answer generation, a shared answer cache, a LiveKit
voice agent, an embedding-model fine-tuning pipeline, and a Ragas eval suite.
Embeddings and chat are **pluggable**: a **fine-tuned local `bge` model** (the
production default, via `ST_MODEL_PATH`), local **Ollama**, or cloud
(Jina embeddings / OpenAI · Gemini chat). Storage is **Postgres + pgvector**.

```
PDF → breed-aware chunking → embeddings (fine-tuned bge / Ollama / Jina) → pgvector
                                                     │
question → hybrid retrieval (vector + FTS + trigram + breed-label, RRF) → LLM (Ollama / OpenAI) → answer
                                                     │
                                          shared answer cache (Postgres)
                                                     │
                              text: FastAPI /query   ·   voice: LiveKit agent
```

Provider priority: **`ST_MODEL_PATH`** (fine-tuned bge, current prod) >
**`INFERENCE_PROVIDER=openai`** (Jina embeddings / OpenAI · Gemini chat) >
**`ollama`** (nomic + llama, local). All embedding options are 768-dim.
Production runs the fine-tuned bge self-hosted on **Fly.io** → **Neon** pgvector,
with **OpenAI `gpt-4o-mini`** for chat. See
[fine-tuning](../docs/fine-tuning.md) and [design-decisions](../docs/design-decisions.md).

For the full design (and **why**), see [`../docs/`](../docs/README.md) —
especially [rag-pipeline](../docs/rag-pipeline.md) and
[design-decisions](../docs/design-decisions.md).

## Prerequisites

- Python 3.11+ and [`uv`](https://github.com/astral-sh/uv)
- Docker (Postgres + pgvector)
- [Ollama](https://ollama.com) with `nomic-embed-text` and `llama3.1:8b` for the
  default local path (embeddings + chat, no API keys)
- Voice only: `OPENAI_API_KEY` (STT/TTS) and LiveKit credentials

```bash
ollama pull nomic-embed-text
ollama pull llama3.1:8b
```

To use the **fine-tuned `bge` embedder** instead (production default), set
`ST_MODEL_PATH=finetune/models/bge-base-dogbreeds` — a local sentence-transformers
model then handles all embeddings, overriding Ollama/Jina (chat is unaffected).
Train it via the [fine-tuning pipeline](#fine-tuning) or point at any local
sentence-transformers model. Cloud chat uses `INFERENCE_PROVIDER=openai` with
`OPENAI_API_KEY` (OpenAI `gpt-4o-mini` in prod).

## Setup

```bash
docker compose up -d postgres          # Postgres+pgvector on host port 5433
cp .env.example .env                   # edit as needed (see docs/configuration.md)
uv sync                                # install deps (incl. dev: ragas, langchain-ollama)
uv run uvicorn src.main:app --reload   # http://localhost:8000 · docs at /docs
```

The PDF in `data/` is **auto-ingested** on first startup (idempotent). Manual:

```bash
uv run python -m src.ingest            # ingest new PDFs in data/
uv run python -m src.ingest --force    # re-ingest (replaces existing)
```

## Endpoints (summary)

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/query` | RAG answer (+ cache, `cached` flag) |
| POST | `/search` | Hybrid retrieval, no answer |
| POST | `/ingest` · `/ingest/upload` | Ingest a PDF (path / upload) |
| GET | `/documents` | List ingested docs |
| DELETE | `/documents/{id}` | Delete a doc + chunks |
| DELETE | `/cache` | Clear the answer cache |
| POST | `/api/voice/session` | Mint a LiveKit room + token |
| DELETE | `/api/voice/session/{room}` | Tear down a voice room |

Full request/response shapes: [`../docs/api-reference.md`](../docs/api-reference.md).
Interactive: `/docs` (Swagger), `/redoc`.

## Voice agent

Built on **livekit-agents 1.x**. OpenAI handles STT (`gpt-4o-transcribe`) + TTS;
the LLM and RAG follow the same provider config as text (local Ollama in dev,
OpenAI `gpt-4o-mini` in prod). Each turn is answered through the **cached**
`RAGService.query`, spoken via `session.say()`, skipping the LLM on a cache hit
(with a safe fallback). Talk to it in your terminal:

```bash
uv run python -m src.livekit_agent download-files   # one-time
uv run python -m src.livekit_agent console          # speak via mic/speakers
uv run python -m src.livekit_agent dev              # run as a worker for a room/frontend
```

Requires `OPENAI_API_KEY` and LiveKit creds. See
[`../docs/architecture.md`](../docs/architecture.md) and
[`../docs/caching.md`](../docs/caching.md).

## Evaluation

[Ragas](https://docs.ragas.io) metrics + a deterministic retrieval check, judged
by local Ollama (no API cost):

```bash
uv run python -m evals.run_eval --limit 3   # quick smoke test
uv run python -m evals.run_eval             # full suite (slow on a local judge)
```

Plus an **adversarial / edge-case** suite — jailbreaks, prompt-injection,
out-of-scope, degenerate inputs — scored per category with a deterministic
refusal metric (`src/refusal.py`):

```bash
uv run python -m evals.run_adversarial              # full (clears cache first)
uv run python -m evals.run_adversarial --category jailbreak
```

Details: [`evals/README.md`](evals/README.md) and
[`../docs/evaluation.md`](../docs/evaluation.md).

## Online eval (production quality)

Every `/query` records quality signals on real traffic (retrieval-score stats +
deterministic refusal on all queries, a sampled reference-free LLM judge on a
fraction) via a background task — no added latency. A `POST /feedback` endpoint
captures thumbs up/down. A CLI rolls windows up into an `online-eval` MLflow
experiment so drift is a trend line:

```bash
uv run python -m scripts.aggregate_online_eval --window-hours 24
```

Config: `ONLINE_EVAL_ENABLED`, `ONLINE_JUDGE_ENABLED`, `ONLINE_JUDGE_SAMPLE_RATE`.
Details: [`../docs/observability.md`](../docs/observability.md).

## Fine-tuning

An offline pipeline (`finetune/`) fine-tunes `bge-base-en-v1.5` on synthetic
in-domain query→passage pairs (MultipleNegativesRankingLoss), tracked in MLflow.
Held-out judge-free win: recall@5 **0.795 → 0.839** vs off-the-shelf bge.

```bash
uv run python -m finetune.generate_pairs    # synth pairs from the corpus (local LLM)
uv run python -m finetune.eval_retrieval    # held-out recall@k / MRR baseline
uv run python -m finetune.train             # train → finetune/models/bge-base-dogbreeds
uv run mlflow ui                            # inspect runs at http://localhost:5000
```

Point the API at the result with `ST_MODEL_PATH` (then re-ingest). Details:
[`../docs/fine-tuning.md`](../docs/fine-tuning.md).

## Project structure

```
server/
├── docker-compose.yml      # Postgres + API + LiveKit server + agent
├── Dockerfile
├── pyproject.toml          # uv deps (runtime + dev: ragas, langchain-ollama)
├── data/                   # PDF knowledge base
├── evals/                  # eval suites: golden.jsonl + run_eval.py (normal),
│                           #   adversarial.jsonl + run_adversarial.py (edge cases)
├── finetune/               # embedding fine-tuning (generate_pairs, train, eval, MLflow)
│   └── models/             # trained model (bge-base-dogbreeds, gitignored)
├── scripts/                # aggregate_online_eval.py (online-eval window → MLflow)
└── src/
    ├── main.py             # FastAPI app + routes (/query, /feedback) + startup auto-ingest
    ├── config.py           # env settings (providers, ST_MODEL_PATH, online-eval)
    ├── logging_config.py   # structlog setup + per-request request_id
    ├── pdf_processor.py    # breed-aware chunking
    ├── embeddings.py       # pluggable embeddings (fine-tuned bge via ST_MODEL_PATH / Ollama / Jina)
    ├── database.py         # pgvector ops, hybrid search (RRF), answer cache, online_eval/feedback
    ├── rag_service.py      # retrieval + generation + cache orchestration
    ├── refusal.py          # deterministic abstention detector (shared: evals + online eval)
    ├── online_eval.py      # per-query production quality signals (background task)
    ├── models.py           # Pydantic request/response models
    ├── ingest.py           # auto-ingest CLI + startup hook
    ├── livekit_agent.py    # voice agent (cached RAG, STT/TTS, fallback)
    └── livekit_server.py   # agent worker entrypoint
```

## Configuration & troubleshooting

See [`../docs/configuration.md`](../docs/configuration.md) (env vars) and
[`../docs/development.md`](../docs/development.md) (troubleshooting). Key gotchas:

- `DATABASE_URL` uses host port **5433** (Docker maps `5433:5432`).
- DB schema is `vector(768)`; every embedder used (fine-tuned bge, nomic-embed-text,
  Jina) is 768-dim, so the schema is stable — but **switching embedders requires a
  re-ingest** (the vector space differs), and a different-dimension model would
  need a schema change.
- `import ragas` requires the pinned langchain 0.3.x line — run `uv sync`.

## License

MIT
