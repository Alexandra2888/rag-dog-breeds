# Documentation

Comprehensive docs for the Dog Breed RAG Assistant. New here? Read in this order.

| Doc | What it covers |
|---|---|
| [architecture.md](architecture.md) | System components, processes, and end-to-end data flow for both text and voice |
| [rag-pipeline.md](rag-pipeline.md) | Ingestion → breed-aware chunking → embeddings → hybrid retrieval → answer generation |
| [design-decisions.md](design-decisions.md) | **Why** it's built this way — tradeoffs, metrics, and interview talking points |
| [api-reference.md](api-reference.md) | Every FastAPI endpoint with request/response shapes |
| [caching.md](caching.md) | The shared Postgres answer cache: keys, invalidation, behavior in text and voice |
| [evaluation.md](evaluation.md) | Offline eval: Ragas metrics + golden dataset, plus the adversarial / edge-case suite with a deterministic refusal metric |
| [fine-tuning.md](fine-tuning.md) | Synthetic query–passage pairs → retriever fine-tuning (bge-base + MNRL) → recall@k/MRR, tracked in MLflow |
| [observability.md](observability.md) | Structured logging (structlog) + **online eval**: per-query quality signals on real traffic, feedback, and drift tracked in MLflow |
| [guardrails.md](guardrails.md) | Runtime input/output guardrails: prompt-injection, grounding+refusal, PII redaction, toxicity/off-topic — shadow-first, enforced |
| [configuration.md](configuration.md) | All environment variables and their defaults |
| [development.md](development.md) | Local setup, running each service, the voice console, troubleshooting |
| [deployment.md](deployment.md) | Docker Compose stack and production notes |

> Presenting this in an interview? Start with [design-decisions.md](design-decisions.md).

## TL;DR of how it works

1. A PDF dog-breed book is **ingested**: split into one chunk per breed (detected
   via the stats info box), embedded with a **fine-tuned `bge-base-en-v1.5`**
   retriever (768-dim), stored in pgvector. The fine-tune lifts recall@5
   **0.80 → 0.84** over off-the-shelf bge; embeddings are pluggable (local bge ·
   Jina/Gemini cloud · nomic on Ollama). See [fine-tuning.md](fine-tuning.md).
2. A question is answered by **hybrid retrieval** (vectors + full-text + fuzzy
   trigrams + breed-label match, fused with RRF) feeding an LLM (Ollama local
   or OpenAI-compatible).
3. The same pipeline serves **text** (FastAPI `/query`) and **voice** (LiveKit
   agent), and both share a **Postgres answer cache** so repeats cost nothing.
4. Quality is tracked three ways: a **Ragas** eval suite (local Ollama judge), an
   **adversarial** edge-case suite (jailbreaks / injection / out-of-scope, scored
   by a deterministic refusal metric), and **online eval** on real traffic whose
   drift is charted in MLflow. See [evaluation.md](evaluation.md) +
   [observability.md](observability.md).
5. **Runtime guardrails** then *enforce* what eval measures — input injection
   screening, output grounding+refusal, PII redaction, toxicity/off-topic — rolled
   out shadow-first. See [guardrails.md](guardrails.md).
5. It's **deployed live**: Vercel → Fly.io FastAPI (self-hosting the fine-tuned
   bge) → Neon Postgres/pgvector → OpenAI chat, with a LiveKit voice agent.

See [architecture.md](architecture.md) for the full picture.
