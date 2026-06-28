# Documentation

Comprehensive docs for the Dog Breed RAG Assistant. New here? Read in this order.

| Doc | What it covers |
|---|---|
| [architecture.md](architecture.md) | System components, processes, and end-to-end data flow for both text and voice |
| [rag-pipeline.md](rag-pipeline.md) | Ingestion → breed-aware chunking → embeddings → hybrid retrieval → answer generation |
| [design-decisions.md](design-decisions.md) | **Why** it's built this way — tradeoffs, metrics, and interview talking points |
| [api-reference.md](api-reference.md) | Every FastAPI endpoint with request/response shapes |
| [caching.md](caching.md) | The shared Postgres answer cache: keys, invalidation, behavior in text and voice |
| [evaluation.md](evaluation.md) | The Ragas eval suite: metrics, golden dataset, running, extending |
| [configuration.md](configuration.md) | All environment variables and their defaults |
| [development.md](development.md) | Local setup, running each service, the voice console, troubleshooting |
| [deployment.md](deployment.md) | Docker Compose stack and production notes |

> Presenting this in an interview? Start with [design-decisions.md](design-decisions.md).

## TL;DR of how it works

1. A PDF dog-breed book is **ingested**: split into one chunk per breed (detected
   via the stats info box), embedded with `nomic-embed-text`, stored in pgvector.
2. A question is answered by **hybrid retrieval** (vectors + full-text + fuzzy
   trigrams + breed-label match, fused with RRF) feeding a local LLM.
3. The same pipeline serves **text** (FastAPI `/query`) and **voice** (LiveKit
   agent), and both share a **Postgres answer cache** so repeats cost nothing.
4. Quality is tracked with a **Ragas** eval suite scored by a local Ollama judge.

See [architecture.md](architecture.md) for the full picture.
