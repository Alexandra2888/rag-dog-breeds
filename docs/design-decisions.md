# Design Decisions & Interview Talking Points

A guided tour of the non-obvious engineering decisions — the problem, the
options, what was chosen, and the tradeoff. Use this to present the project and
to answer "why did you build it that way?"

## 30-second pitch

> A local-first RAG assistant that answers dog-breed questions by text or voice,
> grounded in one book. The interesting work is in **retrieval quality**: I get
> near-perfect per-breed recall by chunking the book one-breed-per-chunk and
> fusing four retrieval signals (vectors, full-text, fuzzy trigrams, and a
> breed-label match) with Reciprocal Rank Fusion — and I **fine-tuned the
> embedding model itself** on synthetic in-domain pairs, which is where the
> biggest measured retrieval gains came from. It has a shared cross-process
> answer cache and a Ragas eval suite, and self-hosts the fine-tuned embedder in
> production so there's no hosted-embedding API cost.

## Decision 1 — Breed-aware chunking over fixed-size chunking

**Problem.** Fixed-size chunks split a breed across boundaries and mix two breeds
in one chunk, so a chunk's embedding represents a blur. Retrieval for a specific
breed suffers.

**Insight.** The corpus is structured: every breed entry has a stats **info box**
(Origin / Weight / Height / Life span). I detect a breed heading as a short
ALL-CAPS line followed within 40 lines by ≥2 of those field labels.

**Why this and not the obvious heuristic.** My first detector required an
ALL-CAPS "tagline" after the name. It silently missed ~130 *compact* breeds whose
entries have no tagline (Pharaoh Hound, Schnauzer…), merging each into the
previous breed. The info box is present in **both** entry formats and absent from
care/reference sections, so it both finds more breeds and rejects non-breeds.

**Result.** 283 → **390 distinct breeds**; false-positive "breeds" (`PELLETS`,
`INHERITED DISORDERS`, registry tags) eliminated.

**Tradeoff.** Heuristic and corpus-specific. I added an entry-size cap (5000
chars) so the last breed doesn't swallow the back-matter, re-chunking the
overflow generically — keeping it retrievable but unlabeled.

## Decision 2 — Hybrid retrieval with RRF, including a breed-label signal

**Problem.** Pure vector search blurs proper nouns — different breeds embed
close together, and a misspelled/mis-transcribed breed name embeds to noise.

**Approach.** Fuse four ranked lists with Reciprocal Rank Fusion:
vectors (semantics), full-text (exact names), trigram similarity (typos/STT), and
a **breed-label** match against each chunk's `breed` metadata.

**The key realization.** Trigram and full-text score *every* chunk mentioning
"schnauzer" identically, so the breed's own entry ties with incidental mentions
in care text. Only breed chunks carry a `breed` label, so a high-weight lane that
fuzzy-matches the query against that label pins the actual entry to the top — even
for misspellings. I also match the **whole breed phrase**, because the bare token
"terrier" scores identically across every terrier breed.

**Result.** Retrieval went from top-1 19/40 to **39/40**, top-5 32/40 → **40/40**;
"Schnouzer", "Daschund", "weimeraner" all resolve to rank 1.

**Why RRF.** It fuses rankings without needing the scores to be on comparable
scales, and weights are easy to reason about (`weight/(k+rank)`).

## Decision 3 — Fine-tuning the embedding model on synthetic in-domain pairs

**Problem.** After chunking and hybrid retrieval, the remaining weak link was the
**embedding model itself**. A general-purpose embedder (nomic-embed-text, then
off-the-shelf `bge-base-en-v1.5`) doesn't know that "hypoallergenic lap dog" or
"good with kids apartment breed" should pull the *right* breed entry — it embeds
generic web-text semantics, not this book's domain.

**Approach.** I built a small fine-tuning pipeline (`server/finetune/`):
1. **Generate synthetic data** — a local LLM writes realistic user questions for
   each passage, producing ~**1,193 query→passage pairs** entirely offline (no
   labeling cost, no data leaving the machine).
2. **Train** `bge-base-en-v1.5` with sentence-transformers using
   **MultipleNegativesRankingLoss** (in-batch negatives — every other passage in
   the batch is a negative, so one positive pair yields many contrastive
   signals). Kept at **768-dim** so the pgvector schema is unchanged.
3. **Evaluate** on a **held-out, judge-free** split — pure retrieval metrics
   (recall@k, MRR), no LLM judge to bias the result.

**Result (held-out).** recall@5 **0.795 → 0.839**, recall@3 0.728 → 0.789,
MRR 0.668 → 0.719 versus off-the-shelf bge. And the model choice alone mattered
even more: nomic-embed-text scored recall@5 **0.322** vs bge's 0.795 — a reminder
that picking the right base model dwarfs most prompt-level tweaking. Everything is
tracked in **MLflow** so runs are comparable. See
[fine-tuning](fine-tuning.md).

**Why this is the strongest story.** It's the full loop: identify the real
bottleneck with measurement, generate the data to fix it, train a small model,
and prove a *held-out, judge-free* win — not a vibes improvement.

**Then I self-hosted it in production**, which is where the interesting tradeoffs
live:
- **Torch baked into the image.** A local sentence-transformers model means
  shipping PyTorch in the container — a much larger image and a **~2 GB
  always-on** memory floor on Fly.io (can't scale to zero like a stateless API).
- **Shipping a gitignored artifact.** The trained weights aren't in git; the
  build/deploy has to pull the model into the image out-of-band, so the deploy
  pipeline owns an artifact the repo doesn't.
- **One Neon, one vector space.** Embeddings *must* be produced by the same model
  that ingested the corpus, so switching to the fine-tuned bge required a one-time
  **re-ingest** into the single Neon database. There's exactly one live vector
  space at a time — you can't mix embedders against one index.
- **Clean override.** All of this is gated by one env var: when `ST_MODEL_PATH`
  is set, `EmbeddingGenerator` loads the local model for **all** embeddings,
  overriding the ollama/openai providers; the chat path is untouched.

## Decision 4 — Shared answer cache in Postgres

**Problem.** Repeated questions re-run the whole pipeline (~30s on a local LLM).
An in-memory cache wouldn't help because **text (FastAPI) and voice (LiveKit
worker) are separate processes**.

**Choice.** A Postgres-backed cache keyed by `(normalized_query, mode, top_k)`,
checked before any embedding/search/LLM work. Invalidated whenever the corpus
changes (ingest/delete).

**Result.** ~**9000× faster** on a hit (29.6s → 0.003s), zero LLM calls, shared
across both processes and persistent across restarts.

**Tradeoff.** Exact-normalized matching (predictable, no wrong answers).
Semantic caching would raise the hit rate but risks serving a near-but-wrong
answer; left as a documented extension since the query embedding is already on
hand.

## Decision 5 — Unify voice answers through the cached pipeline

**Problem.** The voice LLM runs inside LiveKit's pipeline, so its answers weren't
cached and couldn't be reused by text.

**Choice.** On each turn, the agent calls the same cached `RAGService.query`,
speaks the result via `session.say()`, and raises `StopResponse` to skip the LLM
turn. Any failure falls back to the original "inject context, let the LLM reply"
path, so the user is never left silent.

**Tradeoff.** Voice loses token-streaming from the LLM, but answers are short and
now share the text cache. Verified the LiveKit mechanism (`say` + `StopResponse`)
against the library; the live mic path still needs a real-room test.

## Decision 6 — Eval framework + a deterministic anchor

**Problem.** How do you know retrieval/answers are actually good, and that changes
don't regress?

**Choice.** Ragas (RAG-canonical metrics: context precision/recall, faithfulness,
answer relevancy, factual correctness) **plus** a deterministic `breed_retrieved`
check. The judge runs on local Ollama, so evals cost no tokens. The runner exits
non-zero below thresholds (CI gate).

**Why both.** The corpus has objective facts (origins), so a deterministic check
is more trustworthy than an LLM judge alone — it anchors the noisier metric. And
an LLM judge is itself biased/non-deterministic, so it's pinned to `temperature=0`
and treated as a regression guard, not absolute truth.

**War story worth telling.** Ragas 0.4.3 hard-imports langchain 0.3.x paths;
`uv` had resolved langchain 1.x, so `import ragas` failed. Fix: pin the langchain
ecosystem to `<1.0`. Good example of real dependency-resolution debugging.

## Decision 7 — Pluggable inference provider (local-first, cloud-flexible)

**Problem.** Local Ollama is great for development (free, private, offline), but
hosting an 8B model 24/7 in the cloud isn't free — and Fly.io (the first target)
has no free tier. How do you keep the local-first dev story *and* deploy for $0?

**Choice.** A thin provider abstraction resolved by priority — chat and
embeddings are independently configurable:
1. **`ST_MODEL_PATH`** (current prod default) → the fine-tuned local bge for
   **all embeddings** (Decision 3), overriding everything below for the embedding
   path; chat is unaffected.
2. **`INFERENCE_PROVIDER=openai`** → any OpenAI-compatible cloud API — Jina for
   embeddings, Gemini or OpenAI for chat.
3. **`ollama`** → fully local (nomic-embed-text + llama3.1:8b) for offline dev.

Same code, switched by env; every embedding option is **768-dim** so the DB
schema never changes.

**Current production stack.** Vercel (frontend) → **Fly.io** FastAPI
self-hosting the fine-tuned bge (torch in the image, 2 GB always-on) → **Neon**
(Postgres + pgvector, re-ingested with bge) → **OpenAI `gpt-4o-mini`** for chat;
voice is a **LiveKit agent on Fly** + LiveKit Cloud + OpenAI STT/TTS.

**Chat moved to OpenAI `gpt-4o-mini`.** In the earlier free deploy chat ran on
Gemini `gemini-2.5-flash`, but the Gemini *embedding* free tier was too limited
(below), and consolidating chat + STT/TTS on OpenAI kept the prod surface simple
and reliable. `gpt-4o-mini` is cheap enough that per-query chat cost is
negligible.

**Earlier / low-cost alternative stack (kept as an option).** Vercel + **Render**
(FastAPI, free tier) + Neon + **Gemini** (chat) + **Jina** (embeddings,
`jina-embeddings-v2-base-en`, 768-dim). This was the project's first fully-free
deploy and still works via env; the fine-tuned self-hosted stack above superseded
it as *the* production deployment. War stories from that evolution:
- Free-tier model availability shifts: `gemini-2.0-flash` had **zero** free quota
  and `text-embedding-004` was retired, so I discovered the live model list and
  switched to `gemini-2.5-flash` / `gemini-embedding-001`.
- Gemini's free *embedding* tier has a **1000/day** cap — too tight for a
  768-chunk ingest plus live query traffic. So embeddings first moved to **Jina**
  (generous free tier, 768-dim), and ultimately to the **fine-tuned self-hosted
  bge**, which removed the hosted-embedding dependency entirely. Good example of
  letting a real constraint drive an abstraction boundary.
- Switching embedding models changes the vector space → a one-time re-ingest each
  time (nomic → Jina → fine-tuned bge).
- Voice can't be fully free (paid STT/TTS + an always-on worker), so the free
  deploy shipped text first.

## Decision 8 — Structured logging for observability

**Problem.** A multi-process system (FastAPI + LiveKit worker) with a cache and
several fallback paths is hard to reason about from `print`-style logs — you can't
follow a single request across the pipeline.

**Choice.** **structlog** with a per-request **`request_id`** bound into the
context, so every log line for a query (embed → hybrid search → cache
hit/miss → LLM → answer) is correlated and machine-parseable (JSON in prod).
Cheap to add, and it makes "was this a cache hit or a fallback?" answerable from
the logs alone. See [observability](observability.md).

## Things I'd do next (shows awareness, not gaps)

- **Security hardening** for non-local deploys: lock down CORS, add auth
  (especially LiveKit token minting), disable/path-restrict path-based ingest.
- **Semantic cache** with a high-similarity threshold to catch paraphrases.
- **Cache TTS audio** to also cut speech cost on repeated voice answers.
- **Grow the eval set** (size/lifespan/temperament questions, more refusal cases)
  and gate CI on it; consider a stronger judge model for fidelity.
- **Two-column extraction**: the PDF interleaves adjacent columns, occasionally
  mixing a neighbor's field into a breed — a layout-aware extractor would fix the
  residual noise.

## Likely interview questions

- *Why not just vector search?* → proper-noun blur; see Decision 2.
- *How do you handle misspellings / STT errors?* → trigram + breed-label lanes,
  phrase matching, capitalized-token extraction.
- *How do you keep answers grounded / avoid hallucination?* → retrieved context
  only, "say you don't know" prompting, `faithfulness` metric, an out-of-scope
  eval question.
- *How do you improve retrieval beyond chunking?* → Decision 3; fine-tuned the
  embedder on synthetic pairs, held-out recall@5 0.80 → 0.84 (and nomic → bge
  0.32 → 0.80).
- *How do you measure quality?* → Decision 6; concrete before/after numbers.
- *Why local models?* → cost, privacy, offline; tradeoff is a noisier judge and
  more latency, mitigated by the cache.
- *How does the cache stay correct?* → invalidation on corpus change; exact-
  normalized keys; per-mode namespaces.
