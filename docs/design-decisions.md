# Design Decisions & Interview Talking Points

A guided tour of the non-obvious engineering decisions — the problem, the
options, what was chosen, and the tradeoff. Use this to present the project and
to answer "why did you build it that way?"

## 30-second pitch

> A local-first RAG assistant that answers dog-breed questions by text or voice,
> grounded in one book. The interesting work is in **retrieval quality**: I get
> near-perfect per-breed recall by chunking the book one-breed-per-chunk and
> fusing four retrieval signals (vectors, full-text, fuzzy trigrams, and a
> breed-label match) with Reciprocal Rank Fusion. It has a shared cross-process
> answer cache and a Ragas eval suite, both running on local models so there's no
> per-query API cost.

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

## Decision 3 — Shared answer cache in Postgres

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

## Decision 4 — Unify voice answers through the cached pipeline

**Problem.** The voice LLM runs inside LiveKit's pipeline, so its answers weren't
cached and couldn't be reused by text.

**Choice.** On each turn, the agent calls the same cached `RAGService.query`,
speaks the result via `session.say()`, and raises `StopResponse` to skip the LLM
turn. Any failure falls back to the original "inject context, let the LLM reply"
path, so the user is never left silent.

**Tradeoff.** Voice loses token-streaming from the LLM, but answers are short and
now share the text cache. Verified the LiveKit mechanism (`say` + `StopResponse`)
against the library; the live mic path still needs a real-room test.

## Decision 5 — Eval framework + a deterministic anchor

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

## Decision 6 — Pluggable inference provider (local-first, free to deploy)

**Problem.** Local Ollama is great for development (free, private, offline), but
hosting an 8B model 24/7 in the cloud isn't free — and Fly.io (the first target)
has no free tier. How do you keep the local-first dev story *and* deploy for $0?

**Choice.** A thin provider abstraction (`INFERENCE_PROVIDER`): `ollama` for local
dev, or `openai` for **any** OpenAI-compatible API in the cloud. The free deploy
points it at **Google Gemini's** OpenAI-compatible endpoint — one free key serves
both chat (`gemini-2.5-flash`) and embeddings (`gemini-embedding-001`). Same code,
switched by env; embeddings requested at **768-dim** so the DB schema is unchanged.

**Free stack.** Vercel (frontend) + Render (FastAPI, free) + Neon (Postgres +
pgvector, free) + Gemini (LLM + embeddings, free). The expensive part — the model —
is never self-hosted.

**Tradeoffs / war stories worth telling.**
- Free-tier model availability shifts: `gemini-2.0-flash` had **zero** free quota
  and `text-embedding-004` was retired, so I discovered the live model list and
  switched to `gemini-2.5-flash` / `gemini-embedding-001`.
- Gemini embeddings are capped at ~**100 items/min** free, and the limit counts
  *items*, not requests — so batching doesn't dodge it. The bulk ingest is
  throttled (95/batch + 61s pause, with retry/backoff); query-time embedding (one
  per question) is nowhere near the limit.
- Switching embedding models changes the vector space → a one-time re-ingest.
- Voice can't be fully free (paid STT/TTS + an always-on worker), so the free
  deploy ships text first; voice's LLM also moved to Gemini.

The Fly.io path is kept in the repo as the documented **paid alternative** for
anyone who wants fully-local inference (no third-party model API).

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
- *How do you measure quality?* → Decision 5; concrete before/after numbers.
- *Why local models?* → cost, privacy, offline; tradeoff is a noisier judge and
  more latency, mitigated by the cache.
- *How does the cache stay correct?* → invalidation on corpus change; exact-
  normalized keys; per-mode namespaces.
