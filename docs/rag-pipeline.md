# RAG Pipeline

End to end: **ingestion → chunking → embeddings → hybrid retrieval → generation**.
Code lives in `server/src/`.

## 1. Ingestion (`ingest.py`, `pdf_processor.py`)

PDFs in `server/data/*.pdf` are ingested on API startup (idempotent — files
already in the `documents` table are skipped) or manually with
`uv run python -m src.ingest` (`--force` to re-ingest).

Steps per PDF:
1. Extract text per page (`pypdf`), recording each page's character offset so
   chunks can be tagged with a page number.
2. Concatenate pages into one stream and chunk it (below).
3. Embed every chunk and bulk-insert into `chunks` with its metadata.

## 2. Breed-aware chunking (`pdf_processor.py`)

The goal: **one chunk per breed**, so a chunk's embedding represents a single
breed rather than a blur of several.

### Heading detection — the info box

Every breed entry in the book has a stats **info box** with the labels
`Origin`, `Weight range`, `Height range`, `Life span`. A line is treated as a
breed heading when:

- it is a short ALL-CAPS line (3–40 chars, 1–5 words, no digits), **and**
- at least **2 of the info-box field labels appear within the next 40 lines**.

This is robust to the book's two entry formats — *featured* breeds (NAME →
ALL-CAPS tagline → prose → box) and *compact* breeds (NAME → box directly, no
tagline). An earlier heuristic that required an ALL-CAPS tagline missed ~130
compact breeds (e.g. Pharaoh Hound, Schnauzer), merging them into a neighbor.
The info box is also absent from care/reference headers (`PELLETS`,
`INHERITED DISORDERS`), so those are correctly rejected. Registry tags
(`KC`, `FCI`, `AKC`, …) and plural group headers (`SCENT HOUNDS`) are blocklisted.

Result on the bundled book: **390 distinct breeds**, no swallow blobs.

### Slicing and the entry-size cap

The document is sliced between consecutive headings. A real breed entry is at
most a page or two, so each entry is capped at `MAX_ENTRY_CHARS = 5000`. Content
beyond the cap (e.g. the last breed would otherwise swallow the entire
back-matter) is re-chunked **generically without a breed label** — still
retrievable, but not mis-attributed. Chunks shorter than `MIN_CHUNK_CHARS = 80`
are dropped.

### Fallback size chunking

If a document doesn't look like a breed catalogue (<20 headings detected), it
falls back to size-based chunking (`CHUNK_SIZE`/`CHUNK_OVERLAP`, default
1000/200) that breaks on sentence boundaries.

### Chunk metadata

`{ source, total_pages, breed?, page_number, char_start, chunk_index }`.
The `breed` key is present only on breed-attributed chunks and powers the
breed-label retrieval signal below.

## 3. Embeddings (`embeddings.py`)

**Provider is pluggable** via `INFERENCE_PROVIDER`: `ollama` (local dev) or
`openai` (any OpenAI-compatible API). The free cloud deploy uses **Jina**
(`jina-embeddings-v2-base-en`) for embeddings — configured separately from chat
via `INFERENCE_EMBEDDING_*`. All options yield **768-dim** vectors so the DB schema
is unchanged. The `openai` path batches inputs with retry/backoff. See
[design-decisions.md](design-decisions.md#decision-6--pluggable-inference-provider-local-first-free-to-deploy).

Local model: `nomic-embed-text` (768-dim). nomic is trained with **asymmetric task
prefixes**, so the code prepends:

- `search_document: …` for stored chunks
- `search_query: …` for user queries

Skipping these mismatches the query/document vector spaces and tanks recall, so
it's applied automatically for nomic models.

## 4. Hybrid retrieval (`database.py::similarity_search`)

When a query string is supplied, retrieval fuses **four ranked lists** with
**Reciprocal Rank Fusion (RRF)**. Each list contributes `weight / (k + rank)`
(`k = 10`); the per-modality candidate pool is `max(top_k*5, 40)`.

| Lane | Signal | Weight | Catches |
|---|---|---|---|
| `vec` | cosine distance on embeddings | 1.0 | semantics / paraphrase |
| `fts` | Postgres full-text (terms OR'd) | 2.0 | exact breed names / proper nouns |
| `trgm` | `pg_trgm` word similarity on content | 1.2 | misspellings, STT errors |
| `brd` | `pg_trgm` similarity on the chunk's **breed label** | 3.0 | pins a breed's own entry above passing mentions |

### Fuzzy term selection

The fuzzy/breed lanes match a **phrase** built from the query's capitalized
tokens (breed names are proper nouns; users and the STT transcriber capitalize
them), falling back to the longest word for all-lowercase queries. Matching the
whole phrase (`"border terrier"`) discriminates far better than matching tokens
independently (the bare token `terrier` scores identically against every terrier
breed).

### Why the breed-label lane matters

Trigram/full-text score *every* chunk that mentions "schnauzer" identically, so
the breed's own entry ties with incidental mentions in care text. Only breed
chunks carry a `breed` label, so the `brd` lane (highest weight) lifts the actual
entry to the top — including for misspellings (`"Schnouzer"` → `SCHNAUZER`).

Pure vector search (no query text) is used as a fallback and supports an optional
similarity `threshold`.

### Measured quality

On the eval golden set: retrieval **top-1 39/40, top-5 40/40**; misspelled breed
names ("Schnouzer", "Daschund", "weimeraner") resolve to rank 1.

## 5. Answer generation (`rag_service.py`)

Retrieved chunks are formatted with source/page tags into a context block and
sent to the chat model — local Ollama (`llama3.1:8b`) or the hosted provider
(`gemini-2.5-flash`) depending on `INFERENCE_PROVIDER`. Two prompt styles by `mode`:

- `text` — a full answer; says so when the answer isn't in the context.
- `voice` — one or two short, spoken-friendly sentences.

Answers are cached keyed by `(normalized_query, mode, top_k)` — see
[caching.md](caching.md). The cache is checked **before** embedding/search, so an
exact repeat costs zero Ollama calls.
