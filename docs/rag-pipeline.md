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

The retriever's **primary embedder is a fine-tuned model**, not an off-the-shelf
one. `EmbeddingGenerator` resolves its backend in priority order:

1. **`ST_MODEL_PATH` set → local fine-tuned sentence-transformers (current default).**
   When this env var points at a model (prod: `server/finetune/models/bge-base-dogbreeds`),
   the generator loads it directly with `sentence-transformers` and uses it for
   **all** embeddings — queries and stored chunks — overriding the provider below.
   The chat/generation path is unaffected.
2. **`INFERENCE_PROVIDER=openai` → Jina cloud** (`jina-embeddings-v2-base-en`),
   configured separately from chat via `INFERENCE_EMBEDDING_*`. Batches inputs
   with retry/backoff.
3. **`ollama` → local `nomic-embed-text`** (the original local-dev default).

All three yield **768-dim** vectors, so the DB schema (and existing embeddings)
are unchanged regardless of backend. See
[design-decisions.md](design-decisions.md#decision-6--pluggable-inference-provider-local-first-free-to-deploy).

### The fine-tuned model

`bge-base-en-v1.5` fine-tuned on ~1,193 synthetic query→passage pairs with
**MultipleNegativesRankingLoss**, saved at `server/finetune/models/bge-base-dogbreeds`.
On the held-out, judge-free eval it lifts **recall@5 from 0.795 → 0.839** vs
off-the-shelf bge, and far above nomic (0.322). How it was trained and the full
numbers are in [fine-tuning.md](fine-tuning.md).

### Task/instruction prefixes (`_apply_task_prefix`)

Both retriever models use **asymmetric** query/document conditioning, applied
automatically per model family:

- **bge** uses a **query-instruction prefix on the query side only** —
  `Represent this sentence for searching relevant passages: ` is prepended to
  queries, while passages are embedded **plain**.
- **nomic** uses paired task prefixes — `search_document: …` for stored chunks
  and `search_query: …` for user queries.

Skipping the right prefix mismatches the query/document vector spaces and tanks
recall, so `_apply_task_prefix` applies the correct scheme for the active model.

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
sent to the chat model — local Ollama (`llama3.1:8b`) or an OpenAI-compatible
hosted provider (prod: **OpenAI `gpt-4o-mini`**) depending on `INFERENCE_PROVIDER`.
Chat is independent of the embedding backend above. Two prompt styles by `mode`:

- `text` — a full answer; says so when the answer isn't in the context.
- `voice` — one or two short, spoken-friendly sentences.

Answers are cached keyed by `(normalized_query, mode, top_k)` — see
[caching.md](caching.md). The cache is checked **before** embedding/search, so an
exact repeat costs zero embedding and chat-model calls.
