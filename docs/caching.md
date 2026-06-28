# Answer Caching

A repeated question is served straight from Postgres — no embedding, no search,
no LLM. Measured: **~9000× faster on a hit** (29.6s → 0.003s) and zero Ollama
calls.

## Why Postgres (not in-memory)

The text path (FastAPI) and the voice path (LiveKit worker) are **separate
processes**. An in-process dict wouldn't be shared between them and would die on
restart. A Postgres-backed cache is:

- **Shared** across both processes — ask in text, the voice agent benefits too.
- **Persistent** across restarts.
- Free to add — pgvector/Postgres is already a dependency.

## Schema (`query_cache`)

| Column | Purpose |
|---|---|
| `query_norm` | normalized question (cache key part) |
| `mode` | `text` or `voice` (answers differ in style → separate namespaces) |
| `top_k` | retrieval depth (different `top_k` → different answer) |
| `query_text`, `answer`, `chunks` (JSONB) | stored payload |
| `hit_count`, `created_at`, `last_hit_at` | observability |

Unique index on `(query_norm, mode, top_k)` — the cache key.

## Key normalization

`normalize_query()` lowercases, collapses whitespace, and trims surrounding
punctuation. So `"What's a Pug?"`, `"whats a pug"`, and `"  what's a pug  "` hit
the same entry. This also absorbs minor STT punctuation noise on the voice path.

> Matching is **exact-normalized**, by design ("same question twice"). Semantic
> caching (embed the query, reuse a cached answer when cosine ≥ a high threshold)
> is a natural extension — the query embedding is already computed for retrieval —
> but carries a wrong-answer risk for near-but-different questions, so it's left
> off by default.

## Request flow

```
query() → normalize → cache lookup (exact)
   hit  → return { answer, chunks, cached: true }      # no embed/search/LLM
   miss → embed → hybrid search → generate → store → return { ..., cached: false }
```

The lookup happens **before** embedding, so an exact repeat costs nothing. Cache
reads/writes are best-effort: any cache failure degrades to a normal miss and
never breaks a request.

## Invalidation

The cache is cleared whenever the knowledge base changes:

- on `insert_chunks` (ingest / re-ingest), and
- on `delete_document`,

so answers can never reference content that no longer exists. Manual clear:
`DELETE /cache` or `Database().clear_cache()`.

## Voice integration

On a user turn the agent calls the cached `RAGService.query(mode="voice")`:

- **Hit or miss** → it speaks the answer via `session.say()` and raises
  `StopResponse` to skip the LLM turn entirely (on a hit, the LLM never runs).
- **Any error** → it falls back to injecting retrieved context and letting the
  default LLM reply, so the user is never left without an answer.

Note: voice still runs STT (to know the question) and TTS (to speak the answer);
the cache saves the embedding, search, and LLM generation. Caching TTS *audio*
to also save speech cost is a possible future step.

## Observability

`hit_count` / `last_hit_at` per entry, and the `cached` flag on `/query`
responses (surfaced to the client type) let you see hit rates.
