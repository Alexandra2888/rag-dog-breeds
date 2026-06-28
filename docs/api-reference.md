# API Reference

FastAPI backend, base URL `http://localhost:8000`. Interactive docs at `/docs`
(Swagger) and `/redoc`. All bodies are JSON unless noted.

## Service

### `GET /`
Service metadata and the list of available endpoints.

### `GET /health`
```json
{ "status": "healthy", "service": "ollama-rag-dog-breeds", "version": "0.1.0" }
```

## Query & search

### `POST /query`
RAG query: retrieve relevant chunks and generate a grounded answer. Served from
the [answer cache](caching.md) when the question was asked before.

Request:
```json
{ "query": "Where does the Weimaraner come from?", "top_k": 8, "include_metadata": true }
```
- `query` (string, required, non-empty)
- `top_k` (int, default 5, **1–100**)
- `include_metadata` (bool, default true)

Response:
```json
{
  "query": "Where does the Weimaraner come from?",
  "answer": "The Weimaraner originates from Germany.",
  "chunks": [
    { "id": "uuid", "content": "...", "similarity_score": 0.83,
      "metadata": { "breed": "WEIMARANER", "source": "...", "page_number": 246 } }
  ],
  "cached": false
}
```
`cached` is `true` when the answer came from the cache (no LLM call).

### `POST /search`
Vector/hybrid similarity search **without** generating an answer.

Request:
```json
{ "query": "loyal apartment companion", "top_k": 10, "threshold": 0.5 }
```
- `top_k` (int, default 5, 1–100)
- `threshold` (float, optional, 0.0–1.0) — minimum vector similarity

Response: `{ "query", "results": [ChunkResult...], "total_results" }`.

## Ingestion & documents

### `POST /ingest`  → `201`
Ingest a PDF by **server-side path**.
```json
{ "pdf_path": "data/The-Complete-Dog-Breed-Book-...pdf" }
```
Returns `{ "message", "chunks_processed", "document_id" }`.
> Security note: this reads an arbitrary server path — keep it disabled or
> path-restricted on any non-local deployment. Prefer `/ingest/upload`.

### `POST /ingest/upload`  → `201`
Ingest a PDF via multipart upload.
```bash
curl -X POST http://localhost:8000/ingest/upload -F "file=@./data/book.pdf"
```

### `GET /documents`
List ingested documents with chunk counts:
`{ "documents": [{ "id", "document_name", "created_at", "chunk_count" }], "total" }`.

### `DELETE /documents/{document_id}`
Delete a document and its chunks (cascade). `404` if not found. Clears the
answer cache (corpus changed).

## Cache

### `DELETE /cache`
Drop all cached answers. Returns `{ "message", "removed": <count> }`. The cache
is also cleared automatically on ingest and document delete. See
[caching.md](caching.md).

## Voice sessions

### `POST /api/voice/session`
Mint a LiveKit room + access token so the browser can join a voice room. The
voice agent joins automatically.

Request: `{ "user_id": "optional-stable-id" }`
Response: `{ "room_name", "token", "url" }`
`503` if `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` are not configured.

### `DELETE /api/voice/session/{room_name}`
Best-effort teardown of a LiveKit room when the user disconnects.

## Errors

Endpoints return standard HTTP status codes (`400`, `404`, `422` validation,
`500`, `503`). A global handler returns `{ "detail": "Internal server error" }`
for unhandled exceptions; validation errors follow FastAPI's `422` format.
