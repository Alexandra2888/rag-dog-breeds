# Observability — Structured Logging

The API emits **structured logs** via [structlog](https://www.structlog.org). Every
request gets a `request_id` that is bound once and automatically attached to *every*
log line produced while handling that request — including logs from
`database.py`, `rag_service.py`, and `embeddings.py` — with no changes at those call
sites. Code in [`server/src/logging_config.py`](../server/src/logging_config.py).

## Why

Plain `logging.basicConfig` string logs can't be correlated: when two requests
interleave, you can't tell which log line belongs to which. Structured logs fix
this — one JSON object per line, a shared `request_id`, and a per-request access
line with timing — so logs are greppable, correlatable, and ready for a log
aggregator. This closes the production-observability gap.

## How it works

- **One setup, all entrypoints.** `setup_logging()` routes *stdlib* logging through a
  single structlog `ProcessorFormatter`. The existing `logging.getLogger(__name__)`
  calls across the codebase keep working unchanged — they just render as structured
  output now. All three entrypoints (`main.py`, `ingest.py`, `livekit_agent.py`) call
  it, so there is exactly one logging config.
- **request_id via contextvars.** A `contextvar` is bound by the FastAPI middleware
  (`request_context_middleware` in `main.py`) and merged into every event by
  structlog's `merge_contextvars` processor. Because it's a contextvar, downstream
  module logs inherit it automatically within the same request.
- **Inbound propagation.** The middleware honors an incoming `X-Request-ID` header (so
  an id set by the frontend or voice agent carries through) or mints a UUID, and echoes
  it back in the `X-Request-ID` response header.
- **One access line per request** with `method`, `path`, `status`, and `duration_ms`.
- **The global exception handler** logs with a full traceback and inherits the
  `request_id`, so 500s are traceable to the exact request.

## Configuration

Two settings (see [configuration.md](configuration.md)):

| Env var | Default | Effect |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Root log level |
| `LOG_JSON` | `false` | `true` → one JSON object per line (prod / aggregators); `false` → colorized console (dev) |

## Example

`LOG_JSON=true`, a single `/query` request (`X-Request-ID: query-trace-99`). The
access line and the downstream `rag_service` line **share the same `request_id`** —
that shared id is the whole point:

```json
{"method": "POST", "path": "/query", "status": 200, "duration_ms": 1843.2, "event": "request", "request_id": "query-trace-99", "logger": "api.access", "level": "info", "timestamp": "2026-07-09T18:01:28Z"}
{"event": "Retrieved 5 relevant chunks", "request_id": "query-trace-99", "logger": "src.rag_service", "level": "info", "timestamp": "2026-07-09T18:01:27Z"}
```

Grepping `query-trace-99` returns every log line for that one request, across modules.

## Verifying

```bash
cd server
LOG_JSON=true uv run uvicorn src.main:app --port 8000
# in another shell:
curl -si -H 'X-Request-ID: my-trace-42' localhost:8000/health | grep -i x-request-id
```

The response header `X-Request-ID` matches the `request_id` in the access log; a
request without the header gets a freshly minted id, and a second request gets a
distinct one.
