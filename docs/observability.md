# Observability

Two layers: **structured logging** (correlate what happened on any one request)
and **online eval** (track answer *quality* on real traffic, and its drift over
time). Offline quality gates live in [evaluation.md](evaluation.md); this doc is
about production.

# Structured Logging

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

# Online Eval — Production Quality Monitoring

Offline evals ([evaluation.md](evaluation.md)) gate quality *before* deploy on a
fixed dataset. They can't see what real users ask or catch quality **drift** —
when a model/provider swap, a corpus change, or shifting traffic quietly degrades
answers. Online eval closes that gap: it scores real `/query` traffic and rolls
the signals up so drift is a trend line. Code in `server/src/online_eval.py`.

## Why

The live server never touched MLflow, and nothing recorded per-query quality. So
a regression that offline evals don't cover (a new class of question, a provider
outage returning error strings, retrieval scores collapsing) was invisible until
a user complained. Recording cheap signals on every query — plus a sampled LLM
judge — makes quality observable in the same MLflow store the fine-tuning pipeline
already uses.

## How it works

- **No added latency.** `/query` schedules the recorder as a Starlette
  `BackgroundTask`, so the DB write (and the optional judge) run *after* the
  response is sent. The write is best-effort — a Neon cold-start is logged and
  dropped, never surfaced to the user, exactly like the [answer cache](caching.md).
- **Deterministic signals on every query** (free, no LLM): retrieval-score stats
  (`sim_max/mean/min`, `num_chunks`, empty-retrieval), `answer_len`, `cached`,
  the request's `provider`, and **deterministic refusal detection**
  (`src/refusal.py`, shared with the adversarial suite).
- **A reference-free LLM judge on a sample.** Gated by `ONLINE_JUDGE_ENABLED`
  (off in dev) **and** `ONLINE_JUDGE_SAMPLE_RATE` (default `0.1`), and only on
  fresh (non-cached) answers. It scores Ragas `faithfulness` + `answer_relevancy`
  (neither needs a reference answer) using the same local-Ollama judge as the
  offline suite. The sample rate is the cost knob — the judge doubles LLM calls on
  sampled rows.
- **User feedback.** `POST /feedback` records a thumbs up/down (`rating` ±1)
  correlated to the query via `request_id`. See [api-reference.md](api-reference.md).
- **Storage.** One row per query in the `online_eval` Postgres table (and
  `feedback`), alongside `query_cache` in the same Neon DB.

Each row carries `request_id`, so an online-eval record joins back to the exact
structured log lines for that request.

## Drift → MLflow

`scripts/aggregate_online_eval.py` rolls a recent window up into a dedicated
`online-eval` MLflow experiment — the same store the fine-tuning runs use. Run it
on a schedule (Fly cron / GitHub Actions) or by hand; each run is one point in the
drift time series (`step` = epoch-hour). Metrics: `n_queries`, `refusal_rate`,
`cache_hit_rate`, `empty_retrieval_rate`, `sim_max_mean`, `sim_mean_mean`,
`judge_faithfulness_mean`, `judge_relevancy_mean`, `feedback_neg_rate`.

```bash
cd server
uv run python -m scripts.aggregate_online_eval --window-hours 24   # → MLflow
uv run python -m scripts.aggregate_online_eval --no-mlflow         # print only
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db           # open "online-eval"
```

## Configuration

| Env var | Default | Effect |
|---|---|---|
| `ONLINE_EVAL_ENABLED` | `true` | Record per-query signals to `online_eval` |
| `ONLINE_JUDGE_ENABLED` | `false` | Also run the sampled reference-free LLM judge (turn on in prod) |
| `ONLINE_JUDGE_SAMPLE_RATE` | `0.1` | Fraction of fresh answers the judge scores |

## Verifying

```bash
cd server
uv run uvicorn src.main:app --port 8000
# in another shell:
curl -s -X POST localhost:8000/query -H 'Content-Type: application/json' \
  -H 'X-Request-ID: q1' -d '{"query":"Where does the Weimaraner come from?","top_k":5}'
curl -s -X POST localhost:8000/feedback -H 'Content-Type: application/json' \
  -d '{"request_id":"q1","query":"Where does the Weimaraner come from?","rating":1}'
# rows land after the response (background task):
psql "$DATABASE_URL" -c "SELECT request_id, refused, sim_max, cached, judge_sampled FROM online_eval ORDER BY created_at DESC LIMIT 5;"
# force the sampled judge locally:
ONLINE_JUDGE_ENABLED=true ONLINE_JUDGE_SAMPLE_RATE=1.0 uv run uvicorn src.main:app
```

An out-of-scope query (`"How much does a Toyota Corolla cost?"`) should record
`refused = true`; an in-scope one `refused = false`. Request latency is unchanged
before/after — the write and judge never touch the response path.
