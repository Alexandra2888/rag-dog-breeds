# RAG Evaluation (Ragas)

Evaluates the dog-breed RAG system with [Ragas](https://docs.ragas.io). The harness
runs the **real** pipeline (`RAGService.query`) over a golden dataset and scores it.
The judge LLM and embeddings run on the **local Ollama** models the app already uses,
so evaluation costs no API tokens.

## Files

- `golden.jsonl` — versioned golden dataset: one JSON object per line with
  `question`, `ground_truth` (reference answer), and optional `expected_breed`
  (used for a free deterministic retrieval check). Origin facts were extracted
  from the ingested book, not assumed.
- `run_eval.py` — runner: builds the dataset by querying the live RAG service,
  scores with Ragas, prints per-question + mean scores, and exits non-zero if any
  metric mean falls below its threshold (CI-friendly).
- `adversarial.jsonl` / `run_adversarial.py` — the **adversarial / edge-case**
  suite (jailbreaks, prompt-injection, out-of-scope, nonexistent breeds,
  ambiguity, other languages, degenerate inputs). See below.

## Metrics

| Metric | Layer | Question it answers |
|---|---|---|
| `context_precision` | retrieval | Are the retrieved chunks relevant? |
| `context_recall` | retrieval | Do the chunks cover the reference answer? |
| `faithfulness` | generation | Is the answer grounded in the chunks (no hallucination)? |
| `answer_relevancy` | generation | Does the answer address the question? |
| `factual_correctness` | generation | Does the answer match the reference facts? |
| `breed_retrieved` | retrieval (deterministic) | Did the expected breed's chunk get retrieved? (free, no LLM) |

## Prerequisites

- Postgres (pgvector) up and the book ingested (`uv run python -m src.ingest`).
- Ollama running with the chat + embedding models from `.env`
  (`OLLAMA_CHAT_MODEL`, `OLLAMA_EMBEDDING_MODEL`).

## Run

```bash
cd server
uv run python -m evals.run_eval            # full suite (slow on a local 8B judge)
uv run python -m evals.run_eval --limit 3  # quick smoke test
uv run python -m evals.run_eval --quick    # only context_recall + faithfulness
uv run python -m evals.run_eval --workers 1  # if the local model is overloaded
```

## Adversarial / edge-case suite

Where `run_eval.py` asks "does the system answer easy questions well?",
`run_adversarial.py` asks "does it *break* on inputs designed to break it?" It is
scored **per category** so many easy passes can't dilute one jailbreak regression.

- `adversarial.jsonl` — one JSON object per line. Beyond `question`/`ground_truth`/
  `expected_breed` it adds:
  - `category` — `jailbreak` · `prompt_injection` · `out_of_scope` ·
    `ambiguous_multi_breed` · `nonexistent_breed` · `non_english` ·
    `empty_input` · `very_long_input`.
  - `expectation` — `refuse` · `answer` · `answer_or_refuse` (drives the refusal metric).
  - `must_not_contain` — optional strings the answer must not echo (prompt-leak /
    injection guard).
- **Deterministic refusal metric** (`src/refusal.py`): a confident hallucination
  and a correct "I don't know" are told apart without the LLM judge, by matching
  the abstention phrasings the app's own prompts steer toward (and treating the
  empty-retrieval path as a refusal).

| Per-category metric | Meaning |
|---|---|
| `refusal_accuracy` | fraction of applicable rows that behaved per `expectation` |
| `leak_pass_rate` | fraction of rows that did **not** echo a `must_not_contain` string |
| `breed_hit_rate` | expected breed retrieved (where applicable) |
| `faithfulness` / `answer_relevancy` | opt-in Ragas score on answered rows (`--faithfulness`) |

```bash
cd server
uv run python -m evals.run_adversarial                 # full suite (clears cache first)
uv run python -m evals.run_adversarial --limit 5       # smoke test
uv run python -m evals.run_adversarial --category jailbreak
uv run python -m evals.run_adversarial --faithfulness  # + Ragas on answered rows
uv run python -m evals.run_adversarial --no-clear-cache
uv run python -m evals.run_adversarial --strict        # exit non-zero on ANY category miss
```

The gate is **soft by default**: it prints a per-category PASS/FAIL table and
exits non-zero only if a category in `HARD_CATEGORIES` (empty until baselines are
known) regresses. Use `--strict` to fail on any miss, or promote reliable
categories (jailbreak, prompt_injection, out_of_scope, empty_input) into
`HARD_CATEGORIES` to block CI. The cache is cleared before each run by default so
a stale benign answer to a now-adversarial phrasing can't mask a regression.

## Notes

- A local 8B judge is **noisier** than a frontier model. Thresholds in
  `run_eval.py` are deliberately lenient — treat them as regression guards, not
  absolute quality scores. For higher-fidelity scoring, point the judge at a
  stronger model (edit the `ChatOllama(...)` in `run_eval.py`).
- The answer cache is shared, so re-running reuses cached answers (the eval still
  scores them). Clear it with `DELETE /cache` or `Database().clear_cache()` to
  force fresh generation.
- To grow coverage, add lines to `golden.jsonl` (size/lifespan questions,
  more breeds, more "not in the book" refusal cases).
