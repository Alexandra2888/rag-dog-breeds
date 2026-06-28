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
