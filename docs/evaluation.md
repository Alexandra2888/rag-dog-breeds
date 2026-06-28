# Evaluation

Quality is tracked with [Ragas](https://docs.ragas.io). The harness runs the
**real** pipeline (`RAGService.query`) over a versioned golden dataset and scores
it with a **local Ollama judge**, so evaluation costs no API tokens. Code in
[`server/evals/`](../server/evals/README.md).

## Why a framework + a deterministic check

This corpus has **structured, labeled facts** (Origin / Weight / Height / Life
span), so factual answers can be checked objectively. The suite therefore
combines:

- **Ragas metrics** (LLM-judged) for retrieval and answer quality, and
- a **deterministic** `breed_retrieved` check (did the expected breed's chunk get
  retrieved?) that needs no LLM and can't be fooled.

This is more trustworthy than relying on an LLM judge alone — the deterministic
signal anchors the noisy one.

## Metrics

| Metric | Layer | Question |
|---|---|---|
| `context_precision` | retrieval | Are retrieved chunks relevant? |
| `context_recall` | retrieval | Do chunks cover the reference answer? |
| `faithfulness` | generation | Is the answer grounded (no hallucination)? |
| `answer_relevancy` | generation | Does the answer address the question? |
| `factual_correctness` | generation | Does the answer match the reference facts? |
| `breed_retrieved` | retrieval (deterministic) | Did the expected breed's chunk appear? |

## Golden dataset (`evals/golden.jsonl`)

One JSON object per line: `question`, `ground_truth`, and optional
`expected_breed`. Origin facts were **extracted from the ingested book**, not
assumed, and interleaving-corrupted extractions were excluded. It includes
misspelled-breed questions (voice/STT realism) and an out-of-scope question to
test that the system declines rather than hallucinates.

## Running

```bash
cd server
uv run python -m evals.run_eval            # full suite
uv run python -m evals.run_eval --limit 3  # quick smoke test
uv run python -m evals.run_eval --quick    # context_recall + faithfulness only
uv run python -m evals.run_eval --workers 1
```

The runner prints per-question scores and means, and **exits non-zero** if any
metric mean falls below its threshold — drop it into CI as a regression gate.

## Interpreting scores

- A local 8B judge is **noisier** than a frontier model. Thresholds in
  `run_eval.py` are deliberately lenient — treat them as regression guards, not
  absolute quality. For higher-fidelity scoring, point the judge at a stronger
  model (edit `ChatOllama(...)` in `run_eval.py`).
- Smoke test (2 questions) scored `context_recall` and `faithfulness` at 1.0 with
  the expected breed retrieved both times.
- The retrieval design separately measures **top-1 39/40, top-5 40/40** on the
  breed set (see [rag-pipeline.md](rag-pipeline.md)).

## Extending

Add lines to `golden.jsonl` (size/lifespan questions, more breeds, more
out-of-scope refusal cases), tune `THRESHOLDS`, or add Ragas metrics in
`run_eval.py`. Validate new ground-truth facts against the book first.
