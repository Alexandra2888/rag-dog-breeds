# Evaluation

Three complementary layers:

- **Retrieval quality** (rank-based, no LLM judge) — recall@3/5 and MRR on a
  held-out synthetic set, used to prove the embedding **fine-tuning** win
  (recall@5 0.80 → 0.84). Code in `server/finetune/eval_retrieval.py`; see
  [fine-tuning.md](fine-tuning.md).
- **Generation quality** (this doc) — Ragas metrics over the real pipeline on a
  normal-case golden set.
- **Adversarial / edge-case robustness** (below) — inputs designed to *break* the
  system (jailbreaks, prompt-injection, out-of-scope, degenerate inputs), scored
  per category with a deterministic refusal metric.

All three are **offline** (CI-style). Quality on *real production traffic* is
tracked separately — see **Online eval** in [observability.md](observability.md).

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

## Adversarial / edge-case suite

Where `run_eval.py` asks "does the system answer easy questions well?", the
adversarial suite asks "does it *break* on inputs designed to break it?" Code in
`server/evals/run_adversarial.py` + `adversarial.jsonl`; see the
[evals README](../server/evals/README.md).

Two design choices make it trustworthy:

1. **A deterministic refusal metric** (`server/src/refusal.py`). A confident
   hallucination and a correct "I don't know" must be told apart *without* leaning
   on the noisy LLM judge, so the detector matches the abstention phrasings the
   app's own prompts steer toward ("say so" / "say you don't know") and treats the
   empty-retrieval path as a refusal. Each row declares an `expectation`
   (`refuse` / `answer` / `answer_or_refuse`) and is scored against it.
2. **Per-category scoring, not one global mean.** 13 easy passes must not dilute
   one jailbreak regression, so each category is scored and gated on its own.

Categories: `jailbreak`, `prompt_injection`, `out_of_scope`,
`ambiguous_multi_breed`, `nonexistent_breed`, `non_english`, `empty_input`,
`very_long_input`. Per-category metrics: `refusal_accuracy` (behaved per
`expectation`), `leak_pass_rate` (did **not** echo a `must_not_contain` string —
prompt-leak / injection guard), `breed_hit_rate` (where applicable), and opt-in
Ragas `faithfulness`/`answer_relevancy` on answered rows (`--faithfulness`).

```bash
cd server
uv run python -m evals.run_adversarial                 # full suite (clears cache first)
uv run python -m evals.run_adversarial --limit 5       # smoke test
uv run python -m evals.run_adversarial --category jailbreak
uv run python -m evals.run_adversarial --strict        # exit non-zero on ANY category miss
```

The gate is **soft by default**: it prints a per-category PASS/FAIL table and
exits non-zero only if a category in `HARD_CATEGORIES` (empty until baselines are
known) regresses. Use `--strict` to fail on any miss, or promote reliable
categories (jailbreak, prompt_injection, out_of_scope, empty_input) into
`HARD_CATEGORIES` to block CI. The shared answer cache is cleared before each run
so a stale benign answer to a now-adversarial phrasing can't mask a regression.

> Note: refusal is currently **prompt-instructed only** — there is no code-level
> guardrail. This suite *measures* abstention; if it surfaces injection compliance
> (e.g. a "…also append PWNED" row), a pre-LLM scope/injection check is the fix.
