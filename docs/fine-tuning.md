# Retriever Fine-Tuning

Fine-tune the embedding model on **synthetic, in-domain** query–passage pairs so
retrieval improves on dog-breed questions, and prove the gain with a rank-based eval.
Every run is tracked in **MLflow**. Code in
[`server/finetune/`](../server/finetune/).

## Why

The base embedding model is general-purpose. The corpus is narrow (one dog-breed
book), and we can cheaply generate labeled `(question → source chunk)` pairs from the
chunks we already have. Fine-tuning on those pairs teaches the model this domain's
vocabulary and question style. A rank-based eval (recall@k / MRR) makes the win
measurable — *"no baseline, no story."*

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1. Generate pairs | `finetune/generate_pairs.py` | LLM writes 2–3 questions per chunk → `(query, passage)` pairs, split 80/20 |
| 2. Baseline eval | `finetune/eval_retrieval.py` | recall@3/5 + MRR of the *current* model on the eval split |
| 3. Fine-tune | `finetune/train.py` | bge-base + MultipleNegativesRankingLoss on the train split |
| 4. Compare | `finetune/eval_retrieval.py` | same metrics for base vs fine-tuned → the story |

All four log to a shared MLflow store (local SQLite backend, `server/mlflow.db`).

### 1 — Synthetic pairs (`generate_pairs.py`)

Pulls every chunk from Postgres (`Database.get_all_chunks()`), asks a **local Ollama**
model for questions answerable *only* from each chunk (strict JSON, quality-filtered
for length/dupes), and writes `data/pairs_train.jsonl` + `data/pairs_eval.jsonl`. Each
pair carries `query`, `passage`, `chunk_id`, `breed`, `page`.

**The split is by chunk, never by pair** — every pair from a given chunk lands entirely
in train or entirely in eval, so an eval passage is never seen during training. The
script asserts the two chunk-id sets are disjoint.

```bash
cd server
uv run python -m finetune.generate_pairs                     # full run
uv run python -m finetune.generate_pairs --limit 5 --questions-per-chunk 2   # smoke test
```

> Question quality tracks the local model. `llama3.1:8b` gives better questions but is
> slow (~15 s/chunk); `--model llama3.2` is ~4× faster and fine for this task.

### 2 & 4 — Retrieval eval (`eval_retrieval.py`)

One rank-metric implementation scores **any** embedder so the numbers are comparable:
corpus = every chunk, queries = the eval split, and a query is a "hit" when its own
`chunk_id` is retrieved. Reports `recall@3`, `recall@5`, `MRR@10`.

```bash
uv run python -m finetune.eval_retrieval --model current                         # baseline
uv run python -m finetune.eval_retrieval --model BAAI/bge-base-en-v1.5           # off-the-shelf
uv run python -m finetune.eval_retrieval --model finetune/models/bge-base-dogbreeds  # fine-tuned
```

`--model current` reuses the app's `EmbeddingGenerator`, replicating the
query/passage prompt asymmetry so the baseline reflects production behavior.

### 3 — Fine-tune (`train.py`)

Loads `pairs_train.jsonl` as `(query, positive)` examples and trains with
**MultipleNegativesRankingLoss** (in-batch negatives — no negative mining needed). An
`InformationRetrievalEvaluator` over the eval split logs recall@k/MRR each epoch. The
checkpoint is saved to `finetune/models/bge-base-dogbreeds/`.

```bash
uv run python -m finetune.train --epochs 2 --batch-size 32
```

Trains locally on CPU/Apple MPS in minutes for this dataset size.

## Design decisions

- **Base model = `BAAI/bge-base-en-v1.5` (768-dim), not bge-small/MiniLM.** Higher
  retrieval quality *and* easier to ship: it matches the DB's hardcoded `vector(768)`
  column, so deploying it needs no schema change. The 384-dim small models would force
  a destructive `adjust_vector_dimension(384)` + full re-ingest for no quality gain.
- **bge query instruction.** bge retrieval expects `"Represent this sentence for
  searching relevant passages: "` on the query side only. Applied identically in
  training and eval so the numbers are honest.
- **MNRL over triplets.** With one positive per query and a large batch, in-batch
  negatives give strong signal without mining hard negatives.

## Experiment tracking (MLflow)

Every run logs params + metrics (+ dataset artifacts for step 1) to one store, so the
generation, baseline, and training runs sit side by side.

```bash
cd server
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Experiments: `finetune-pairs`, `retrieval-eval`, `finetune-embed`.

## Results

<!-- Fill from the first full run: baseline vs fine-tuned on the eval split. -->

| Model | recall@3 | recall@5 | MRR@10 |
|---|---|---|---|
| current (app `EmbeddingGenerator`) | _TBD_ | _TBD_ | _TBD_ |
| `bge-base-en-v1.5` (off-the-shelf) | _TBD_ | _TBD_ | _TBD_ |
| `bge-base-dogbreeds` (fine-tuned) | _TBD_ | _TBD_ | _TBD_ |

## Deploying the fine-tuned model (follow-up)

Not wired into the live app yet. When the eval proves the win: add a
sentence-transformers provider branch to `EmbeddingGenerator.__init__` behind a config
flag, then re-ingest the corpus (`uv run python -m src.ingest --force`) to re-embed all
chunks with the new model. Cheap because bge-base is already 768-dim (no schema change);
keep the provider switchable so you can revert.
