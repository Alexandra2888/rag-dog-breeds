"""Fine-tune bge-base on the synthetic dog-breed pairs with MNRL.

Loads ``pairs_train.jsonl`` as (query, positive passage) examples and fine-tunes
``BAAI/bge-base-en-v1.5`` with MultipleNegativesRankingLoss (in-batch negatives —
no negative mining needed). An InformationRetrievalEvaluator over the eval split
(corpus = every DB chunk, same as eval_retrieval.py) tracks recall@k / MRR each
epoch. Params + per-epoch metrics + the final model go to MLflow (experiment
``finetune-embed``); the checkpoint is saved under ``finetune/models/``.

Usage
-----
    uv run python -m finetune.train
    uv run python -m finetune.train --epochs 2 --batch-size 32
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# MPS can miss ops for some transformer layers; fall back to CPU rather than crash.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Make ``src`` importable when run as a module from server/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import Database  # noqa: E402
from finetune.mlflow_utils import setup_mlflow  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_OUT = Path(__file__).resolve().parent / "models" / "bge-base-dogbreeds"
# bge retrieval expects this instruction on the QUERY side (must match eval_retrieval.py).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_pairs(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def build_evaluator(eval_items: list[dict], query_prefix: str):
    """IR evaluator over the eval split against the full-corpus of DB chunks."""
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    chunks = Database().get_all_chunks()
    corpus = {c["id"]: c["content"] for c in chunks}
    queries, relevant = {}, {}
    for i, it in enumerate(eval_items):
        if it["chunk_id"] not in corpus:
            continue
        qid = f"q{i}"
        queries[qid] = query_prefix + it["query"]
        relevant[qid] = {it["chunk_id"]}
    return InformationRetrievalEvaluator(
        queries=queries, corpus=corpus, relevant_docs=relevant,
        precision_recall_at_k=[3, 5], mrr_at_k=[10], accuracy_at_k=[1, 3, 5],
        name="dogbreeds-eval", show_progress_bar=False,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Fine-tune bge-base with MNRL")
    ap.add_argument("--base-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--train-path", type=Path, default=DATA_DIR / "pairs_train.jsonl")
    ap.add_argument("--eval-path", type=Path, default=DATA_DIR / "pairs_eval.jsonl")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--mlflow-experiment", default="finetune-embed")
    ap.add_argument("--mlflow-uri", default=None)
    args = ap.parse_args()

    from torch.utils.data import DataLoader
    from sentence_transformers import SentenceTransformer, InputExample, losses

    train_items = load_pairs(args.train_path)
    eval_items = load_pairs(args.eval_path)
    if not train_items:
        print(f"No training pairs in {args.train_path} — run generate_pairs first.")
        return 1
    print(f"Loaded {len(train_items)} train / {len(eval_items)} eval pairs.")

    model = SentenceTransformer(args.base_model)
    print(f"Device: {model.device}")

    train_examples = [
        InputExample(texts=[BGE_QUERY_PREFIX + it["query"], it["passage"]])
        for it in train_items
    ]
    train_loader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size, drop_last=True)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    evaluator = build_evaluator(eval_items, BGE_QUERY_PREFIX)

    steps_per_epoch = max(1, len(train_examples) // args.batch_size)
    warmup = int(0.1 * steps_per_epoch * args.epochs)

    setup_mlflow(args.mlflow_experiment, args.mlflow_uri)
    import mlflow
    with mlflow.start_run(run_name=f"train:{args.base_model.split('/')[-1]}"):
        mlflow.log_params({
            "base_model": args.base_model,
            "loss": "MultipleNegativesRankingLoss",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "n_train_pairs": len(train_examples),
            "n_eval_queries": len(eval_items),
            "warmup_steps": warmup,
        })

        # Log the primary IR score each epoch (callback receives score/epoch/steps).
        def epoch_cb(score, epoch, steps):
            mlflow.log_metric("primary_score", float(score), step=epoch)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        model.fit(
            train_objectives=[(train_loader, train_loss)],
            evaluator=evaluator,
            epochs=args.epochs,
            warmup_steps=warmup,
            optimizer_params={"lr": args.lr},
            output_path=str(args.output_dir),
            save_best_model=True,
            callback=epoch_cb,
            show_progress_bar=True,
        )

        # Final eval → log the full metric dict (recall@k, mrr, accuracy@k).
        final = evaluator(model)
        if isinstance(final, dict):
            clean = {k.replace("dogbreeds-eval_", ""): float(v)
                     for k, v in final.items() if isinstance(v, (int, float))}
            mlflow.log_metrics({f"final_{k}": v for k, v in clean.items()})
            print("\n=== Final IR metrics ===")
            for k, v in clean.items():
                print(f"  {k:32} {v:.4f}")
        mlflow.log_param("output_dir", str(args.output_dir))

    print(f"\nSaved fine-tuned model to {args.output_dir}")
    print(f"Next: uv run python -m finetune.eval_retrieval --model {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
