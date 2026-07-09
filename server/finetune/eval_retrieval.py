"""Retrieval eval: recall@3, recall@5, MRR over the synthetic eval split.

Scores an embedding model on a realistic retrieval task: the corpus is every chunk
in the DB, the queries are ``pairs_eval.jsonl``, and a query is "hit" when its own
source chunk (by ``chunk_id``) is retrieved. One numpy metric implementation is used
for every model, so the numbers are directly comparable across:

    --model current                 # the app's live EmbeddingGenerator (baseline)
    --model BAAI/bge-base-en-v1.5   # an off-the-shelf sentence-transformers model
    --model finetune/models/bge-base-dogbreeds   # a fine-tuned checkpoint

Runs are logged to the shared MLflow store (experiment ``retrieval-eval``).

Usage
-----
    uv run python -m finetune.eval_retrieval --model current
    uv run python -m finetune.eval_retrieval --model BAAI/bge-base-en-v1.5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Make ``src`` importable when run as a module from server/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import Database  # noqa: E402
from finetune.mlflow_utils import setup_mlflow  # noqa: E402

DEFAULT_EVAL_PATH = Path(__file__).resolve().parent / "data" / "pairs_eval.jsonl"
# bge retrieval models expect this instruction on the QUERY side only.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_eval(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def ir_metrics(query_embs: np.ndarray, corpus_embs: np.ndarray,
               relevant_idx: list[int], ks=(3, 5), mrr_at: int = 10) -> tuple[dict, float]:
    """Cosine-rank each query against the corpus; one relevant doc per query."""
    q = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-12)
    c = corpus_embs / (np.linalg.norm(corpus_embs, axis=1, keepdims=True) + 1e-12)
    sims = q @ c.T                          # (Q, C)
    order = np.argsort(-sims, axis=1)       # descending similarity
    ranks = np.array([
        int(np.where(order[i] == rel)[0][0])  # 0-based rank of the relevant doc
        for i, rel in enumerate(relevant_idx)
    ])
    recall_at = {k: float(np.mean(ranks < k)) for k in ks}
    rr = np.where(ranks < mrr_at, 1.0 / (ranks + 1), 0.0)
    return recall_at, float(np.mean(rr))


def embed_current(queries: list[str], passages: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Baseline: the app's EmbeddingGenerator, replicating the query/doc asymmetry."""
    from src.embeddings import EmbeddingGenerator
    eg = EmbeddingGenerator()
    q = np.array([eg.generate_embedding(t, input_type="search_query") for t in queries], dtype=np.float32)
    p = np.array(eg.generate_embeddings_batch(passages), dtype=np.float32)  # search_document
    return q, p


def embed_sentence_transformer(model_name: str, queries: list[str], passages: list[str],
                               query_prefix: str | None) -> tuple[np.ndarray, np.ndarray]:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    prefix = query_prefix if query_prefix is not None else (BGE_QUERY_PREFIX if "bge" in model_name.lower() else "")
    q = model.encode([prefix + t for t in queries], normalize_embeddings=False,
                     batch_size=64, show_progress_bar=True)
    p = model.encode(passages, normalize_embeddings=False, batch_size=64, show_progress_bar=True)
    return np.asarray(q, dtype=np.float32), np.asarray(p, dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description="Retrieval eval (recall@k, MRR)")
    ap.add_argument("--model", required=True,
                    help="'current' (app EmbeddingGenerator) or a sentence-transformers name/path")
    ap.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL_PATH)
    ap.add_argument("--query-prefix", type=str, default=None,
                    help="override the query instruction prefix (default: auto for bge)")
    ap.add_argument("--mlflow-experiment", type=str, default="retrieval-eval")
    ap.add_argument("--mlflow-uri", type=str, default=None)
    ap.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    args = ap.parse_args()

    eval_items = load_eval(args.eval_path)
    if not eval_items:
        print(f"No eval items in {args.eval_path} — run generate_pairs first.")
        return 1

    # Corpus = every chunk in the DB; index by chunk_id -> row position.
    chunks = Database().get_all_chunks()
    corpus_ids = [c["id"] for c in chunks]
    corpus_texts = [c["content"] for c in chunks]
    id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}

    # Keep only eval queries whose relevant chunk is in the corpus.
    queries, relevant_idx, dropped = [], [], 0
    for it in eval_items:
        idx = id_to_idx.get(it["chunk_id"])
        if idx is None:
            dropped += 1
            continue
        queries.append(it["query"])
        relevant_idx.append(idx)
    if dropped:
        print(f"WARNING: dropped {dropped} eval quer(ies) whose chunk_id is not in the corpus.")
    if not queries:
        print("No eval queries map to the corpus.")
        return 1

    print(f"Embedding {len(queries)} queries + {len(corpus_texts)} corpus passages "
          f"with '{args.model}'...")
    if args.model == "current":
        q_embs, c_embs = embed_current(queries, corpus_texts)
    else:
        q_embs, c_embs = embed_sentence_transformer(args.model, queries, corpus_texts, args.query_prefix)

    recall_at, mrr = ir_metrics(q_embs, c_embs, relevant_idx)
    metrics = {"recall@3": recall_at[3], "recall@5": recall_at[5], "mrr": mrr}

    print("\n=== Retrieval eval ===")
    print(f"model        : {args.model}")
    print(f"queries      : {len(queries)}   corpus: {len(corpus_texts)}")
    print(f"recall@3     : {metrics['recall@3']:.3f}")
    print(f"recall@5     : {metrics['recall@5']:.3f}")
    print(f"MRR@10       : {metrics['mrr']:.3f}")

    if not args.no_mlflow:
        import mlflow
        setup_mlflow(args.mlflow_experiment, args.mlflow_uri)
        with mlflow.start_run(run_name=f"eval:{Path(args.model).name}"):
            mlflow.log_params({
                "model": args.model,
                "n_queries": len(queries),
                "corpus_size": len(corpus_texts),
                "dim": int(c_embs.shape[1]),
            })
            mlflow.log_metrics(metrics)
        print("\nLogged to MLflow experiment 'retrieval-eval'.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
