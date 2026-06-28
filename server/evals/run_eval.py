"""Ragas evaluation harness for the dog-breed RAG system.

Runs the *real* RAG pipeline (``RAGService.query``) over a golden dataset of
questions and scores it with Ragas metrics. The judge LLM and embeddings run on
the same local Ollama models the app uses, so the eval costs no API tokens.

Metrics
-------
Retrieval (is the right context fetched?):
  - context_precision  : are the retrieved chunks relevant to the question?
  - context_recall     : do the retrieved chunks cover the reference answer?
Generation (is the answer good?):
  - faithfulness       : is the answer grounded in the retrieved chunks (no hallucination)?
  - answer_relevancy   : does the answer actually address the question?
  - factual_correctness: does the answer match the reference facts?

Usage
-----
    uv run python -m evals.run_eval                 # full suite
    uv run python -m evals.run_eval --limit 3       # quick smoke test
    uv run python -m evals.run_eval --quick         # only the cheap metrics

Exits non-zero if any metric mean falls below its threshold (CI-friendly).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

# Ragas <1.0 emits deprecation warnings for the classic metric imports below;
# they are the stable API for evaluate() in 0.4.x, so silence the noise.
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

# Make ``src`` importable when run as a script or module from server/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    FactualCorrectness,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig  # noqa: E402
from langchain_ollama import ChatOllama, OllamaEmbeddings  # noqa: E402

from src.config import settings  # noqa: E402
from src.database import Database  # noqa: E402
from src.embeddings import EmbeddingGenerator  # noqa: E402
from src.rag_service import RAGService  # noqa: E402

GOLDEN_PATH = Path(__file__).resolve().parent / "golden.jsonl"

# Per-metric pass thresholds. Deliberately lenient: a local 8B judge is noisier
# than a frontier model, so treat these as regression guards, not absolutes.
THRESHOLDS = {
    "context_precision": 0.70,
    "context_recall": 0.70,
    "faithfulness": 0.70,
    "answer_relevancy": 0.70,
    "factual_correctness(mode=f1)": 0.50,
}


def load_golden() -> list[dict]:
    items = []
    for line in GOLDEN_PATH.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            items.append(json.loads(line))
    return items


def build_dataset(rag: RAGService, items: list[dict], top_k: int) -> tuple[EvaluationDataset, list[dict]]:
    """Run the real RAG pipeline for each question and collect Ragas samples."""
    samples, rows = [], []
    for i, it in enumerate(items, 1):
        res = rag.query(it["question"], top_k=top_k, mode="text")
        contexts = [c["content"] for c in res.get("chunks", [])]
        answer = res.get("answer") or ""
        retrieved_breeds = [
            (c.get("metadata") or {}).get("breed") for c in res.get("chunks", [])
        ]
        breed_hit = (
            it.get("expected_breed") in retrieved_breeds
            if it.get("expected_breed")
            else None
        )
        print(
            f"  [{i}/{len(items)}] {'cache' if res.get('cached') else 'fresh'} "
            f"| breed_retrieved={breed_hit} | {it['question'][:55]!r}"
        )
        samples.append(
            SingleTurnSample(
                user_input=it["question"],
                response=answer,
                retrieved_contexts=contexts,
                reference=it["ground_truth"],
            )
        )
        rows.append({"breed_hit": breed_hit})
    return EvaluationDataset(samples=samples), rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Ragas eval for the dog-breed RAG")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    parser.add_argument("--top-k", type=int, default=8, help="chunks to retrieve per question")
    parser.add_argument("--quick", action="store_true", help="only the cheap retrieval+faithfulness metrics")
    parser.add_argument("--workers", type=int, default=2, help="concurrent metric workers (keep low for local Ollama)")
    args = parser.parse_args()

    items = load_golden()
    if args.limit:
        items = items[: args.limit]

    rag = RAGService(EmbeddingGenerator(), Database())
    print(f"Running RAG over {len(items)} eval question(s) (top_k={args.top_k})...")
    dataset, rows = build_dataset(rag, items, args.top_k)

    judge = LangchainLLMWrapper(
        ChatOllama(model=settings.ollama_chat_model, base_url=settings.ollama_base_url, temperature=0)
    )
    embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=settings.ollama_embedding_model, base_url=settings.ollama_base_url)
    )

    if args.quick:
        metrics = [LLMContextRecall(), Faithfulness()]
    else:
        metrics = [
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
            Faithfulness(),
            ResponseRelevancy(),
            FactualCorrectness(),
        ]

    run_config = RunConfig(timeout=300, max_retries=2, max_workers=args.workers)
    print(f"\nScoring with {len(metrics)} metric(s) via local Ollama judge "
          f"({settings.ollama_chat_model})... this can take a few minutes.\n")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge,
        embeddings=embeddings,
        run_config=run_config,
        show_progress=True,
    )

    df = result.to_pandas()
    # Attach the deterministic breed-retrieval signal alongside the LLM metrics.
    for idx, row in enumerate(rows):
        df.loc[idx, "breed_hit"] = row["breed_hit"]

    print("\n=== Per-question scores ===")
    metric_cols = [c for c in df.columns if c in THRESHOLDS or c.startswith("factual")]
    with_q = ["user_input"] + metric_cols + (["breed_hit"] if "breed_hit" in df else [])
    try:
        import pandas as pd  # noqa
        with pd.option_context("display.max_colwidth", 45, "display.width", 200):
            print(df[with_q].to_string(index=False))
    except Exception:
        print(df[with_q].to_string(index=False))

    print("\n=== Means vs thresholds ===")
    failed = []
    for col in metric_cols:
        mean = df[col].mean(skipna=True)
        thr = THRESHOLDS.get(col)
        if thr is None:
            print(f"  {col:34} {mean:.3f}")
            continue
        ok = mean >= thr
        print(f"  {col:34} {mean:.3f}  (>= {thr:.2f})  {'PASS' if ok else 'FAIL'}")
        if not ok:
            failed.append(col)

    if "breed_hit" in df:
        hits = df["breed_hit"].dropna()
        if len(hits):
            print(f"  {'breed_retrieved (deterministic)':34} {hits.mean():.3f}  ({int(hits.sum())}/{len(hits)})")

    if failed:
        print(f"\nFAILED: {', '.join(failed)}")
        return 1
    print("\nAll metrics passed their thresholds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
