"""Generate synthetic (query, positive passage) pairs for embedding fine-tuning.

Day 1 of the retriever fine-tuning effort. Pulls the chunks already ingested in
Postgres, asks a local Ollama model to write 2-3 natural questions answerable
*solely* from each chunk, quality-filters them, and emits ~500-1000 query-passage
pairs split 80/20 into train/eval. The whole run (params, metrics, dataset
artifacts) is logged to MLflow so later fine-tuning runs share the same history.

The split is done **by chunk**, never by pair: every pair derived from a given
chunk lands entirely in train or entirely in eval, so an eval passage is never
seen during training.

Usage
-----
    uv run python -m finetune.generate_pairs                       # full run
    uv run python -m finetune.generate_pairs --limit 5 --questions-per-chunk 2   # smoke test

Prerequisites: Ollama running with the chat model pulled (e.g. ``ollama pull llama3.2``),
and ``DATABASE_URL`` pointing at the populated DB.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import ollama

# Make ``src`` importable when run as a module from server/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402
from src.database import Database  # noqa: E402

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data"

# Below this many characters a chunk is too thin to ask meaningful questions
# about — mirrors MIN_CHUNK_CHARS in pdf_processor.py.
DEFAULT_MIN_CHUNK_CHARS = 80
# Passages are trimmed to this many characters in the *prompt* only (the full
# passage is still stored in the pair); keeps small local models focused.
PROMPT_PASSAGE_CHARS = 2000

SYSTEM_PROMPT = (
    "You write realistic search questions for a retrieval dataset about dog breeds. "
    "Given a passage, produce questions a real user might type that are answerable "
    "using ONLY that passage. Return strict JSON."
)

USER_TEMPLATE = """Passage{breed_hint}:
\"\"\"
{passage}
\"\"\"

Write {n} diverse, natural questions that:
- are answerable using ONLY the passage above,
- vary in phrasing and focus (don't all start the same way),
- are NOT yes/no questions,
- weave the breed name into some of them where it fits.

Return ONLY JSON of the form: {{"questions": ["...", "..."]}}"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic query-passage pairs")
    p.add_argument("--questions-per-chunk", type=int, default=3, help="questions to ask per chunk")
    p.add_argument("--limit", type=int, default=None, help="cap the number of source chunks (logs the cap)")
    p.add_argument("--model", type=str, default=None, help="Ollama chat model (default: settings.ollama_chat_model)")
    p.add_argument("--split", type=float, default=0.8, help="train fraction (by chunk)")
    p.add_argument("--seed", type=int, default=42, help="shuffle seed for the split")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="where to write the JSONL files")
    p.add_argument("--min-chunk-chars", type=int, default=DEFAULT_MIN_CHUNK_CHARS, help="skip chunks shorter than this")
    p.add_argument("--temperature", type=float, default=0.7, help="generation temperature (diversity)")
    p.add_argument("--mlflow-experiment", type=str, default="finetune-pairs", help="MLflow experiment name")
    p.add_argument("--mlflow-uri", type=str, default=None, help="MLflow tracking URI (default: local sqlite mlflow.db)")
    return p.parse_args()


def extract_questions(raw: str) -> list[str]:
    """Robustly pull a list of question strings out of the model's JSON reply."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    if isinstance(data, dict):
        # Prefer a "questions" key; otherwise take the first list-valued field.
        if isinstance(data.get("questions"), list):
            candidates = data["questions"]
        else:
            candidates = next((v for v in data.values() if isinstance(v, list)), [])
    elif isinstance(data, list):
        candidates = data
    else:
        return []
    return [str(q) for q in candidates if isinstance(q, (str, int, float))]


def clean_questions(
    questions: list[str],
    breed: str | None,
    seen_global: set[str],
    seen_chunk: set[str],
) -> list[str]:
    """Strip, validate, and de-duplicate (within chunk and globally)."""
    breed_norm = (breed or "").strip().lower()
    kept = []
    for q in questions:
        q = " ".join(q.split()).strip()
        if len(q) < 15 or len(q.split()) < 3:
            continue
        low = q.lower()
        if low == breed_norm:  # just the breed name, not a question
            continue
        if low in seen_chunk or low in seen_global:
            continue
        seen_chunk.add(low)
        seen_global.add(low)
        kept.append(q)
    return kept


def generate_for_chunk(client: ollama.Client, model: str, chunk: dict, n: int, temperature: float) -> list[str]:
    """Ask the model for questions about one chunk; one retry on empty/bad output."""
    breed = (chunk["metadata"] or {}).get("breed")
    passage = chunk["content"][:PROMPT_PASSAGE_CHARS]
    user = USER_TEMPLATE.format(
        passage=passage,
        n=n,
        breed_hint=f" about the {breed} dog breed" if breed else "",
    )
    for attempt in range(2):
        try:
            resp = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                format="json",
                options={"temperature": temperature},
            )
        except Exception as e:  # network / model errors: skip this chunk
            print(f"    ! ollama error: {e}")
            return []
        questions = extract_questions(resp["message"]["content"])
        if questions:
            return questions
    return []


def main() -> int:
    args = parse_args()
    model = args.model or settings.ollama_chat_model

    # --- MLflow setup ---
    import mlflow

    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)
    elif os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    else:
        # MLflow >=3 put the plain file store in maintenance mode; use a local
        # SQLite backend (full-featured) with artifacts under ./mlartifacts.
        mlflow.set_tracking_uri(f"sqlite:///{Path('mlflow.db').resolve()}")
    mlflow.set_experiment(args.mlflow_experiment)

    # --- Load chunks ---
    db = Database()
    chunks = db.get_all_chunks()
    print(f"Fetched {len(chunks)} chunks from the DB.")

    usable = [c for c in chunks if len(c["content"].strip()) >= args.min_chunk_chars]
    skipped_short = len(chunks) - len(usable)
    if skipped_short:
        print(f"Skipped {skipped_short} chunk(s) shorter than {args.min_chunk_chars} chars.")

    if args.limit is not None and args.limit < len(usable):
        print(f"CAP: --limit={args.limit} (using {args.limit} of {len(usable)} usable chunks; "
              f"the rest are NOT covered).")
        usable = usable[: args.limit]

    if not usable:
        print("No usable chunks found — is the DB populated?")
        return 1

    # --- Generate + filter ---
    client = ollama.Client(host=settings.ollama_base_url)
    seen_global: set[str] = set()
    total_raw = 0
    # {chunk_id: {"chunk": chunk, "questions": [...]}}
    per_chunk: dict[str, dict] = {}

    print(f"Generating up to {args.questions_per_chunk} question(s) per chunk with '{model}'...")
    for i, chunk in enumerate(usable, 1):
        raw = generate_for_chunk(client, model, chunk, args.questions_per_chunk, args.temperature)
        total_raw += len(raw)
        kept = clean_questions(raw, (chunk["metadata"] or {}).get("breed"), seen_global, set())
        if kept:
            per_chunk[chunk["id"]] = {"chunk": chunk, "questions": kept}
        if i % 25 == 0 or i == len(usable):
            print(f"  [{i}/{len(usable)}] chunks done | {len(per_chunk)} with pairs")

    n_pairs = sum(len(v["questions"]) for v in per_chunk.values())
    rejection_rate = 1 - (n_pairs / total_raw) if total_raw else 0.0
    print(f"\nGenerated {total_raw} raw questions -> kept {n_pairs} pairs "
          f"(rejection rate {rejection_rate:.1%}).")

    # --- Split BY CHUNK ---
    chunk_ids = list(per_chunk.keys())
    random.Random(args.seed).shuffle(chunk_ids)
    n_train_chunks = round(len(chunk_ids) * args.split)
    train_ids = set(chunk_ids[:n_train_chunks])
    eval_ids = set(chunk_ids[n_train_chunks:])
    assert train_ids.isdisjoint(eval_ids), "chunk leakage between train and eval!"

    def to_pairs(ids: set[str]) -> list[dict]:
        rows = []
        for cid in ids:
            rec = per_chunk[cid]
            meta = rec["chunk"]["metadata"] or {}
            for q in rec["questions"]:
                rows.append({
                    "query": q,
                    "passage": rec["chunk"]["content"],
                    "chunk_id": cid,
                    "breed": meta.get("breed"),
                    "page": meta.get("page_number"),
                })
        return rows

    train_pairs = to_pairs(train_ids)
    eval_pairs = to_pairs(eval_ids)

    # --- Write JSONL ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "pairs_train.jsonl"
    eval_path = args.output_dir / "pairs_eval.jsonl"
    for path, rows in ((train_path, train_pairs), (eval_path, eval_pairs)):
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(train_pairs)} train / {len(eval_pairs)} eval pairs to {args.output_dir}")

    # --- Log to MLflow ---
    with mlflow.start_run(run_name="generate-pairs"):
        mlflow.log_params({
            "model": model,
            "questions_per_chunk": args.questions_per_chunk,
            "split_ratio": args.split,
            "seed": args.seed,
            "n_source_chunks": len(usable),
            "min_chunk_chars": args.min_chunk_chars,
            "temperature": args.temperature,
            "limit": args.limit if args.limit is not None else "none",
        })
        mlflow.log_metrics({
            "n_pairs": n_pairs,
            "n_train": len(train_pairs),
            "n_eval": len(eval_pairs),
            "n_chunks_used": len(per_chunk),
            "n_train_chunks": len(train_ids),
            "n_eval_chunks": len(eval_ids),
            "avg_questions_per_chunk": n_pairs / len(per_chunk) if per_chunk else 0.0,
            "rejection_rate": rejection_rate,
        })
        mlflow.log_artifact(str(train_path), artifact_path="dataset")
        mlflow.log_artifact(str(eval_path), artifact_path="dataset")
        print(f"Logged run to MLflow experiment '{args.mlflow_experiment}'.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
