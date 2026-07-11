"""Roll up online-eval traffic into MLflow so quality drift is a trend line.

Reads the ``online_eval`` (and ``feedback``) Postgres tables over a recent window,
computes aggregate quality metrics, and logs them to a dedicated MLflow experiment
(``online-eval``) — the same tracking store the fine-tuning pipeline already uses.
Run it on a schedule (Fly cron / GitHub Actions) or by hand; each run is one point
in the drift time series.

Usage
-----
    uv run python -m scripts.aggregate_online_eval                 # last 24h
    uv run python -m scripts.aggregate_online_eval --window-hours 6
    uv run python -m scripts.aggregate_online_eval --no-mlflow     # print only

Needs ``DATABASE_URL`` (and ``MLFLOW_TRACKING_URI`` if not using the local
sqlite store) in the environment.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make ``src`` and ``finetune`` importable when run from server/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import Database  # noqa: E402
from finetune.mlflow_utils import setup_mlflow  # noqa: E402

EXPERIMENT = "online-eval"


def _mean(values) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _rate(flags) -> float | None:
    vals = [bool(v) for v in flags]
    return sum(vals) / len(vals) if vals else None


def compute_window_metrics(rows: list[dict], feedback: list[dict]) -> dict:
    """Aggregate one window of online_eval + feedback rows into MLflow metrics."""
    n = len(rows)
    sampled = [r for r in rows if r.get("judge_sampled")]
    neg = [1 for f in feedback if f.get("rating", 0) < 0]
    metrics = {
        "n_queries": float(n),
        "refusal_rate": _rate([r["refused"] for r in rows]),
        "cache_hit_rate": _rate([r["cached"] for r in rows]),
        "empty_retrieval_rate": _rate([r["num_chunks"] == 0 for r in rows]),
        "sim_max_mean": _mean([r["sim_max"] for r in rows]),
        "sim_mean_mean": _mean([r["sim_mean"] for r in rows]),
        "answer_len_mean": _mean([r["answer_len"] for r in rows]),
        "n_judge_sampled": float(len(sampled)),
        "judge_faithfulness_mean": _mean([r["judge_faithfulness"] for r in sampled]),
        "judge_relevancy_mean": _mean([r["judge_relevancy"] for r in sampled]),
        "n_feedback": float(len(feedback)),
        "feedback_neg_rate": (len(neg) / len(feedback)) if feedback else None,
    }
    # Drop metrics with no data so MLflow doesn't chart spurious zeros.
    return {k: v for k, v in metrics.items() if v is not None}


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate online-eval → MLflow")
    parser.add_argument("--window-hours", type=int, default=24, help="lookback window")
    parser.add_argument("--mlflow-experiment", default=EXPERIMENT)
    parser.add_argument("--mlflow-uri", default=None, help="override tracking URI")
    parser.add_argument("--no-mlflow", action="store_true", help="print metrics only")
    args = parser.parse_args()

    db = Database()
    rows = db.fetch_online_eval_window(args.window_hours)
    feedback = db.fetch_feedback_window(args.window_hours)

    if not rows:
        print(f"No online_eval rows in the last {args.window_hours}h — nothing to aggregate.")
        return 0

    metrics = compute_window_metrics(rows, feedback)

    print(f"\n=== Online-eval window ({args.window_hours}h, {len(rows)} queries) ===")
    for k, v in metrics.items():
        print(f"  {k:26} {v:.3f}")

    if args.no_mlflow:
        return 0

    import mlflow

    setup_mlflow(args.mlflow_experiment, args.mlflow_uri)
    now = datetime.now(timezone.utc)
    # Epoch-hour step so repeated runs chart as a time series in MLflow.
    step = int(now.timestamp() // 3600)
    with mlflow.start_run(run_name=f"window_{now:%Y%m%dT%H}"):
        mlflow.log_params({
            "window_hours": args.window_hours,
            "window_end_utc": now.isoformat(timespec="seconds"),
        })
        mlflow.log_metrics(metrics, step=step)
    print(f"\nLogged to MLflow experiment '{args.mlflow_experiment}' (step={step}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
