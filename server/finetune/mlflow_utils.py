"""Shared MLflow tracking setup for the fine-tuning pipeline.

Every run in the pipeline (pair generation, retrieval eval, fine-tuning) logs to
one tracking store so params/metrics/artifacts sit side by side. MLflow >=3 put
the plain file store in maintenance mode, so we default to a local SQLite backend.

Precedence for the tracking URI: explicit ``uri`` arg > ``MLFLOW_TRACKING_URI``
env var > local ``sqlite:///mlflow.db`` (resolved against the current directory,
i.e. ``server/`` when run as documented).
"""
from __future__ import annotations

import os
from pathlib import Path


def resolve_tracking_uri(uri: str | None = None) -> str:
    if uri:
        return uri
    if os.getenv("MLFLOW_TRACKING_URI"):
        return os.environ["MLFLOW_TRACKING_URI"]
    return f"sqlite:///{Path('mlflow.db').resolve()}"


def setup_mlflow(experiment: str, uri: str | None = None) -> str:
    """Point MLflow at the shared store and select ``experiment``.

    Returns the resolved tracking URI (handy for logging / --backend-store-uri).
    """
    import mlflow

    tracking_uri = resolve_tracking_uri(uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    return tracking_uri
