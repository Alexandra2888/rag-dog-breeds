# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
# The knowledge-base PDF(s) must ship in the image so startup auto-ingest works.
COPY data/ ./data/
# Fine-tuned embedding model, baked in. ST_MODEL_PATH points here so the app
# embeds locally with bge (no external embeddings API). ~440 MB.
COPY finetune/models/bge-base-dogbreeds ./finetune/models/bge-base-dogbreeds

# Install runtime deps, then the local-embedder stack: CPU-only torch (small —
# avoids the multi-GB CUDA build) followed by sentence-transformers. Order
# matters: install CPU torch first so sentence-transformers doesn't pull CUDA torch.
RUN uv pip install --system -e . \
 && uv pip install --system --index-url https://download.pytorch.org/whl/cpu torch \
 && uv pip install --system "sentence-transformers>=3.0"

# The bge weights are baked in — never reach out to HuggingFace at runtime
# (faster, deterministic, works if the network blips mid-demo).
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Expose API port (for FastAPI)
EXPOSE 8000

# Default to the FastAPI API (the primary service). Respects $PORT (Render) and
# falls back to 8000 locally. The voice agent overrides this command
# (docker-compose `livekit-agent`, fly.toml `agent` process).
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

