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

# Install Python dependencies with uv (runtime deps only; dev deps excluded)
RUN uv pip install --system -e .

# Expose API port (for FastAPI)
EXPOSE 8000

# Default to the FastAPI API (the primary service). Respects $PORT (Render) and
# falls back to 8000 locally. The voice agent overrides this command
# (docker-compose `livekit-agent`, fly.toml `agent` process).
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

