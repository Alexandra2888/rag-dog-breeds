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

# Default to the LiveKit voice-agent worker (compose / LiveKit Cloud may override).
# For the FastAPI service, override with: uvicorn src.main:app --host 0.0.0.0 --port 8000
CMD ["python", "-m", "src.livekit_agent", "start"]

