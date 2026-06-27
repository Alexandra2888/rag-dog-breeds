# Ollama RAG Dog Breeds

A Retrieval-Augmented Generation (RAG) service for querying dog breeds information from PDF documents using Ollama, PostgreSQL with pgvector, and FastAPI.

## Architecture

This service processes PDF documents, extracts text, generates embeddings using Ollama, stores them in PostgreSQL with pgvector, and provides a REST API for querying. It's designed to be containerized and called by a Golang backend for use in Next.js and React Native applications.

```
PDF → Text Extraction → Chunking → Ollama Embeddings → PostgreSQL (pgvector) → FastAPI API
                                                                                      ↓
                                                                              Golang Backend
                                                                                      ↓
                                                                         Next.js/React Native
```

## Features

- PDF text extraction and intelligent chunking
- Vector embeddings using Ollama
- PostgreSQL with pgvector for efficient similarity search
- FastAPI REST API with automatic documentation
- Docker containerization for easy deployment
- Configurable chunking and embedding models

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) - Python package manager
- Docker and Docker Compose
- [Ollama](https://ollama.ai/) installed and running
- Ollama models:
  - Embedding model: `nomic-embed-text` (or similar)
  - Chat model: `llama3.2` (or similar)

### Installing Ollama Models

```bash
# Install embedding model
ollama pull nomic-embed-text

# Install chat model
ollama pull llama3.2
```

## Setup

### 1. Clone and Navigate

```bash
cd ollama-rag-dog-breeds
```

### 2. Install Dependencies with uv

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv pip install -e .
```

### 3. Configure Environment

Copy the example environment file and adjust as needed:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ragdb
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_CHAT_MODEL=llama3.2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### 4. Start Services with Docker Compose

```bash
docker-compose up -d
```

This will start:

- PostgreSQL with pgvector extension on port 5432
- FastAPI service on port 8000

### 5. Ingest the PDF (knowledge base)

The PDF in `data/` is **ingested automatically** the first time the API starts
(see the `lifespan` startup hook). It is idempotent — already-ingested files
are skipped, so restarts are cheap.

To ingest manually (or before running just the voice agent), run:

```bash
uv run python -m src.ingest
```

This scans `data/*.pdf`, chunks each PDF, embeds the chunks with Ollama, and
stores them in pgvector. You can also ingest a specific PDF over HTTP:

```bash
# By server-side path
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"pdf_path": "data/The-Complete-Dog-Breed-Book-Choose-the-Perfect-Dog-for-You-New-Edition.pdf"}'

# Or upload directly
curl -X POST "http://localhost:8000/ingest/upload" -F "file=@./data/your.pdf"
```

## API Endpoints

### Health Check

```bash
GET /health
```

Returns service health status.

### Ingest PDF

```bash
POST /ingest
Content-Type: application/json

{
  "pdf_path": "path/to/document.pdf"
}
```

Processes a PDF file, extracts text, generates embeddings, and stores them in the database.

**Response:**

```json
{
  "message": "Successfully ingested document.pdf",
  "chunks_processed": 1234,
  "document_id": "uuid-here"
}
```

### Query RAG

```bash
POST /query
Content-Type: application/json

{
  "query": "What are the characteristics of a Golden Retriever?",
  "top_k": 5,
  "include_metadata": true
}
```

Performs a RAG query: retrieves relevant chunks and generates an answer.

**Response:**

```json
{
  "query": "What are the characteristics of a Golden Retriever?",
  "chunks": [
    {
      "id": "uuid",
      "content": "Golden Retrievers are...",
      "similarity_score": 0.85,
      "metadata": {
        "source": "document.pdf",
        "page_number": 42
      }
    }
  ],
  "answer": "Golden Retrievers are friendly, intelligent dogs..."
}
```

### Upload PDF (multipart)

```bash
POST /ingest/upload
Content-Type: multipart/form-data
```

Upload a PDF directly instead of referencing a server-side path:

```bash
curl -X POST "http://localhost:8000/ingest/upload" \
  -F "file=@./data/my-document.pdf"
```

Returns the same `IngestResponse` as `/ingest`.

### List Documents

```bash
GET /documents
```

Returns all ingested documents with their chunk counts.

**Response:**

```json
{
  "documents": [
    {
      "id": "uuid",
      "document_name": "document.pdf",
      "created_at": "2026-06-27T10:00:00",
      "chunk_count": 1234
    }
  ],
  "total": 1
}
```

### Delete Document

```bash
DELETE /documents/{document_id}
```

Deletes a document and all of its chunks. Returns `404` if the document does not exist.

### Vector Search

```bash
POST /search
Content-Type: application/json

{
  "query": "loyal companion dogs",
  "top_k": 10,
  "threshold": 0.7
}
```

Performs vector similarity search without generating an answer.

**Response:**

```json
{
  "query": "loyal companion dogs",
  "results": [
    {
      "id": "uuid",
      "content": "These dogs are known for...",
      "similarity_score": 0.92,
      "metadata": {...}
    }
  ],
  "total_results": 10
}
```

## API Documentation

Interactive API documentation is available at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Development

### Running Locally (without Docker)

1. Start PostgreSQL with pgvector locally
2. Ensure Ollama is running
3. Set up environment variables in `.env`
4. Run the service:

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### Project Structure

```
ollama-rag-dog-breeds/
├── pyproject.toml          # uv dependencies
├── .env.example            # Environment template
├── Dockerfile              # Python service container
├── docker-compose.yml      # Full stack (PostgreSQL + API)
├── README.md               # This file
├── data/                   # PDF documents directory
│   └── *.pdf
└── src/
    ├── __init__.py
    ├── main.py             # FastAPI application
    ├── config.py           # Configuration management
    ├── pdf_processor.py    # PDF extraction and chunking
    ├── embeddings.py       # Ollama embedding generation
    ├── database.py         # PostgreSQL/pgvector operations
    ├── rag_service.py      # RAG query logic
    ├── models.py           # Pydantic models for API
    ├── ingest.py           # Auto-ingest PDFs from data/ (CLI + startup hook)
    ├── livekit_agent.py    # LiveKit 1.x voice agent (Ollama LLM + RAG, OpenAI STT/TTS)
    └── livekit_server.py   # LiveKit agent worker entrypoint
```

## Voice Agent with LiveKit (Optional)

This service includes a LiveKit voice agent that uses your local Ollama for LLM and RAG queries. The agent provides real-time voice interactions where users can speak questions about dog breeds and receive spoken answers.

### Architecture

```
User Voice → LiveKit Server → LiveKit Agent → RAG Service → Ollama (LLM + Embeddings) → PostgreSQL
                                                                    ↓
                                                              Document Knowledge Base
```

> Built on **livekit-agents 1.x** (`AgentSession` / `Agent`). The agent grounds
> every answer in the ingested book via RAG, injected through the
> `Agent.on_user_turn_completed` hook.

The agent uses:

- **Ollama** for the LLM (via the OpenAI-compatible endpoint at `http://localhost:11434/v1`)
- **Ollama + pgvector** for RAG retrieval
- **OpenAI** for speech-to-text and text-to-speech (set `OPENAI_API_KEY`)
- **Silero VAD** (bundled) for turn detection

### Prerequisites for Voice Agent

- Ollama running with the `nomic-embed-text` and `llama3.2` models
- Postgres (pgvector) running and the PDF ingested (`uv run python -m src.ingest`)
- `OPENAI_API_KEY` set in `.env` (required for STT + TTS)
- LiveKit credentials in `.env` — either LiveKit Cloud (`wss://...livekit.cloud`)
  or the local `livekit-server` from docker-compose

### Talk to the agent in your terminal (easiest)

`console` mode uses your computer's microphone and speakers directly — no
frontend or LiveKit room needed:

```bash
# 1. Make sure Postgres + Ollama are up and the PDF is ingested
docker compose up -d postgres
uv run python -m src.ingest

# 2. Download the agent's model files once (Silero VAD, etc.)
uv run python -m src.livekit_agent download-files

# 3. Start talking
uv run python -m src.livekit_agent console
```

Speak a question like *"What's a good apartment dog?"* and the agent answers
out loud, grounded in the book.

### Run as a worker for LiveKit Cloud / a frontend

To serve real users (via the LiveKit Agents Playground or your own Next.js/RN
app), run the worker in `dev` (hot-reload) or `start` (production) mode. It
registers with the LiveKit project in your `.env` and joins rooms automatically:

```bash
uv run python -m src.livekit_agent dev      # local development
# or, in Docker (production):
docker compose up -d livekit-agent
```

Then open <https://agents-playground.livekit.io>, connect it to your LiveKit
project, and start a voice session — the agent joins and responds.

### Go Backend Integration for Voice

Your Go backend needs to:

1. **Create LiveKit rooms** when users want to start voice chat
2. **Generate access tokens** for users and agents
3. **Return connection info** to your Next.js frontend

Example Go code:

```go
import (
    "github.com/livekit/protocol/auth"
    "github.com/livekit/server-sdk-go"
)

// Create a LiveKit room
func CreateVoiceSession(userID string) (*Session, error) {
    roomClient := lksdk.NewRoomServiceClient(
        "http://localhost:7880",
        "devkey",
        "devsecret",
    )

    room, err := roomClient.CreateRoom(ctx, &livekit.CreateRoomRequest{
        Name: fmt.Sprintf("user-%s", userID),
    })

    // Generate user token
    at := auth.NewAccessToken("devkey", "devsecret")
    at.AddGrant(&auth.VideoGrant{
        RoomJoin: true,
        Room: room.Name,
    }).SetIdentity(userID)

    token, _ := at.ToJWT()

    return &Session{
        RoomName: room.Name,
        Token: token,
        URL: "ws://localhost:7880",
    }, nil
}
```

### Next.js Frontend Integration

```typescript
import { Room } from "livekit-client";

const room = new Room();
await room.connect("ws://localhost:7880", userToken);

// Enable microphone
await room.localParticipant.setMicrophoneEnabled(true);

// The agent will automatically join and respond to voice
```

## Integration with Golang Backend (Text API)

This Python service is designed to be called by a Golang backend. Example integration:

```go
// Example Go client
type RAGClient struct {
    baseURL string
    client  *http.Client
}

func (c *RAGClient) Query(query string) (*QueryResponse, error) {
    reqBody := map[string]interface{}{
        "query": query,
        "top_k": 5,
    }
    // ... HTTP request to http://rag-api:8000/query
}
```

## Environment Variables

| Variable                 | Description                   | Default                                               |
| ------------------------ | ----------------------------- | ----------------------------------------------------- |
| `DATABASE_URL`           | PostgreSQL connection string  | `postgresql://postgres:postgres@localhost:5432/ragdb` |
| `OLLAMA_BASE_URL`        | Ollama API URL                | `http://localhost:11434`                              |
| `OLLAMA_EMBEDDING_MODEL` | Ollama embedding model name   | `nomic-embed-text`                                    |
| `OLLAMA_CHAT_MODEL`      | Ollama chat model name        | `llama3.2`                                            |
| `CHUNK_SIZE`             | Text chunk size in characters | `1000`                                                |
| `CHUNK_OVERLAP`          | Overlap between chunks        | `200`                                                 |
| `API_HOST`               | API host address              | `0.0.0.0`                                             |
| `API_PORT`               | API port                      | `8000`                                                |
| `LIVEKIT_URL`            | LiveKit WebSocket URL         | `ws://localhost:7880`                                 |
| `LIVEKIT_API_KEY`        | LiveKit API key               | `devkey` (dev)                                        |
| `LIVEKIT_API_SECRET`     | LiveKit API secret            | `devsecret` (dev)                                     |
| `OPENAI_API_KEY`         | OpenAI API key (for STT)      | Optional, only needed for Whisper STT                 |

## Troubleshooting

### Ollama Connection Issues

If running in Docker, ensure Ollama is accessible:

- On Linux: Use `host.docker.internal` or the host's IP
- Update `OLLAMA_BASE_URL` in docker-compose.yml if needed

### Vector Dimension Mismatch

If you get dimension errors, the embedding model dimension may differ. Check the model's dimension and update the database schema:

```python
# In src/main.py, uncomment during initial setup:
database.adjust_vector_dimension(actual_dimension)
```

### Database Connection Issues

Ensure PostgreSQL is running and accessible:

```bash
docker-compose ps
docker-compose logs postgres
```

## License

MIT
