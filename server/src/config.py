"""Configuration management for the RAG service."""
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/ragdb",
        alias="DATABASE_URL"
    )
    
    # Ollama
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        alias="OLLAMA_BASE_URL"
    )
    ollama_embedding_model: str = Field(
        default="nomic-embed-text",
        alias="OLLAMA_EMBEDDING_MODEL"
    )
    ollama_chat_model: str = Field(
        default="llama3.2",
        alias="OLLAMA_CHAT_MODEL"
    )
    
    # Inference provider — where the LLM + embeddings run.
    #   "ollama"  -> local Ollama (default; for local dev)
    #   "openai"  -> any OpenAI-compatible API (e.g. Google Gemini's free
    #                endpoint) — used for the free cloud deploy.
    inference_provider: str = Field(default="ollama", alias="INFERENCE_PROVIDER")
    # OpenAI-compatible endpoint + key (only used when provider == "openai").
    # Gemini: https://generativelanguage.googleapis.com/v1beta/openai/
    inference_base_url: str = Field(default="", alias="INFERENCE_BASE_URL")
    inference_api_key: str = Field(default="", alias="INFERENCE_API_KEY")
    # Model names for the "openai" provider (Gemini defaults; both free).
    # text-embedding-004 is 768-dim, matching the DB schema.
    inference_chat_model: str = Field(
        default="gemini-2.0-flash", alias="INFERENCE_CHAT_MODEL"
    )
    inference_embedding_model: str = Field(
        default="gemini-embedding-001", alias="INFERENCE_EMBEDDING_MODEL"
    )
    # Output dimension for the "openai" embedding provider. gemini-embedding-001
    # defaults to 3072 but supports 768 to match the DB schema (vector(768)).
    # Set 0 to omit the param (for providers with a fixed dimension).
    inference_embedding_dim: int = Field(default=768, alias="INFERENCE_EMBEDDING_DIM")

    # Chunking
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    
    # API
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    # Comma-separated list of allowed CORS origins. Defaults to "*" for local
    # dev; set to your frontend origin(s) in production, e.g.
    # ALLOWED_ORIGINS=https://your-app.vercel.app
    allowed_origins: str = Field(default="*", alias="ALLOWED_ORIGINS")
    
    # LiveKit
    livekit_url: str = Field(
        default="ws://localhost:7880",
        alias="LIVEKIT_URL"
    )
    livekit_api_key: str = Field(
        default="",
        alias="LIVEKIT_API_KEY"
    )
    livekit_api_secret: str = Field(
        default="",
        alias="LIVEKIT_API_SECRET"
    )
    livekit_agent_port: int = Field(
        default=8080,
        alias="LIVEKIT_AGENT_PORT"
    )
    
    # Optional: OpenAI API key for STT (if not using local Whisper)
    openai_api_key: str = Field(
        default="",
        alias="OPENAI_API_KEY"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

