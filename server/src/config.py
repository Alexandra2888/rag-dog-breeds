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

