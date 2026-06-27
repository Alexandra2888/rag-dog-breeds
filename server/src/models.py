"""Pydantic models for API requests and responses."""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, Field

class IngestRequest(BaseModel):
    """Request model for PDF ingestion."""
    pdf_path: str = Field(..., description="Path to the PDF file to ingest")


class IngestResponse(BaseModel):
    """Response model for PDF ingestion."""
    message: str
    chunks_processed: int
    document_id: str


class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    query: str = Field(..., description="The user's query")
    top_k: int = Field(default=5, description="Number of relevant chunks to retrieve")
    include_metadata: bool = Field(default=True, description="Include metadata in response")


class SearchRequest(BaseModel):
    """Request model for vector similarity search."""
    query: str = Field(..., description="The search query")
    top_k: int = Field(default=5, description="Number of results to return")
    threshold: Optional[float] = Field(default=None, description="Minimum similarity threshold")


class ChunkResult(BaseModel):
    """Model for a retrieved chunk."""
    id: str
    content: str
    similarity_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class DocumentInfo(BaseModel):
    """Model for an ingested document."""
    id: str
    document_name: str
    created_at: Optional[str] = None
    chunk_count: int = 0


class DocumentListResponse(BaseModel):
    """Response model for listing documents."""
    documents: List[DocumentInfo]
    total: int


class DeleteResponse(BaseModel):
    """Response model for document deletion."""
    message: str
    document_id: str


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    query: str
    chunks: List[ChunkResult]
    answer: Optional[str] = None


class SearchResponse(BaseModel):
    """Response model for vector search."""
    query: str
    results: List[ChunkResult]
    total_results: int


class VoiceSessionRequest(BaseModel):
    """Request model for creating a LiveKit voice session."""
    user_id: Optional[str] = Field(default=None, description="Stable identity for the user")


class VoiceSessionResponse(BaseModel):
    """Connection info the frontend needs to join a LiveKit room."""
    room_name: str
    token: str
    url: str

