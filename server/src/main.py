"""FastAPI application for RAG service."""
import logging
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import settings
from src.models import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
    ChunkResult,
    DocumentInfo,
    DocumentListResponse,
    DeleteResponse,
    VoiceSessionRequest,
    VoiceSessionResponse,
)
from src.pdf_processor import PDFProcessor
from src.embeddings import EmbeddingGenerator
from src.database import Database
from src.rag_service import RAGService
from src.ingest import ingest_data_directory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize services
pdf_processor = PDFProcessor()
embedding_generator = EmbeddingGenerator()
database = Database()
rag_service = RAGService(embedding_generator, database)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    logger.info("Starting RAG service...")
    try:
        # Check embedding dimension and adjust if needed
        dimension = embedding_generator.get_embedding_dimension()
        logger.info(f"Embedding dimension: {dimension}")
        # Note: Adjusting dimension will delete existing data
        # Uncomment only during initial setup:
        # database.adjust_vector_dimension(dimension)
    except Exception as e:
        logger.warning(f"Could not verify embedding dimension: {e}")

    # Auto-ingest any PDFs sitting in the data/ directory (idempotent).
    try:
        ingested = ingest_data_directory(database=database)
        if ingested:
            logger.info(f"Auto-ingested {len(ingested)} document(s) from data/")
        else:
            logger.info("No new documents to auto-ingest")
    except Exception as e:
        logger.warning(f"Auto-ingest skipped: {e}")

    yield
    logger.info("Shutting down RAG service...")


# Initialize FastAPI app
app = FastAPI(
    title="Ollama RAG Dog Breeds API",
    description="RAG service for querying dog breeds information",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware. Origins come from ALLOWED_ORIGINS (comma-separated);
# defaults to "*" for local dev. A wildcard with credentials is invalid per the
# CORS spec, so credentials are only enabled when origins are explicitly listed.
_origins = [o.strip() for o in settings.allowed_origins.split(",") if o.strip()] or ["*"]
_allow_all = _origins == ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=not _allow_all,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ingest_pdf_file(pdf_path: Path) -> IngestResponse:
    """Process a PDF file, embed its chunks, and store them. Shared by ingest routes."""
    chunks = pdf_processor.process_pdf(str(pdf_path))

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No text could be extracted from the PDF"
        )

    # Create document record
    document_id = database.create_document(pdf_path.name)

    # Generate embeddings
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_generator.generate_embeddings_batch(texts)

    # Insert into database
    chunks_inserted = database.insert_chunks(document_id, chunks, embeddings)

    logger.info(f"Successfully ingested {chunks_inserted} chunks from {pdf_path.name}")

    return IngestResponse(
        message=f"Successfully ingested {pdf_path.name}",
        chunks_processed=chunks_inserted,
        document_id=document_id
    )


@app.get("/")
async def root():
    """Service metadata and available endpoints."""
    return {
        "service": "ollama-rag-dog-breeds",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": ["/health", "/ingest", "/ingest/upload", "/query", "/search", "/documents"],
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "ollama-rag-dog-breeds",
        "version": "0.1.0"
    }


@app.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_pdf(request: IngestRequest):
    """
    Ingest a PDF file into the vector database.
    
    Args:
        request: IngestRequest with pdf_path
        
    Returns:
        IngestResponse with processing results
    """
    try:
        pdf_path = Path(request.pdf_path)
        
        # Validate PDF exists
        if not pdf_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"PDF file not found: {pdf_path}"
            )
        
        logger.info(f"Ingesting PDF: {pdf_path}")
        return _ingest_pdf_file(pdf_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting PDF: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error ingesting PDF: {str(e)}"
        )


@app.post("/ingest/upload", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_upload(file: UploadFile = File(...)):
    """
    Ingest a PDF uploaded directly via multipart form-data.

    Args:
        file: The uploaded PDF file

    Returns:
        IngestResponse with processing results
    """
    filename = file.filename or "upload.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported"
        )

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / Path(filename).name
    try:
        with tmp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Ingesting uploaded PDF: {filename}")
        return _ingest_pdf_file(tmp_path)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting uploaded PDF: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error ingesting uploaded PDF: {str(e)}"
        )
    finally:
        await file.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """List all ingested documents with their chunk counts."""
    try:
        documents = database.list_documents()
        return DocumentListResponse(
            documents=[DocumentInfo(**doc) for doc in documents],
            total=len(documents),
        )
    except Exception as e:
        logger.error(f"Error listing documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing documents: {str(e)}"
        )


@app.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document(document_id: str):
    """Delete a document and all of its chunks."""
    try:
        deleted = database.delete_document(document_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {document_id}"
            )
        return DeleteResponse(
            message="Document deleted",
            document_id=document_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document: {str(e)}"
        )


@app.delete("/cache")
async def clear_cache():
    """Drop all cached answers (e.g. after changing the knowledge base manually)."""
    try:
        removed = database.clear_cache()
        return {"message": "Answer cache cleared", "removed": removed}
    except Exception as e:
        logger.error(f"Error clearing cache: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error clearing cache: {str(e)}"
        )


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system with a natural language question.
    
    Args:
        request: QueryRequest with query and parameters
        
    Returns:
        QueryResponse with relevant chunks and generated answer
    """
    try:
        logger.info(f"Processing query: {request.query}")
        
        result = rag_service.query(
            query=request.query,
            top_k=request.top_k,
            generate_answer=True
        )
        
        # Convert to response model
        chunk_results = [
            ChunkResult(
                id=chunk["id"],
                content=chunk["content"],
                similarity_score=chunk.get("similarity_score"),
                metadata=chunk.get("metadata") if request.include_metadata else None
            )
            for chunk in result["chunks"]
        ]
        
        return QueryResponse(
            query=result["query"],
            chunks=chunk_results,
            answer=result.get("answer"),
            cached=result.get("cached", False),
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


@app.post("/search", response_model=SearchResponse)
async def search_vectors(request: SearchRequest):
    """
    Perform vector similarity search without generating an answer.
    
    Args:
        request: SearchRequest with query and parameters
        
    Returns:
        SearchResponse with search results
    """
    try:
        logger.info(f"Performing search: {request.query}")
        
        result = rag_service.search(
            query=request.query,
            top_k=request.top_k,
            threshold=request.threshold
        )
        
        # Convert to response model
        chunk_results = [
            ChunkResult(
                id=chunk["id"],
                content=chunk["content"],
                similarity_score=chunk.get("similarity_score"),
                metadata=chunk.get("metadata")
            )
            for chunk in result["results"]
        ]
        
        return SearchResponse(
            query=result["query"],
            results=chunk_results,
            total_results=result["total_results"]
        )
    
    except Exception as e:
        logger.error(f"Error performing search: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error performing search: {str(e)}"
        )


def _livekit_http_url() -> str:
    """LiveKit server admin URL (http/https), derived from the ws(s) client URL."""
    url = settings.livekit_url
    if url.startswith("wss://"):
        return "https://" + url[len("wss://"):]
    if url.startswith("ws://"):
        return "http://" + url[len("ws://"):]
    return url


@app.post("/api/voice/session", response_model=VoiceSessionResponse)
async def create_voice_session(request: VoiceSessionRequest):
    """
    Mint a LiveKit access token so the frontend can join a voice room.

    The LiveKit voice agent joins the room automatically and answers questions
    grounded in the ingested book via RAG.
    """
    if not settings.livekit_api_key or not settings.livekit_api_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LiveKit is not configured. Set LIVEKIT_API_KEY and LIVEKIT_API_SECRET.",
        )

    from livekit import api

    user_id = request.user_id or f"user-{uuid.uuid4().hex[:8]}"
    room_name = f"voice-{user_id}"

    token = (
        api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
        .with_identity(user_id)
        .with_name(user_id)
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )

    logger.info(f"Created voice session for {user_id} in room {room_name}")
    return VoiceSessionResponse(
        room_name=room_name,
        token=token,
        url=settings.livekit_url,
    )


@app.delete("/api/voice/session/{room_name}")
async def end_voice_session(room_name: str):
    """Best-effort teardown of a LiveKit room when the user disconnects."""
    if not settings.livekit_api_key or not settings.livekit_api_secret:
        return {"message": "LiveKit not configured; nothing to clean up"}

    from livekit import api

    lkapi = api.LiveKitAPI(
        url=_livekit_http_url(),
        api_key=settings.livekit_api_key,
        api_secret=settings.livekit_api_secret,
    )
    try:
        await lkapi.room.delete_room(api.DeleteRoomRequest(room=room_name))
        logger.info(f"Deleted voice room {room_name}")
    except Exception as e:
        logger.warning(f"Could not delete room {room_name}: {e}")
    finally:
        await lkapi.aclose()

    return {"message": f"Ended voice session {room_name}"}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )

