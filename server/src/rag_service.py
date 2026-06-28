"""RAG service for query processing."""
import logging
import re
from typing import List, Dict, Any, Optional
import ollama

from src.embeddings import EmbeddingGenerator
from src.database import Database
from src.config import settings

logger = logging.getLogger(__name__)


def normalize_query(query: str) -> str:
    """Canonical form used as the cache key.

    Lower-cases, collapses whitespace and trims surrounding punctuation so trivial
    variants of the same question ("What's a Pug?", "what's a pug") share a cache
    entry. This also absorbs minor speech-to-text punctuation noise on the voice
    path.
    """
    return re.sub(r"\s+", " ", query.strip().lower()).strip(" \t\n.?!,;:")


class RAGService:
    """Handles RAG queries and context retrieval."""
    
    def __init__(
        self,
        embedding_generator: EmbeddingGenerator = None,
        database: Database = None,
        chat_model: str = None
    ):
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.database = database or Database()
        self.chat_model = chat_model or settings.ollama_chat_model
        # Initialize Ollama client
        try:
            self.client = ollama.Client(host=settings.ollama_base_url)
        except (AttributeError, TypeError):
            self.client = None
            import os
            host = settings.ollama_base_url.replace('http://', '').replace('https://', '')
            os.environ['OLLAMA_HOST'] = host
    
    def query(
        self,
        query: str,
        top_k: int = 5,
        generate_answer: bool = True,
        document_id: Optional[str] = None,
        mode: str = "text",
    ) -> Dict[str, Any]:
        """
        Process a RAG query.

        Args:
            query: User's query
            top_k: Number of relevant chunks to retrieve
            generate_answer: Whether to generate an answer using LLM
            document_id: Filter by document ID (optional)
            mode: Answer style / cache namespace ("text" or "voice"). Voice
                answers are short and spoken, so they are cached separately.

        Returns:
            Dictionary with query, chunks, optional answer, and a `cached` flag.
        """
        logger.info(f"Processing query: {query}")

        # Answer cache: a repeated question skips embedding, search and the LLM
        # entirely. Only cache whole-corpus answers (document-scoped queries are
        # rare and would complicate the key).
        cacheable = generate_answer and document_id is None
        query_norm = normalize_query(query)
        if cacheable and query_norm:
            cached = self.database.get_cached_answer(query_norm, mode, top_k)
            if cached is not None:
                logger.info(f"Answer cache HIT ({mode}) for: {query!r}")
                return {
                    "query": query,
                    "chunks": cached["chunks"],
                    "answer": cached["answer"],
                    "cached": True,
                }

        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(
            query, input_type="search_query"
        )

        # Perform hybrid (vector + keyword) similarity search
        chunks = self.database.similarity_search(
            query_embedding,
            top_k=top_k,
            document_id=document_id,
            query_text=query
        )

        logger.info(f"Retrieved {len(chunks)} relevant chunks")

        result = {
            "query": query,
            "chunks": chunks,
            "cached": False,
        }

        # Generate answer if requested
        if generate_answer and chunks:
            answer = self._generate_answer(query, chunks, mode=mode)
            result["answer"] = answer
            if cacheable and query_norm:
                self.database.put_cached_answer(
                    query_norm, mode, top_k, query, answer, chunks
                )

        return result

    def _generate_answer(
        self, query: str, chunks: List[Dict[str, Any]], mode: str = "text"
    ) -> str:
        """
        Generate an answer using retrieved context.

        Args:
            query: User's query
            chunks: Retrieved context chunks
            mode: "text" for a full answer, "voice" for a short spoken one.

        Returns:
            Generated answer
        """
        # Build context from chunks
        context = "\n\n".join([
            f"[Source: {chunk.get('metadata', {}).get('source', 'unknown')}, "
            f"Page {chunk.get('metadata', {}).get('page_number', 'N/A')}]\n"
            f"{chunk['content']}"
            for chunk in chunks
        ])

        # Create prompt — voice answers must be short since they are spoken aloud.
        if mode == "voice":
            prompt = f"""You are a friendly voice assistant answering questions about dog breeds.
Answer the user's question in ONE or TWO short, conversational sentences, based only on the context below.
If the answer is not in the context, say you don't know rather than guessing.

Context:
{context}

Question: {query}

Answer:"""
        else:
            prompt = f"""Based on the following context from a dog breeds book, answer the user's question.
If the answer cannot be found in the context, say so.

Context:
{context}

Question: {query}

Answer:"""

        try:
            if self.client:
                response = self.client.generate(model=self.chat_model, prompt=prompt)
            else:
                response = ollama.generate(model=self.chat_model, prompt=prompt)
            return response["response"]
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: Optional[float] = None,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform vector similarity search without generating an answer.
        
        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            document_id: Filter by document ID (optional)
            
        Returns:
            Dictionary with query and results
        """
        logger.info(f"Performing search: {query}")
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(
            query, input_type="search_query"
        )
        
        # Perform hybrid (vector + keyword) similarity search
        results = self.database.similarity_search(
            query_embedding,
            top_k=top_k,
            threshold=threshold,
            document_id=document_id,
            query_text=query
        )
        
        return {
            "query": query,
            "results": results,
            "total_results": len(results)
        }

