"""RAG service for query processing."""
import logging
import re
from typing import List, Dict, Any, Optional
import ollama

from src.embeddings import EmbeddingGenerator
from src.database import Database
from src.config import settings
from src.guardrails import (
    GuardrailContext,
    build_input_pipeline,
    build_output_pipeline,
    canned_refusal,
)
from src.guardrails.base import Action

logger = logging.getLogger(__name__)


def _current_request_id() -> Optional[str]:
    """Read the request_id bound by the FastAPI middleware (copied into worker
    threads by asyncio.to_thread on the voice path)."""
    try:
        import structlog
        return structlog.contextvars.get_contextvars().get("request_id")
    except Exception:
        return None


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
        chat_model: str = None,
        input_pipeline=None,
        output_pipeline=None,
    ):
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.database = database or Database()
        self.provider = settings.inference_provider.lower()
        self.openai_client = None
        self.client = None

        if self.provider == "openai":
            # Any OpenAI-compatible chat API (e.g. Google Gemini's free endpoint).
            from openai import OpenAI
            self.chat_model = chat_model or settings.inference_chat_model
            self.openai_client = OpenAI(
                base_url=settings.inference_base_url or None,
                api_key=settings.inference_api_key or "not-needed",
            )
        else:
            self.chat_model = chat_model or settings.ollama_chat_model
            try:
                self.client = ollama.Client(host=settings.ollama_base_url)
            except (AttributeError, TypeError):
                import os
                host = settings.ollama_base_url.replace('http://', '').replace('https://', '')
                os.environ['OLLAMA_HOST'] = host

        # Guardrail pipelines (injectable for tests). Built from settings; empty
        # and no-op when guardrails are disabled.
        self.input_pipeline = input_pipeline or build_input_pipeline()
        self.output_pipeline = output_pipeline or build_output_pipeline(
            self.embedding_generator, self.openai_client
        )
    
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
        request_id = _current_request_id()
        guardrail_results = []

        # INPUT guardrail: screen the query before any cache/embed/generate. In
        # enforce mode a blocked/refused input short-circuits to a canned refusal;
        # in shadow mode we only record the decision and continue.
        in_res = self.input_pipeline.run(
            GuardrailContext(query=query, mode=mode, top_k=top_k, request_id=request_id)
        )
        guardrail_results.append(in_res)
        if self.input_pipeline.enforce and in_res.final_action in (Action.BLOCK, Action.REFUSE):
            logger.info(f"Guardrail blocked input ({in_res.final_action.value}): {query!r}")
            return {
                "query": query,
                "chunks": [],
                "answer": canned_refusal(mode),
                "cached": False,
                "guardrail_triggered": True,
                "guardrail_action": in_res.final_action.value,
                "_guardrail_results": guardrail_results,
            }

        # Answer cache: a repeated question skips embedding, search and the LLM
        # entirely. Only cache whole-corpus answers (document-scoped queries are
        # rare and would complicate the key).
        cacheable = generate_answer and document_id is None
        query_norm = normalize_query(query)
        if cacheable and query_norm:
            cached = self.database.get_cached_answer(query_norm, mode, top_k)
            if cached is not None:
                logger.info(f"Answer cache HIT ({mode}) for: {query!r}")
                # A cached answer must still pass the OUTPUT guardrail — it may
                # have been stored before enforcement or a threshold change.
                safe_answer, out_res = self._guard_output(
                    query, cached["chunks"], cached["answer"], mode, top_k, request_id
                )
                guardrail_results.append(out_res)
                triggered, action = self._surface(guardrail_results)
                return {
                    "query": query,
                    "chunks": cached["chunks"],
                    "answer": safe_answer,
                    "cached": True,
                    "guardrail_triggered": triggered,
                    "guardrail_action": action,
                    "_guardrail_results": guardrail_results,
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
            "_guardrail_results": guardrail_results,
        }

        # Generate answer if requested
        if generate_answer and chunks:
            answer = self._generate_answer(query, chunks, mode=mode)
            safe_answer, out_res = self._guard_output(
                query, chunks, answer, mode, top_k, request_id
            )
            guardrail_results.append(out_res)
            result["answer"] = safe_answer
            # Never cache a blocked/redacted raw answer under enforcement. In
            # shadow mode the answer is unaltered, so cache normally.
            skip_cache = self.output_pipeline.enforce and out_res.triggered
            if cacheable and query_norm and not skip_cache:
                self.database.put_cached_answer(
                    query_norm, mode, top_k, query, safe_answer, chunks
                )

        triggered, action = self._surface(guardrail_results)
        result["guardrail_triggered"] = triggered
        result["guardrail_action"] = action
        return result

    def _guard_output(self, query, chunks, answer, mode, top_k, request_id):
        """Run the OUTPUT guardrail pipeline; return (safe_answer, PipelineResult).

        Shared by the cache-hit and fresh-generation return sites so cached
        answers are validated too.
        """
        ctx = GuardrailContext(
            query=query, mode=mode, top_k=top_k, chunks=chunks,
            answer=answer, request_id=request_id,
        )
        res = self.output_pipeline.run(ctx)
        safe = res.output_text if res.output_text is not None else answer
        return safe, res

    @staticmethod
    def _surface(results):
        """Aggregate input+output pipeline results into (triggered, action_str)
        for the API response metadata."""
        from src.guardrails.base import _ACTION_RANK
        triggered = [r for r in results if r and r.triggered]
        if not triggered:
            return False, None
        strongest = max(triggered, key=lambda r: _ACTION_RANK[r.final_action])
        return True, strongest.final_action.value

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
            if self.provider == "openai":
                response = self.openai_client.chat.completions.create(
                    model=self.chat_model,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content
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

