"""Embedding generation using Ollama."""
import logging
from typing import List
import ollama

from src.config import settings

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings using Ollama."""
    
    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.ollama_embedding_model
        # Initialize Ollama client
        try:
            # Try to use Client class if available
            self.client = ollama.Client(host=self.base_url)
        except (AttributeError, TypeError):
            # Fallback: use module-level functions
            self.client = None
            import os
            # Extract host from URL (remove http:// or https://)
            host = self.base_url.replace('http://', '').replace('https://', '')
            os.environ['OLLAMA_HOST'] = host
    
    def _apply_task_prefix(self, text: str, input_type: str) -> str:
        """
        nomic-embed-text is trained with asymmetric task instructions:
        documents must be embedded as 'search_document: ...' and queries as
        'search_query: ...'. Without these prefixes the query and document
        vectors live in mismatched spaces and retrieval quality drops sharply.
        Only applied for nomic models; a no-op for everything else.
        """
        if "nomic" in self.model.lower():
            return f"{input_type}: {text}"
        return text

    def generate_embedding(
        self, text: str, input_type: str = "search_document"
    ) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            input_type: 'search_document' for stored chunks (default) or
                'search_query' for user queries (nomic task prefix).

        Returns:
            List of floats representing the embedding vector
        """
        prompt = self._apply_task_prefix(text, input_type)
        try:
            if self.client:
                response = self.client.embeddings(model=self.model, prompt=prompt)
            else:
                response = ollama.embeddings(model=self.model, prompt=prompt)
            return response["embedding"]
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        total = len(texts)
        
        logger.info(f"Generating embeddings for {total} texts in batches of {batch_size}")
        
        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []
            
            for text in batch:
                embedding = self.generate_embedding(text)
                batch_embeddings.append(embedding)
            
            embeddings.extend(batch_embeddings)
            
            if (i + batch_size) % (batch_size * 10) == 0 or i + batch_size >= total:
                logger.info(f"Processed {min(i + batch_size, total)}/{total} embeddings")
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings from the model.
        This is model-specific and may need adjustment.
        
        Returns:
            Embedding dimension
        """
        # Test with a small text to get dimension
        test_embedding = self.generate_embedding("test")
        return len(test_embedding)

