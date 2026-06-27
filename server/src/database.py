"""Database operations with PostgreSQL and pgvector."""
import logging
import uuid
import json
from typing import List, Dict, Any, Optional
from psycopg2.extras import execute_values, Json
from psycopg2.pool import ThreadedConnectionPool
from pgvector.psycopg2 import register_vector

from src.config import settings

logger = logging.getLogger(__name__)


class Database:
    """Handles database operations with pgvector."""
    
    def __init__(self, connection_string: str = None):
        self.connection_string = connection_string or settings.database_url
        self.pool = None
        self._initialize_pool()
        self._ensure_extension()
        self._create_tables()
    
    def _initialize_pool(self):
        """Initialize connection pool."""
        try:
            self.pool = ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                dsn=self.connection_string
            )
            logger.info("Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            # Don't raise immediately - allow lazy connection retry
            self.pool = None
            raise
    
    def _get_connection(self, register_pgvector=True):
        """Get a connection from the pool and optionally register pgvector."""
        conn = self.pool.getconn()
        if register_pgvector:
            try:
                register_vector(conn)
            except Exception:
                pass
        return conn
    
    def _return_connection(self, conn):
        """Return a connection to the pool."""
        self.pool.putconn(conn)
    
    def _ensure_extension(self):
        """Ensure pgvector extension is enabled."""
        # Get connection without registering vector first (extension doesn't exist yet)
        conn = self._get_connection(register_pgvector=False)
        try:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                conn.commit()
                logger.info("pgvector extension enabled")
            # Now register vector on this connection
            try:
                register_vector(conn)
            except Exception as e:
                # If registration fails, it's okay - we'll register on next connection
                logger.warning(f"Could not register vector type immediately: {e}")
        except Exception as e:
            logger.error(f"Error enabling pgvector extension: {e}")
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
    
    def _create_tables(self):
        """Create necessary tables if they don't exist."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Documents table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        document_name VARCHAR(255) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Chunks table with vector column
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS chunks (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
                        content TEXT NOT NULL,
                        embedding vector(768),  -- Default dimension, will be adjusted
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # NOTE: No ANN (ivfflat/HNSW) index on `embedding`.
                # For a single-book corpus (~1k chunks) an exact brute-force
                # scan is sub-millisecond and gives 100% recall. An ivfflat
                # index with default probes=1 only scans one cluster and
                # silently misses most relevant chunks — bad accuracy. Drop
                # the old index if it exists (left over from earlier setups).
                cur.execute("DROP INDEX IF EXISTS chunks_embedding_idx;")

                # Create index on document_id for faster lookups
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS chunks_document_id_idx
                    ON chunks(document_id);
                """)

                # GIN full-text index for the keyword half of hybrid search.
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS chunks_content_fts_idx
                    ON chunks USING GIN (to_tsvector('english', content));
                """)
                
                conn.commit()
                logger.info("Database tables created/verified")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
    
    def adjust_vector_dimension(self, dimension: int):
        """
        Adjust the vector column dimension if needed.
        Note: This requires dropping and recreating the column, which will lose data.
        Only use this during initial setup.
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Check current dimension
                cur.execute("""
                    SELECT atttypmod FROM pg_attribute 
                    WHERE attrelid = 'chunks'::regclass 
                    AND attname = 'embedding';
                """)
                result = cur.fetchone()
                
                if result and result[0] != dimension:
                    logger.warning(f"Vector dimension mismatch. Current: {result[0]}, Required: {dimension}")
                    logger.warning("Dropping and recreating embedding column. This will delete existing data!")
                    
                    cur.execute("ALTER TABLE chunks DROP COLUMN embedding;")
                    cur.execute(f"ALTER TABLE chunks ADD COLUMN embedding vector({dimension});")
                    cur.execute("""
                        CREATE INDEX chunks_embedding_idx 
                        ON chunks USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100);
                    """)
                    conn.commit()
                    logger.info(f"Vector dimension adjusted to {dimension}")
        except Exception as e:
            logger.error(f"Error adjusting vector dimension: {e}")
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
    
    def create_document(self, document_name: str) -> str:
        """
        Create a new document record.
        
        Args:
            document_name: Name of the document
            
        Returns:
            Document ID (UUID as string)
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                doc_id = str(uuid.uuid4())
                cur.execute(
                    "INSERT INTO documents (id, document_name) VALUES (%s, %s) RETURNING id;",
                    (doc_id, document_name)
                )
                conn.commit()
                logger.info(f"Created document: {document_name} ({doc_id})")
                return doc_id
        except Exception as e:
            logger.error(f"Error creating document: {e}")
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
    
    def insert_chunks(
        self, 
        document_id: str, 
        chunks: List[Dict[str, Any]], 
        embeddings: List[List[float]]
    ) -> int:
        """
        Insert chunks with embeddings into the database.
        
        Args:
            document_id: ID of the parent document
            chunks: List of chunk dictionaries with 'text' and 'metadata'
            embeddings: List of embedding vectors
            
        Returns:
            Number of chunks inserted
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Prepare data for bulk insert
                values = []
                for chunk, embedding in zip(chunks, embeddings):
                    chunk_id = str(uuid.uuid4())
                    values.append((
                        chunk_id,
                        document_id,
                        chunk["text"],
                        embedding,  # pgvector will handle conversion
                        Json(chunk.get("metadata", {}))
                    ))
                
                # Bulk insert
                execute_values(
                    cur,
                    """
                    INSERT INTO chunks (id, document_id, content, embedding, metadata)
                    VALUES %s;
                    """,
                    values,
                    template=None
                )
                
                conn.commit()
                inserted_count = len(values)
                logger.info(f"Inserted {inserted_count} chunks for document {document_id}")
                return inserted_count
        except Exception as e:
            logger.error(f"Error inserting chunks: {e}")
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
    
    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        threshold: Optional[float] = None,
        document_id: Optional[str] = None,
        query_text: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant chunks.

        When ``query_text`` is given, performs HYBRID search: it fuses dense
        vector similarity with Postgres full-text (keyword) ranking via
        Reciprocal Rank Fusion (RRF). Dense vectors capture semantics; keyword
        search captures proper nouns (e.g. breed names) that embeddings blur
        together. Falls back to pure vector search when ``query_text`` is None.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            threshold: Minimum vector-similarity threshold (optional)
            document_id: Filter by document ID (optional)
            query_text: Raw query string for the keyword half of hybrid search

        Returns:
            List of chunk dictionaries with similarity scores
        """
        # Candidate pool per modality before fusion.
        pool = max(top_k * 5, 40)
        # Weighted Reciprocal Rank Fusion. A small k sharpens the advantage of
        # top-ranked hits; weighting keyword > vector ensures a query naming a
        # breed surfaces that breed even when the embedding ranks it poorly
        # (dense vectors blur proper nouns together).
        rrf_k = 10
        w_vec = 1.0
        w_fts = 2.0

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                if query_text:
                    doc_vec = "WHERE c.document_id = %(doc)s" if document_id else ""
                    doc_fts = "AND c.document_id = %(doc)s" if document_id else ""
                    thresh_sql = (
                        "WHERE (1 - (c.embedding <=> %(qvec)s::vector)) >= %(thresh)s"
                        if threshold is not None else ""
                    )
                    query = f"""
                        WITH vec AS (
                            SELECT c.id,
                                   ROW_NUMBER() OVER (
                                       ORDER BY c.embedding <=> %(qvec)s::vector
                                   ) AS rnk
                            FROM chunks c
                            {doc_vec}
                            ORDER BY c.embedding <=> %(qvec)s::vector
                            LIMIT %(pool)s
                        ),
                        fts AS (
                            -- OR the query terms (plainto ANDs them, so a
                            -- natural-language question matches nothing). This
                            -- lets a chunk rank on the terms it does contain
                            -- (e.g. the breed name).
                            SELECT c.id,
                                   ROW_NUMBER() OVER (
                                       ORDER BY ts_rank(
                                           to_tsvector('english', c.content),
                                           to_tsquery('english', replace(
                                               plainto_tsquery('english', %(qtext)s)::text,
                                               '&', '|'))
                                       ) DESC
                                   ) AS rnk
                            FROM chunks c
                            WHERE to_tsvector('english', c.content)
                                  @@ to_tsquery('english', replace(
                                         plainto_tsquery('english', %(qtext)s)::text,
                                         '&', '|'))
                                  {doc_fts}
                            LIMIT %(pool)s
                        ),
                        fused AS (
                            SELECT COALESCE(v.id, f.id) AS id,
                                   COALESCE(%(wv)s / (%(k)s + v.rnk), 0)
                                 + COALESCE(%(wf)s / (%(k)s + f.rnk), 0) AS rrf
                            FROM vec v
                            FULL OUTER JOIN fts f ON v.id = f.id
                        )
                        SELECT c.id, c.content, c.metadata,
                               1 - (c.embedding <=> %(qvec)s::vector) AS similarity
                        FROM fused
                        JOIN chunks c ON c.id = fused.id
                        {thresh_sql}
                        ORDER BY fused.rrf DESC
                        LIMIT %(topk)s;
                    """
                    cur.execute(query, {
                        "qvec": query_embedding,
                        "qtext": query_text,
                        "doc": document_id,
                        "pool": pool,
                        "k": rrf_k,
                        "wv": w_vec,
                        "wf": w_fts,
                        "topk": top_k,
                        "thresh": threshold,
                    })
                else:
                    query = """
                        SELECT c.id, c.content, c.metadata,
                               1 - (c.embedding <=> %s::vector) as similarity
                        FROM chunks c
                    """
                    params = [query_embedding]
                    if document_id:
                        query += " WHERE c.document_id = %s"
                        params.append(document_id)
                    if threshold is not None:
                        where_clause = "WHERE" if not document_id else "AND"
                        query += f" {where_clause} (1 - (c.embedding <=> %s::vector)) >= %s"
                        params.append(query_embedding)
                        params.append(threshold)
                    query += """
                        ORDER BY c.embedding <=> %s::vector
                        LIMIT %s;
                    """
                    params.append(query_embedding)
                    params.append(top_k)
                    cur.execute(query, params)

                results = cur.fetchall()

                chunks = []
                for row in results:
                    metadata = row[2]
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)
                    elif metadata is None:
                        metadata = {}

                    chunks.append({
                        "id": str(row[0]),
                        "content": row[1],
                        "metadata": metadata,
                        "similarity_score": float(row[3])
                    })

                return chunks
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            raise
        finally:
            self._return_connection(conn)
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all ingested documents with their chunk counts.

        Returns:
            List of document dictionaries
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        d.id,
                        d.document_name,
                        d.created_at,
                        COUNT(c.id) AS chunk_count
                    FROM documents d
                    LEFT JOIN chunks c ON c.document_id = d.id
                    GROUP BY d.id, d.document_name, d.created_at
                    ORDER BY d.created_at DESC;
                    """
                )
                results = cur.fetchall()

                return [
                    {
                        "id": str(row[0]),
                        "document_name": row[1],
                        "created_at": row[2].isoformat() if row[2] else None,
                        "chunk_count": int(row[3]),
                    }
                    for row in results
                ]
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            raise
        finally:
            self._return_connection(conn)

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and its chunks (via ON DELETE CASCADE).

        Args:
            document_id: Document ID

        Returns:
            True if a document was deleted, False if it did not exist
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM documents WHERE id = %s;",
                    (document_id,)
                )
                deleted = cur.rowcount
                conn.commit()
                if deleted:
                    logger.info(f"Deleted document {document_id}")
                return deleted > 0
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            List of chunk dictionaries
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, content, metadata
                    FROM chunks
                    WHERE document_id = %s
                    ORDER BY created_at;
                    """,
                    (document_id,)
                )
                results = cur.fetchall()
                
                chunks = []
                for row in results:
                    metadata = row[2]
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)
                    elif metadata is None:
                        metadata = {}
                    
                    chunks.append({
                        "id": str(row[0]),
                        "content": row[1],
                        "metadata": metadata
                    })
                
                return chunks
        except Exception as e:
            logger.error(f"Error getting document chunks: {e}")
            raise
        finally:
            self._return_connection(conn)

