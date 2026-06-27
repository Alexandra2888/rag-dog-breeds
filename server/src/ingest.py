"""Auto-ingestion of PDF documents from the data directory.

This populates the vector database from the PDFs sitting in ``data/`` so the
RAG service and the LiveKit voice agent have a knowledge base to answer from,
without anyone having to upload a PDF by hand. It is idempotent: PDFs whose
filename is already present in the ``documents`` table are skipped.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional

from src.config import settings
from src.pdf_processor import PDFProcessor
from src.embeddings import EmbeddingGenerator
from src.database import Database

logger = logging.getLogger(__name__)

# Repo-root/data — resolves regardless of the current working directory.
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def ingest_pdf(
    pdf_path: Path,
    pdf_processor: PDFProcessor,
    embedding_generator: EmbeddingGenerator,
    database: Database,
) -> int:
    """Process, embed and store a single PDF. Returns the number of chunks stored."""
    chunks = pdf_processor.process_pdf(str(pdf_path))
    if not chunks:
        logger.warning(f"No text extracted from {pdf_path.name}; skipping")
        return 0

    document_id = database.create_document(pdf_path.name)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_generator.generate_embeddings_batch(texts)
    return database.insert_chunks(document_id, chunks, embeddings)


def ingest_data_directory(
    data_dir: Optional[Path] = None,
    database: Optional[Database] = None,
    force: bool = False,
) -> List[Dict[str, int]]:
    """Ingest every PDF in ``data_dir`` that has not been ingested yet.

    Args:
        data_dir: Directory to scan for ``*.pdf`` (defaults to repo ``data/``).
        database: Optional existing Database instance to reuse its pool.
        force: Re-ingest even if a document with the same filename already exists.

    Returns:
        A list of ``{"document": name, "chunks": n}`` summaries for what was ingested.
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    if not data_dir.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        return []

    pdf_paths = sorted(data_dir.glob("*.pdf"))
    if not pdf_paths:
        logger.info(f"No PDFs found in {data_dir}")
        return []

    pdf_processor = PDFProcessor()
    embedding_generator = EmbeddingGenerator()
    database = database or Database()

    # Map each filename to *all* of its document ids (earlier runs may have
    # created duplicates with the same name).
    existing: Dict[str, List[str]] = {}
    for doc in database.list_documents():
        existing.setdefault(doc["document_name"], []).append(doc["id"])
    results: List[Dict[str, int]] = []

    for pdf_path in pdf_paths:
        if pdf_path.name in existing:
            if not force:
                logger.info(f"Already ingested, skipping: {pdf_path.name}")
                continue
            # Re-ingest: drop every old copy (and its chunks via cascade) first
            # so we replace rather than accumulate duplicates.
            doc_ids = existing[pdf_path.name]
            logger.info(
                f"Re-ingesting {pdf_path.name}: removing {len(doc_ids)} previous version(s)"
            )
            for doc_id in doc_ids:
                database.delete_document(doc_id)

        logger.info(f"Ingesting {pdf_path.name} ...")
        chunks = ingest_pdf(pdf_path, pdf_processor, embedding_generator, database)
        if chunks:
            logger.info(f"Ingested {pdf_path.name}: {chunks} chunks")
            results.append({"document": pdf_path.name, "chunks": chunks})

    return results


def main() -> None:
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    force = "--force" in sys.argv[1:]
    logger.info(f"Embedding model: {settings.ollama_embedding_model}")
    if force:
        logger.info("Force mode: existing documents will be re-ingested")
    results = ingest_data_directory(force=force)
    if results:
        total = sum(r["chunks"] for r in results)
        logger.info(f"Done. Ingested {len(results)} document(s), {total} chunks total.")
    else:
        logger.info("Nothing to ingest (data directory empty or already ingested).")


if __name__ == "__main__":
    main()
