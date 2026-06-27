"""PDF processing and text chunking."""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from pypdf import PdfReader

from src.config import settings

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF text extraction and chunking."""
    
    # Drop chunks shorter than this (kills 1-char fragments from page edges).
    MIN_CHUNK_CHARS = 80

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
    
    def extract_text(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from PDF with page information.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries with 'text' and 'page' keys
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Extracting text from PDF: {pdf_path}")
        reader = PdfReader(str(pdf_path))
        
        pages = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text.strip():  # Only add non-empty pages
                pages.append({
                    "text": text,
                    "page": page_num,
                    "metadata": {
                        "source": str(pdf_path.name),
                        "page_number": page_num,
                        "total_pages": len(reader.pages)
                    }
                })
        
        logger.info(f"Extracted {len(pages)} pages from PDF")
        return pages
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries
        """
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        metadata = metadata or {}
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # Extract chunk
            chunk_text = text[start:end]
            
            # Try to break at sentence boundary if not at end
            if end < len(text):
                # Look for sentence endings in the last 100 chars
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > start + self.chunk_size - 100:
                    chunk_text = chunk_text[:break_point + 1]
                    end = start + break_point + 1
            
            stripped = chunk_text.strip()
            # Skip degenerate fragments (e.g. a page edge with a stray char).
            if len(stripped) >= self.MIN_CHUNK_CHARS:
                chunk_metadata = {
                    **metadata,
                    "chunk_index": len(chunks),
                    "char_start": start,
                    "char_end": end
                }
                chunks.append({
                    "text": stripped,
                    "metadata": chunk_metadata
                })
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    # A breed heading is a short ALL-CAPS line (the breed name) immediately
    # followed by a longer ALL-CAPS line (the marketing tagline). This pattern
    # reliably separates "GOLDEN RETRIEVER" from running headers like
    # "258 GUIDE TO BREEDS" (whose following line is another short name).
    # Section titles / running headers that look ALL-CAPS but aren't breeds.
    _SECTION_WORDS = {
        "GUIDE", "BREEDS", "CONTENTS", "INDEX", "GLOSSARY", "HEALTH", "CARE",
        "INTRODUCTION", "ACKNOWLEDGMENTS", "ACKNOWLEDGEMENTS", "GROUPS",
    }

    @staticmethod
    def _is_caps(s: str) -> bool:
        """Has letters and no lowercase — tolerant of commas, digits, punctuation."""
        return any(c.isalpha() for c in s) and not any(c.islower() for c in s)

    def _is_breed_name_line(self, line: str) -> bool:
        s = line.strip()
        if not (2 <= len(s) <= 40):
            return False
        # Breed names carry no digits; a digit means it's a page-header artifact
        # like "43WORKING DOGS" or "264 GUIDE TO BREEDS".
        if any(ch.isdigit() for ch in s):
            return False
        if not self._is_caps(s):
            return False
        words = s.split()
        if not (1 <= len(words) <= 5):
            return False
        # Reject section headers ("WORKING DOGS", "TERRIERS", "GUIDE TO BREEDS").
        if any(w in self._SECTION_WORDS for w in words):
            return False
        if words[-1] == "DOGS":  # category headers: TOY DOGS, WORKING DOGS, ...
            return False
        return True

    def _is_tagline_line(self, line: str) -> bool:
        s = line.strip()
        return self._is_caps(s) and len(s.split()) >= 5

    def _find_breed_headings(self, lines: List[str]) -> List[int]:
        """Indices of lines that start a breed entry (name + tagline pattern)."""
        headings: List[int] = []
        for i, line in enumerate(lines):
            if not self._is_breed_name_line(line):
                continue
            # Next non-empty line must look like a tagline sentence.
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and self._is_tagline_line(lines[j]):
                headings.append(i)
        return headings

    def _chunk_by_breed_entries(
        self,
        full_text: str,
        base_metadata: Dict[str, Any],
        page_for_offset,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Split a breed-catalogue document into one chunk per breed entry, so each
        chunk's embedding represents a single breed. Returns None if the document
        doesn't look like a breed catalogue (caller falls back to size chunking).
        """
        lines = full_text.split("\n")
        headings = self._find_breed_headings(lines)
        if len(headings) < 20:
            return None  # Not a breed catalogue — use generic chunking.

        # Char offset of the start of each line within full_text.
        line_offsets: List[int] = []
        off = 0
        for line in lines:
            line_offsets.append(off)
            off += len(line) + 1  # +1 for the '\n'

        chunks: List[Dict[str, Any]] = []
        for idx, start_line in enumerate(headings):
            end_line = headings[idx + 1] if idx + 1 < len(headings) else len(lines)
            breed_name = lines[start_line].strip()
            section = "\n".join(lines[start_line:end_line]).strip()
            if len(section) < self.MIN_CHUNK_CHARS:
                continue

            char_start = line_offsets[start_line]
            page_number = page_for_offset(char_start)

            # Very long entries (rare) get size-split, each piece keeping the name.
            pieces = (
                [section]
                if len(section) <= int(self.chunk_size * 1.6)
                else self._split_oversized(section)
            )
            for piece in pieces:
                chunks.append({
                    "text": piece,
                    "metadata": {
                        **base_metadata,
                        "breed": breed_name,
                        "chunk_index": len(chunks),
                        "char_start": char_start,
                        "page_number": page_number,
                    },
                })

        logger.info(f"Breed-aware chunking produced {len(chunks)} chunks "
                    f"from {len(headings)} detected breed entries")
        return chunks

    def _split_oversized(self, text: str) -> List[str]:
        """Size-split a section, returning only the text pieces."""
        return [c["text"] for c in self.chunk_text(text)]

    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Process PDF: extract text and chunk it.

        Pages are concatenated into a single text stream, then chunked. For breed
        catalogues we split on breed headings so each chunk is one breed (far
        better per-breed retrieval); otherwise we fall back to size-based chunks
        across the document. Each chunk is tagged with the page it starts on.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of chunk dictionaries ready for embedding
        """
        pages = self.extract_text(pdf_path)
        if not pages:
            logger.info("No extractable text in PDF")
            return []

        # Concatenate pages, recording the char offset where each page starts.
        separator = "\n\n"
        full_text_parts: List[str] = []
        page_starts: List[Tuple[int, int]] = []  # (char_offset, page_number)
        cursor = 0
        for page_data in pages:
            page_starts.append((cursor, page_data["page"]))
            full_text_parts.append(page_data["text"])
            cursor += len(page_data["text"]) + len(separator)
        full_text = separator.join(full_text_parts)

        source = pages[0]["metadata"]["source"]
        total_pages = pages[0]["metadata"]["total_pages"]
        base_metadata = {"source": source, "total_pages": total_pages}

        def page_for_offset(offset: int) -> int:
            page = page_starts[0][1]
            for start_offset, page_number in page_starts:
                if offset >= start_offset:
                    page = page_number
                else:
                    break
            return page

        # Prefer breed-aware chunking; fall back to size-based.
        chunks = self._chunk_by_breed_entries(full_text, base_metadata, page_for_offset)
        if chunks is None:
            chunks = self.chunk_text(full_text, dict(base_metadata))
            for chunk in chunks:
                chunk["metadata"]["page_number"] = page_for_offset(
                    chunk["metadata"].get("char_start", 0)
                )

        logger.info(f"Created {len(chunks)} chunks from PDF")
        return chunks

