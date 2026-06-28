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

    # A single breed entry never legitimately runs longer than this. Content
    # beyond it (between the heading and the next one) is treated as non-breed
    # overflow rather than attributed to the breed. ~5 chunks' worth.
    MAX_ENTRY_CHARS = 5000

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

                # break_point is an index *within* chunk_text, so it must be
                # compared against a chunk-relative threshold. Using the
                # absolute `start` here meant the sentence-boundary trim only
                # ever fired on the first chunk (start == 0).
                if break_point > self.chunk_size - 100:
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
    
    # A breed heading is a short ALL-CAPS line (the breed name) whose entry
    # contains a stats "info box" shortly after it — every breed in the book has
    # one, listing Origin / Weight range / Height range / Life span. Keying off
    # the info box (rather than the old "ALL-CAPS tagline on the next line"
    # heuristic) is what makes detection reliable: the book has TWO entry
    # formats — featured breeds (NAME, ALL-CAPS tagline, prose, then the box)
    # and compact breeds (NAME then the box directly, with NO tagline). The
    # tagline rule silently missed every compact breed (~130 of them, e.g.
    # PHARAOH HOUND, SCHNAUZER), merging each into the preceding breed's chunk.
    # The info box is present in both formats and absent from care/reference
    # headers like "PELLETS" or "INHERITED DISORDERS", so it also rejects those.
    _INFOBOX_FIELDS = ("origin", "weight range", "height range", "life span")
    # How many lines after the name to scan for the info box. Featured breeds
    # put the box after a paragraph of prose, so the window must be generous.
    _INFOBOX_WINDOW = 40

    # Section titles / running headers that look ALL-CAPS but aren't breeds.
    # Plural group words (HOUNDS, TERRIERS) tag dividers like "SCENT HOUNDS";
    # real breeds use the singular (AFGHAN HOUND, NORWICH TERRIER).
    _SECTION_WORDS = {
        "GUIDE", "BREEDS", "CONTENTS", "INDEX", "GLOSSARY", "HEALTH", "CARE",
        "INTRODUCTION", "ACKNOWLEDGMENTS", "ACKNOWLEDGEMENTS", "GROUPS",
        "HOUNDS", "TERRIERS", "GUNDOGS",
    }
    # Kennel-club / registry abbreviations that sit inside every info box and
    # would otherwise be mistaken for one-word breed names.
    _REGISTRY_WORDS = {"KC", "FCI", "AKC", "UKC", "ANKC", "CKC", "NZKC"}

    @staticmethod
    def _is_caps(s: str) -> bool:
        """Has letters and no lowercase — tolerant of commas, digits, punctuation."""
        return any(c.isalpha() for c in s) and not any(c.islower() for c in s)

    def _is_breed_name_line(self, line: str) -> bool:
        s = line.strip()
        if not (3 <= len(s) <= 40):
            return False
        # Breed names carry no digits; a digit means it's a page-header artifact
        # like "43WORKING DOGS" or "264 GUIDE TO BREEDS".
        if any(ch.isdigit() for ch in s):
            return False
        if not self._is_caps(s):
            return False
        # Registry markers (KC, FCI, AKC, ...) are ALL-CAPS and sit by the box.
        if s in self._REGISTRY_WORDS:
            return False
        words = s.split()
        if not (1 <= len(words) <= 5):
            return False
        # Reject section headers ("WORKING DOGS", "SCENT HOUNDS", "GUIDE TO BREEDS").
        if any(w in self._SECTION_WORDS for w in words):
            return False
        if words[-1] == "DOGS":  # category headers: TOY DOGS, WORKING DOGS, ...
            return False
        return True

    def _has_infobox_after(self, lines: List[str], i: int) -> bool:
        """True if a breed stats box (>=2 field labels) appears soon after line i."""
        window = " ".join(lines[i + 1 : i + 1 + self._INFOBOX_WINDOW]).lower()
        return sum(field in window for field in self._INFOBOX_FIELDS) >= 2

    def _find_breed_headings(self, lines: List[str]) -> List[int]:
        """Indices of lines that start a breed entry (name followed by an info box)."""
        return [
            i
            for i, line in enumerate(lines)
            if self._is_breed_name_line(line) and self._has_infobox_after(lines, i)
        ]

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

            # A real breed entry is at most a page or two. When the gap to the
            # next detected heading is far larger, the surplus is NOT this breed
            # — it's inter-section or back-matter content (e.g. the LAST breed
            # otherwise swallows the whole care/glossary/index tail). Cap the
            # breed entry and re-chunk the overflow generically, WITHOUT the
            # (wrong) breed label, so it stays retrievable but unattributed.
            entry, overflow = section[:self.MAX_ENTRY_CHARS], section[self.MAX_ENTRY_CHARS:]

            # Very long entries (rare) get size-split, each piece keeping the name.
            pieces = (
                [entry]
                if len(entry) <= int(self.chunk_size * 1.6)
                else self._split_oversized(entry)
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

            overflow = overflow.strip()
            if len(overflow) >= self.MIN_CHUNK_CHARS:
                overflow_start = char_start + len(entry)
                for gc in self.chunk_text(overflow, dict(base_metadata)):
                    rel = gc["metadata"].get("char_start", 0)
                    gc["metadata"]["page_number"] = page_for_offset(overflow_start + rel)
                    gc["metadata"]["chunk_index"] = len(chunks)
                    chunks.append(gc)

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

