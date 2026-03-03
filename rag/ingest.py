"""
rag/ingest.py — PDF ingestion + per-page metadata extraction.

PDF → List[PageDoc] with cleaned text and structural metadata.
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import IO, List, Union


@dataclass
class PageDoc:
    """Single PDF page with its text and provenance metadata."""
    doc_id: str           # SHA-256 of the source file bytes
    source_filename: str  # original filename (no path)
    page_num: int         # 1-indexed
    text: str             # cleaned page text


def compute_file_hash(source: Union[str, Path, bytes, IO]) -> str:
    """
    Return SHA-256 hex digest of a file.
    Accepts a path string/Path, raw bytes, or a file-like object.
    The file-like object is rewound after reading.
    """
    h = hashlib.sha256()
    if isinstance(source, (str, Path)):
        with open(source, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    elif isinstance(source, bytes):
        h.update(source)
    else:  # file-like
        pos = source.tell()
        for chunk in iter(lambda: source.read(65536), b""):
            h.update(chunk)
        source.seek(pos)
    return h.hexdigest()


def _clean_text(raw: str) -> str:
    """Normalize PDF-extracted text: strip control chars, collapse whitespace."""
    # Replace non-printable / non-ASCII except common whitespace
    text = re.sub(r"[^\x20-\x7E\n\t]", " ", raw)
    # Collapse internal whitespace while preserving paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(
    source: Union[str, Path, bytes, IO],
    filename: str = "document.pdf",
) -> List[PageDoc]:
    """
    Extract and clean text from every page of a PDF.

    Args:
        source:   file path, raw bytes, or seekable file-like object.
        filename: label stored in metadata (should be the original filename).

    Returns:
        List[PageDoc] — one entry per non-blank page.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "pypdf is required for PDF ingestion. "
            "Install it with: pip install pypdf"
        ) from exc

    # Resolve source
    if isinstance(source, (str, Path)):
        raw_bytes = Path(source).read_bytes()
        from io import BytesIO
        reader_input = BytesIO(raw_bytes)
    elif isinstance(source, bytes):
        raw_bytes = source
        from io import BytesIO
        reader_input = BytesIO(raw_bytes)
    else:
        # file-like — read once for hashing, wrap for pypdf
        pos = source.tell()
        raw_bytes = source.read()
        source.seek(pos)
        from io import BytesIO
        reader_input = BytesIO(raw_bytes)

    doc_id = hashlib.sha256(raw_bytes).hexdigest()

    reader = PdfReader(reader_input)
    pages: List[PageDoc] = []

    for page_num, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned = _clean_text(raw_text)
        if not cleaned:
            continue  # skip blank/image-only pages
        pages.append(
            PageDoc(
                doc_id=doc_id,
                source_filename=filename,
                page_num=page_num,
                text=cleaned,
            )
        )

    if not pages:
        raise ValueError(
            f"No readable text could be extracted from '{filename}'. "
            "The PDF may be scanned/image-only."
        )

    return pages
