"""
rag/chunking.py — Text chunking with rich per-chunk metadata.

Supports:
  - Recursive character splitting via LangChain RecursiveCharacterTextSplitter
  - Semantic splitting using embedding-distance breakpoints (custom, no heavy deps)
"""

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import RAGConfig
from rag.ingest import PageDoc


@dataclass
class Chunk:
    """A single text chunk with full provenance metadata."""
    chunk_id: str          # SHA-256 of text content
    doc_id: str            # SHA-256 of the source PDF bytes
    source_filename: str   # original PDF filename
    page_start: int        # first page this chunk draws from (1-indexed)
    page_end: int          # last page this chunk draws from (1-indexed)
    text: str              # chunk content
    hash: str              # same as chunk_id
    created_at: str        # ISO-8601 UTC timestamp
    embedding_model: str   # which model will embed this chunk

    @staticmethod
    def _make_id(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @classmethod
    def build(cls, text: str, doc_id: str, source_filename: str,
              page_start: int, page_end: int, embedding_model: str) -> "Chunk":
        cid = cls._make_id(text)
        return cls(
            chunk_id=cid,
            doc_id=doc_id,
            source_filename=source_filename,
            page_start=page_start,
            page_end=page_end,
            text=text,
            hash=cid,
            created_at=datetime.now(timezone.utc).isoformat(),
            embedding_model=embedding_model,
        )

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source_filename": self.source_filename,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "text": self.text,
            "hash": self.hash,
            "created_at": self.created_at,
            "embedding_model": self.embedding_model,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)


# ── Recursive Chunking (LangChain) ────────────────────────────────────────────

def recursive_chunk(pages: List[PageDoc], config: RAGConfig) -> List[Chunk]:
    """Split pages using LangChain's RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[Chunk] = []
    for page in pages:
        splits = splitter.split_text(page.text)
        for text in splits:
            if not text.strip():
                continue
            chunks.append(Chunk.build(
                text=text,
                doc_id=page.doc_id,
                source_filename=page.source_filename,
                page_start=page.page_num,
                page_end=page.page_num,
                embedding_model=config.embedding_model,
            ))
    return chunks


# ── Semantic Chunking ─────────────────────────────────────────────────────────
# LangChain's SemanticChunker exists but requires the full langchain stack.
# We use a lightweight cosine-distance approach directly with our embedder.

def semantic_chunk(pages: List[PageDoc], config: RAGConfig, embedder) -> List[Chunk]:
    """
    Split pages at points where embedding similarity between consecutive
    sentences drops below a threshold. Falls back to recursive chunking
    for pages that are too short.
    """
    import numpy as np

    # Use LangChain splitter to get initial fine-grained sentence-level splits
    sentence_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,        # small — we want sentence-level pieces
        chunk_overlap=0,
        separators=[". ", "! ", "? ", "\n", " ", ""],
    )

    chunks: List[Chunk] = []

    for page in pages:
        sentences = [s.strip() for s in sentence_splitter.split_text(page.text) if s.strip()]

        if len(sentences) < 2:
            # Too short — fall back to recursive
            chunks.extend(recursive_chunk([page], config))
            continue

        embs = embedder.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)

        # Cosine similarity between consecutive sentences (already normalised → dot product)
        sims = (embs[:-1] * embs[1:]).sum(axis=1)

        # Group sentences into chunks, splitting where similarity drops
        groups: List[List[str]] = []
        current: List[str] = [sentences[0]]

        for i, sim in enumerate(sims):
            if sim < config.semantic_breakpoint_threshold:
                groups.append(current)
                current = [sentences[i + 1]]
            else:
                current.append(sentences[i + 1])
        if current:
            groups.append(current)

        for group in groups:
            text = " ".join(group).strip()
            if text:
                chunks.append(Chunk.build(
                    text=text,
                    doc_id=page.doc_id,
                    source_filename=page.source_filename,
                    page_start=page.page_num,
                    page_end=page.page_num,
                    embedding_model=config.embedding_model,
                ))

    return chunks


# ── Dispatcher ────────────────────────────────────────────────────────────────

def chunk_pages(pages: List[PageDoc], config: RAGConfig, embedder=None) -> List[Chunk]:
    """Route to the right chunking strategy based on config."""
    if config.semantic_chunking:
        if embedder is None:
            raise ValueError("embedder is required for semantic chunking")
        return semantic_chunk(pages, config, embedder)
    return recursive_chunk(pages, config)
