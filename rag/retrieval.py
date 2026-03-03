"""
rag/retrieval.py — Cosine similarity search over the FAISS index.
Returns ranked chunks with scores and structured source metadata.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from rag.config import RAGConfig
from rag.index import IndexManager


@dataclass
class RetrievedChunk:
    """A retrieved chunk with its score and provenance."""
    chunk_id: str
    doc_id: str
    source_filename: str
    page_start: int
    page_end: int
    text: str
    score: float
    citation: str = ""   # e.g. "[report.pdf p.5]"

    def to_source_dict(self) -> dict:
        pages = (
            f"{self.page_start}"
            if self.page_start == self.page_end
            else f"{self.page_start}-{self.page_end}"
        )
        return {
            "doc_id": self.doc_id,
            "filename": self.source_filename,
            "pages": pages,
            "chunk_id": self.chunk_id,
            "score": round(self.score, 4),
        }


def _chunk_dict_to_retrieved(chunk: dict, score: float) -> RetrievedChunk:
    page_start = chunk.get("page_start", 0)
    page_end = chunk.get("page_end", 0)
    pages = (
        f"p.{page_start}"
        if page_start == page_end
        else f"pp.{page_start}-{page_end}"
    )
    citation = f"[{chunk['source_filename']} {pages}]"
    return RetrievedChunk(
        chunk_id=chunk["chunk_id"],
        doc_id=chunk["doc_id"],
        source_filename=chunk["source_filename"],
        page_start=page_start,
        page_end=page_end,
        text=chunk["text"],
        score=score,
        citation=citation,
    )


class Retriever:
    """Wraps IndexManager to provide cosine-similarity search."""

    def __init__(self, index: IndexManager, config: RAGConfig, embedder):
        self.index = index
        self.config = config
        self.embedder = embedder

    def retrieve(self, query: str, k: Optional[int] = None) -> List[RetrievedChunk]:
        """Return top-k chunks most similar to the query."""
        k = k or self.config.top_k
        q_vec = self.embedder.encode(query, convert_to_numpy=True)
        hits = self.index.search(q_vec, k)
        results: List[RetrievedChunk] = []
        for cid, score in hits:
            chunk = self.index.get_chunk(cid)
            if chunk:
                results.append(_chunk_dict_to_retrieved(chunk, score))
        return results
