"""
rag/rerank.py — Optional cross-encoder reranker.

When enabled, takes the initial FAISS retrieval candidates and re-scores
them using a cross-encoder for higher precision.
"""

import logging
from typing import List, Optional

from rag.config import RAGConfig
from rag.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


class Reranker:
    """
    Wraps a sentence-transformers CrossEncoder for reranking.
    Lazy-loaded on first use to avoid startup overhead when disabled.
    """

    def __init__(self, config: "RAGConfig"):
        self.config = config
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            logger.info("Loading cross-encoder: %s", self.config.reranker_model)
            self._model = CrossEncoder(self.config.reranker_model)
            logger.info("Cross-encoder ready.")
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for reranking. "
                "Install it with: pip install sentence-transformers"
            ) from exc

    def rerank(
        self,
        query: str,
        chunks: List["RetrievedChunk"],
        top_k: Optional[int] = None,
    ) -> List["RetrievedChunk"]:
        """
        Rerank retrieved chunks using the cross-encoder.
        Returns chunks sorted by cross-encoder score (descending).
        """
        if not chunks:
            return chunks

        self._load()

        pairs = [(query, c.text) for c in chunks]
        scores: list = self._model.predict(pairs)  # type: ignore[union-attr]

        scored = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        if top_k is not None:
            scored = scored[:top_k]

        # Update score field with cross-encoder score
        result: List["RetrievedChunk"] = []
        for chunk, score in scored:
            # Replace FAISS score with cross-encoder score (normalise to 0-1 range)
            from dataclasses import replace
            result.append(replace(chunk, score=float(score)))
        return result
