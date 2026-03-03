"""
rag/index.py — Local FAISS vector index + JSON metadata store.

Responsibilities:
  - Build index from chunks + embeddings
  - Persist / load to disk (faiss.index + metadata.json)
  - Incremental updates: re-index only changed/new docs
  - Detect embedding-model changes → full rebuild
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from rag.chunking import Chunk
from rag.config import RAGConfig

logger = logging.getLogger(__name__)


class IndexManager:
    """
    Manages a FAISS IndexFlatIP (inner-product / cosine with normalised vecs)
    alongside a JSON metadata store keyed by chunk_id.
    """

    DIMENSION_KEY = "__dimension__"
    MODEL_KEY = "__embedding_model__"

    def __init__(self, config: "RAGConfig"):
        self.config = config
        self._dim: Optional[int] = None
        self._index: Optional[faiss.Index] = None
        # Ordered list of chunk_ids matching FAISS row indices
        self._id_list: List[str] = []
        # Full metadata store: chunk_id → chunk dict
        self._store: Dict[str, dict] = {}
        # doc_id → file_hash mapping for change detection
        self._doc_hashes: Dict[str, str] = {}

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_index(self, dim: int) -> faiss.Index:
        idx = faiss.IndexFlatIP(dim)
        return idx

    def _normalize(self, vecs: np.ndarray) -> np.ndarray:
        vecs = vecs.astype("float32")
        faiss.normalize_L2(vecs)
        return vecs

    # ── Build / update ────────────────────────────────────────────────────────

    def build(self, chunks: List["Chunk"], embedder) -> None:
        """
        Encode all chunks and build a fresh FAISS index.
        Replaces any existing in-memory index.
        """
        if not chunks:
            raise ValueError("Cannot build index from empty chunk list")

        texts = [c.text for c in chunks]
        logger.info("Encoding %d chunks ...", len(texts))
        raw_embs: np.ndarray = embedder.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )
        embs = self._normalize(raw_embs)
        dim = embs.shape[1]

        self._dim = dim
        self._index = self._make_index(dim)
        self._index.add(embs)

        self._id_list = [c.chunk_id for c in chunks]
        self._store = {c.chunk_id: c.to_dict() for c in chunks}
        # Record file hashes
        self._doc_hashes = {}
        for c in chunks:
            self._doc_hashes.setdefault(c.doc_id, c.hash)

        logger.info("Index built: %d vectors (dim=%d)", self._index.ntotal, dim)

    def update(self, new_chunks: List["Chunk"], embedder) -> None:
        """
        Append new chunks to an existing index without a full rebuild.
        Skips chunks already present (by chunk_id).
        """
        if self._index is None:
            self.build(new_chunks, embedder)
            return

        to_add = [c for c in new_chunks if c.chunk_id not in self._store]
        if not to_add:
            logger.info("No new chunks to add; index unchanged.")
            return

        texts = [c.text for c in to_add]
        raw_embs: np.ndarray = embedder.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )
        embs = self._normalize(raw_embs)
        self._index.add(embs)
        for c in to_add:
            self._id_list.append(c.chunk_id)
            self._store[c.chunk_id] = c.to_dict()
        logger.info("Index updated: added %d chunks (%d total)", len(to_add), self._index.ntotal)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist FAISS index and metadata store to disk."""
        if self._index is None:
            raise RuntimeError("No index to save — call build() first")

        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.config.faiss_path))

        payload = {
            self.DIMENSION_KEY: self._dim,
            self.MODEL_KEY: self.config.embedding_model,
            "__id_list__": self._id_list,
            "__doc_hashes__": self._doc_hashes,
            "chunks": self._store,
        }
        with open(self.config.metadata_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        logger.info(
            "Saved index to %s and metadata to %s",
            self.config.faiss_path,
            self.config.metadata_path,
        )

    def load(self) -> bool:
        """
        Load persisted index from disk.
        Returns True if successful, False if no index exists yet.
        """
        if not self.config.faiss_path.exists() or not self.config.metadata_path.exists():
            return False

        self._index = faiss.read_index(str(self.config.faiss_path))

        with open(self.config.metadata_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self._dim = payload.get(self.DIMENSION_KEY)
        self._id_list = payload.get("__id_list__", [])
        self._doc_hashes = payload.get("__doc_hashes__", {})
        self._store = payload.get("chunks", {})

        stored_model = payload.get(self.MODEL_KEY, "")
        if stored_model and stored_model != self.config.embedding_model:
            logger.warning(
                "Embedding model changed (%s → %s). Index must be rebuilt.",
                stored_model,
                self.config.embedding_model,
            )
            # Reset so callers know to rebuild
            self._index = None
            self._id_list = []
            self._store = {}
            self._doc_hashes = {}
            return False

        logger.info(
            "Loaded index (%d vectors) from %s",
            self._index.ntotal,
            self.config.faiss_path,
        )
        return True

    # ── Change detection ──────────────────────────────────────────────────────

    def needs_reindex(self, doc_id: str, file_hash: str) -> bool:
        """
        Return True if this document is new or its file hash has changed.
        """
        stored = self._doc_hashes.get(doc_id)
        if stored is None:
            return True        # new document
        return stored != file_hash  # changed document

    def remove_doc(self, doc_id: str) -> None:
        """
        Remove all chunks belonging to doc_id from the metadata store.
        Note: FAISS FlatIndex doesn't support deletion; chunks are only removed
        from metadata so they won't appear in search results (soft-delete).
        A full rebuild is recommended when re-indexing changed docs.
        """
        to_remove = [cid for cid, c in self._store.items() if c["doc_id"] == doc_id]
        for cid in to_remove:
            del self._store[cid]
            if cid in self._id_list:
                self._id_list.remove(cid)
        self._doc_hashes.pop(doc_id, None)
        logger.info("Soft-removed %d chunks for doc_id=%s", len(to_remove), doc_id)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self, query_vec: np.ndarray, k: int
    ) -> List[Tuple[str, float]]:
        """
        Run FAISS nearest-neighbour search.
        Returns list of (chunk_id, score) pairs, filtered to chunks in store.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        q = self._normalize(query_vec.reshape(1, -1))
        actual_k = min(k * 2, self._index.ntotal)  # over-fetch to allow filtering
        scores, indices = self._index.search(q, actual_k)

        results: List[Tuple[str, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self._id_list):
                continue
            cid = self._id_list[idx]
            if cid not in self._store:
                continue  # soft-deleted
            results.append((cid, float(score)))
            if len(results) == k:
                break
        return results

    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        return self._store.get(chunk_id)

    @property
    def is_ready(self) -> bool:
        return self._index is not None and self._index.ntotal > 0

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal if self._index else 0
