"""
tests/test_index.py — Unit tests for rag/index.py
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from rag.chunking import Chunk
from rag.config import RAGConfig
from rag.index import IndexManager


def _make_config(tmp_path: Path) -> RAGConfig:
    cfg = RAGConfig()
    cfg.data_dir = tmp_path
    return cfg


def _make_chunks(n: int, doc_id: str = "doc1") -> list[Chunk]:
    chunks = []
    for i in range(n):
        # Include doc_id in text so chunks from different docs have different SHA-256 IDs
        c = Chunk.build(
            text=f"Chunk {i} from {doc_id}: unique content for testing the index pipeline.",
            doc_id=doc_id,
            source_filename="test.pdf",
            page_start=i + 1,
            page_end=i + 1,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )
        chunks.append(c)
    return chunks


class FakeEmbedder:
    """Returns deterministic random-ish embeddings (dim=32) without loading a real model."""

    DIM = 32

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False,
               normalize_embeddings=False):
        rng = np.random.default_rng(seed=42)
        embs = rng.random((len(texts), self.DIM)).astype("float32")
        if normalize_embeddings:
            norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
            embs /= norms
        return embs


# ── Build & search ────────────────────────────────────────────────────────────

def test_build_and_search(tmp_path):
    cfg = _make_config(tmp_path)
    idx = IndexManager(cfg)
    embedder = FakeEmbedder()
    chunks = _make_chunks(5)

    idx.build(chunks, embedder)

    assert idx.is_ready
    assert idx.total_vectors == 5

    q_vec = np.random.default_rng(0).random(FakeEmbedder.DIM).astype("float32")
    results = idx.search(q_vec, k=3)
    assert len(results) == 3
    for cid, score in results:
        assert isinstance(cid, str)
        assert isinstance(score, float)


# ── Save / load roundtrip ─────────────────────────────────────────────────────

def test_save_load_roundtrip(tmp_path):
    cfg = _make_config(tmp_path)
    idx = IndexManager(cfg)
    embedder = FakeEmbedder()
    chunks = _make_chunks(4)

    idx.build(chunks, embedder)
    idx.save()

    # Load into a fresh manager
    idx2 = IndexManager(cfg)
    loaded = idx2.load()
    assert loaded, "load() should return True when files exist"
    assert idx2.total_vectors == 4

    # Search results should be non-empty
    q_vec = np.ones(FakeEmbedder.DIM, dtype="float32")
    results = idx2.search(q_vec, k=2)
    assert len(results) == 2


def test_load_returns_false_when_no_files(tmp_path):
    cfg = _make_config(tmp_path)
    idx = IndexManager(cfg)
    assert idx.load() is False


# ── Incremental update ────────────────────────────────────────────────────────

def test_incremental_update_appends(tmp_path):
    cfg = _make_config(tmp_path)
    idx = IndexManager(cfg)
    embedder = FakeEmbedder()

    first_batch = _make_chunks(3, doc_id="docA")
    idx.build(first_batch, embedder)
    assert idx.total_vectors == 3

    second_batch = _make_chunks(2, doc_id="docB")
    idx.update(second_batch, embedder)
    assert idx.total_vectors == 5


def test_update_skips_existing_chunks(tmp_path):
    cfg = _make_config(tmp_path)
    idx = IndexManager(cfg)
    embedder = FakeEmbedder()

    chunks = _make_chunks(3)
    idx.build(chunks, embedder)
    # Update with same chunks — should add 0
    idx.update(chunks, embedder)
    assert idx.total_vectors == 3


# ── Hash-based change detection ───────────────────────────────────────────────

def test_needs_reindex_new_doc(tmp_path):
    cfg = _make_config(tmp_path)
    idx = IndexManager(cfg)
    assert idx.needs_reindex("brand_new_doc_id", "abc123") is True


def test_needs_reindex_unchanged_doc(tmp_path):
    cfg = _make_config(tmp_path)
    idx = IndexManager(cfg)
    idx._doc_hashes["doc1"] = "hash_xyz"
    assert idx.needs_reindex("doc1", "hash_xyz") is False


def test_needs_reindex_changed_doc(tmp_path):
    cfg = _make_config(tmp_path)
    idx = IndexManager(cfg)
    idx._doc_hashes["doc1"] = "old_hash"
    assert idx.needs_reindex("doc1", "new_hash") is True


# ── Embedding model change detection ─────────────────────────────────────────

def test_load_detects_model_change(tmp_path):
    """If the stored model differs from config model, load() should return False."""
    cfg = _make_config(tmp_path)
    idx = IndexManager(cfg)
    embedder = FakeEmbedder()
    idx.build(_make_chunks(2), embedder)
    idx.save()

    # Change the config model
    cfg2 = _make_config(tmp_path)
    cfg2.embedding_model = "different/model"
    idx2 = IndexManager(cfg2)
    result = idx2.load()
    assert result is False, "Should return False when embedding model changed"
