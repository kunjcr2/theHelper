"""
tests/test_chunking.py — Unit tests for rag/chunking.py
"""

import pytest
from rag.chunking import Chunk, recursive_chunk, chunk_pages
from rag.ingest import PageDoc


def _make_page(text: str, page_num: int = 1) -> PageDoc:
    return PageDoc(
        doc_id="deadbeef",
        source_filename="test.pdf",
        page_num=page_num,
        text=text,
    )


class FakeConfig:
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size = 100
    chunk_overlap = 20
    semantic_chunking = False
    semantic_breakpoint_threshold = 0.85


# ── Chunk ID determinism ──────────────────────────────────────────────────────

def test_chunk_id_is_deterministic():
    """Same text must always produce the same chunk_id (SHA-256)."""
    text = "Hello, world. This is a test sentence."
    c1 = Chunk.build(text, "doc1", "a.pdf", 1, 1, "model-v1")
    c2 = Chunk.build(text, "doc1", "a.pdf", 1, 1, "model-v1")
    assert c1.chunk_id == c2.chunk_id


def test_chunk_id_differs_for_different_text():
    c1 = Chunk.build("Alpha text.", "d", "f.pdf", 1, 1, "m")
    c2 = Chunk.build("Beta text.", "d", "f.pdf", 1, 1, "m")
    assert c1.chunk_id != c2.chunk_id


def test_chunk_id_equals_hash():
    """chunk_id and hash fields must be identical."""
    c = Chunk.build("some text", "d", "f.pdf", 2, 2, "m")
    assert c.chunk_id == c.hash


# ── Metadata propagation ──────────────────────────────────────────────────────

def test_page_metadata_propagates():
    """Chunks must carry the correct page_start/page_end from their source page."""
    cfg = FakeConfig()
    pages = [_make_page("A" * 50, page_num=3)]
    chunks = recursive_chunk(pages, cfg)
    assert all(c.page_start == 3 for c in chunks)
    assert all(c.page_end == 3 for c in chunks)
    assert all(c.source_filename == "test.pdf" for c in chunks)
    assert all(c.doc_id == "deadbeef" for c in chunks)


def test_embedding_model_recorded():
    cfg = FakeConfig()
    pages = [_make_page("Some text for embedding model test.")]
    chunks = recursive_chunk(pages, cfg)
    assert all(c.embedding_model == cfg.embedding_model for c in chunks)


# ── Chunking behaviour ────────────────────────────────────────────────────────

def test_short_text_produces_one_chunk():
    cfg = FakeConfig()
    pages = [_make_page("Short text.")]
    chunks = recursive_chunk(pages, cfg)
    assert len(chunks) == 1


def test_long_text_produces_multiple_chunks():
    cfg = FakeConfig()
    # 500 chars > chunk_size=100, should split
    pages = [_make_page("word " * 100)]
    chunks = recursive_chunk(pages, cfg)
    assert len(chunks) > 1


def test_no_empty_chunks():
    cfg = FakeConfig()
    pages = [_make_page("   \n\n   \n")]
    # Blank page → ingest would skip it, but if it slips through, no empty chunks
    chunks = recursive_chunk(pages, cfg)
    assert all(c.text.strip() for c in chunks)


def test_chunk_to_dict_roundtrip():
    c = Chunk.build("Hello world", "doc", "file.pdf", 1, 1, "model")
    d = c.to_dict()
    c2 = Chunk.from_dict(d)
    assert c.chunk_id == c2.chunk_id
    assert c.text == c2.text
    assert c.page_start == c2.page_start


def test_chunk_pages_dispatches_to_recursive():
    cfg = FakeConfig()
    pages = [_make_page("Test dispatch to recursive chunker.")]
    chunks = chunk_pages(pages, cfg, embedder=None)
    assert len(chunks) >= 1
