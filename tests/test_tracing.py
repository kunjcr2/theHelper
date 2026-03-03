"""
tests/test_tracing.py — Unit tests for rag/tracing.py
"""

import json
import tempfile
from pathlib import Path

import pytest

from rag.config import RAGConfig
from rag.tracing import Tracer
from rag.retrieval import RetrievedChunk


def _make_config(tmp_path: Path) -> RAGConfig:
    cfg = RAGConfig()
    cfg.data_dir = tmp_path
    return cfg


def _make_retrieved_chunk(chunk_id: str = "abc123") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="doc_xyz",
        source_filename="report.pdf",
        page_start=1,
        page_end=2,
        text="This is a test chunk with some content.",
        score=0.92,
        citation="[report.pdf p.1-2]",
    )


# ── Tracer writes JSONL ───────────────────────────────────────────────────────

def test_record_writes_jsonl_line(tmp_path):
    cfg = _make_config(tmp_path)
    tracer = Tracer(cfg, session_id="test-session")
    request_id = tracer.start_request()
    chunk = _make_retrieved_chunk()

    tracer.record(
        request_id=request_id,
        query="What is the main finding?",
        retrieval_config={"k": 5, "use_mmr": False},
        retrieved_chunks=[chunk],
        timings={"retrieval_ms": 42.1, "generation_ms": 310.5},
        model="gpt-4o-mini",
        answer="The main finding is X.",
        prompt="System prompt here.",
    )

    # Find the JSONL trace file
    trace_files = list((tmp_path / "traces").glob("*.jsonl"))
    assert len(trace_files) == 1, "Exactly one JSONL trace file should exist"

    lines = trace_files[0].read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["query"] == "What is the main finding?"
    assert record["session_id"] == "test-session"
    assert record["request_id"] == request_id
    assert record["model"] == "gpt-4o-mini"
    assert record["answer_length"] == len("The main finding is X.")


def test_jsonl_schema_has_required_fields(tmp_path):
    """Verify all required schema fields are present in the JSONL record."""
    required_fields = [
        "timestamp", "session_id", "request_id", "query",
        "retrieval_config", "retrieved_chunk_refs",
        "timings_ms", "model", "answer_length", "error",
    ]

    cfg = _make_config(tmp_path)
    tracer = Tracer(cfg, session_id="schema-test")
    request_id = tracer.start_request()

    tracer.record(
        request_id=request_id,
        query="Schema test query",
        retrieval_config={"k": 3},
        retrieved_chunks=[_make_retrieved_chunk()],
        timings={"retrieval_ms": 10.0},
        model="gpt-4o-mini",
        answer="Answer text.",
        prompt="",
    )

    trace_files = list((tmp_path / "traces").glob("*.jsonl"))
    record = json.loads(trace_files[0].read_text(encoding="utf-8").strip())

    for field in required_fields:
        assert field in record, f"Missing required field: {field}"


def test_chunk_refs_in_trace(tmp_path):
    """Chunk refs must include chunk_id and score."""
    cfg = _make_config(tmp_path)
    tracer = Tracer(cfg)
    request_id = tracer.start_request()
    chunk = _make_retrieved_chunk(chunk_id="deadbeef1234")

    tracer.record(
        request_id=request_id,
        query="test",
        retrieval_config={},
        retrieved_chunks=[chunk],
        timings={},
        model="gpt-4o-mini",
        answer="ans",
        prompt="",
    )

    trace_files = list((tmp_path / "traces").glob("*.jsonl"))
    record = json.loads(trace_files[0].read_text(encoding="utf-8").strip())
    refs = record["retrieved_chunk_refs"]
    assert len(refs) == 1
    assert refs[0]["chunk_id"] == "deadbeef1234"
    assert "score" in refs[0]


# ── Artifact files ────────────────────────────────────────────────────────────

def test_artifact_files_written(tmp_path):
    cfg = _make_config(tmp_path)
    tracer = Tracer(cfg)
    request_id = tracer.start_request()

    tracer.record(
        request_id=request_id,
        query="artifact test",
        retrieval_config={"k": 5},
        retrieved_chunks=[_make_retrieved_chunk()],
        timings={"retrieval_ms": 5.0},
        model="gpt-4o-mini",
        answer="The answer.",
        prompt="The prompt.",
    )

    art_dir = tmp_path / "artifacts" / request_id
    assert (art_dir / "prompt.txt").exists()
    assert (art_dir / "context.json").exists()
    assert (art_dir / "response.json").exists()


def test_artifact_prompt_content(tmp_path):
    cfg = _make_config(tmp_path)
    tracer = Tracer(cfg)
    request_id = tracer.start_request()
    prompt_text = "System prompt with details."

    tracer.record(
        request_id=request_id,
        query="q",
        retrieval_config={},
        retrieved_chunks=[],
        timings={},
        model="gpt-4o-mini",
        answer="a",
        prompt=prompt_text,
    )

    art_dir = tmp_path / "artifacts" / request_id
    assert (art_dir / "prompt.txt").read_text(encoding="utf-8") == prompt_text


def test_artifact_response_json_schema(tmp_path):
    cfg = _make_config(tmp_path)
    tracer = Tracer(cfg)
    request_id = tracer.start_request()

    tracer.record(
        request_id=request_id,
        query="q",
        retrieval_config={},
        retrieved_chunks=[],
        timings={},
        model="gpt-4o-mini",
        answer="My answer.",
        prompt="",
    )

    art_dir = tmp_path / "artifacts" / request_id
    response = json.loads((art_dir / "response.json").read_text(encoding="utf-8"))

    for field in ["request_id", "timestamp", "model", "answer", "error"]:
        assert field in response

    assert response["request_id"] == request_id
    assert response["answer"] == "My answer."


def test_multiple_records_append_to_same_file(tmp_path):
    """Multiple records on the same day should produce multiple JSONL lines in one file."""
    cfg = _make_config(tmp_path)
    tracer = Tracer(cfg)

    for i in range(3):
        rid = tracer.start_request()
        tracer.record(
            request_id=rid,
            query=f"Query {i}",
            retrieval_config={},
            retrieved_chunks=[],
            timings={},
            model="gpt-4o-mini",
            answer=f"Answer {i}",
            prompt="",
        )

    trace_files = list((tmp_path / "traces").glob("*.jsonl"))
    assert len(trace_files) == 1
    lines = [l for l in trace_files[0].read_text(encoding="utf-8").strip().split("\n") if l]
    assert len(lines) == 3
