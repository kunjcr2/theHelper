"""
rag/tracing.py — Local observability: JSONL query traces + per-request artifacts.

One JSONL record per query written to data/traces/<YYYY-MM-DD>.jsonl
Per-request artifacts written to data/artifacts/<request_id>/
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag.config import RAGConfig
from rag.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Tracer:
    """
    Writes structured traces + file artifacts for every query.

    Usage:
        tracer = Tracer(config)
        request_id = tracer.start_request(query)
        # ... do retrieval + generation ...
        tracer.record(
            request_id=request_id,
            query=query,
            retrieval_config={...},
            retrieved_chunks=[...],
            timings={...},
            model="gpt-4o-mini",
            answer="...",
            prompt="...",
            context_chunks=[...],
        )
    """

    def __init__(self, config: "RAGConfig", session_id: Optional[str] = None):
        self.config = config
        self.session_id: str = session_id or str(uuid.uuid4())
        config.ensure_dirs()

    def _trace_path(self) -> Path:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.config.traces_dir / f"{date_str}.jsonl"

    def _artifact_dir(self, request_id: str) -> Path:
        d = self.config.artifacts_dir / request_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def start_request(self) -> str:
        """Generate and return a fresh request_id (UUID)."""
        return str(uuid.uuid4())

    def record(
        self,
        request_id: str,
        query: str,
        retrieval_config: Dict[str, Any],
        retrieved_chunks: List["RetrievedChunk"],
        timings: Dict[str, float],
        model: str,
        answer: str,
        prompt: str = "",
        error: Optional[str] = None,
    ) -> None:
        """
        Write one JSONL trace record and per-request artifact files.
        """
        timestamp = _utc_now()
        chunk_refs = [
            {"chunk_id": c.chunk_id, "score": round(c.score, 4)}
            for c in retrieved_chunks
        ]
        context_dicts = [
            {"chunk_id": c.chunk_id, "text": c.text, "source": c.citation}
            for c in retrieved_chunks
        ]

        trace = {
            "timestamp": timestamp,
            "session_id": self.session_id,
            "request_id": request_id,
            "query": query,
            "retrieval_config": retrieval_config,
            "retrieved_chunk_refs": chunk_refs,
            "timings_ms": timings,
            "model": model,
            "answer_length": len(answer),
            "error": error,
        }

        # Write JSONL trace
        try:
            with open(self._trace_path(), "a", encoding="utf-8") as f:
                f.write(json.dumps(trace, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.error("Failed to write trace: %s", exc)

        # Write per-request artifact files
        try:
            art_dir = self._artifact_dir(request_id)

            # prompt.txt
            (art_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

            # context.json
            (art_dir / "context.json").write_text(
                json.dumps(context_dicts, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            # response.json
            response_payload = {
                "request_id": request_id,
                "timestamp": timestamp,
                "model": model,
                "answer": answer,
                "error": error,
            }
            (art_dir / "response.json").write_text(
                json.dumps(response_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.error("Failed to write artifacts for %s: %s", request_id, exc)
