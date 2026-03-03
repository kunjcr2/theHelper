"""
rag/pipeline.py — The main RAG orchestrator.

Ties together: ingest → chunk → index → retrieve → (rerank) → generate + cite.
Also owns the Tracer and exposes the interface used by both CLI and Streamlit UI.
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import IO, List, Optional, Union
from pathlib import Path

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from rag.config import RAGConfig, load_config
from rag.ingest import extract_pages, compute_file_hash
from rag.chunking import chunk_pages
from rag.index import IndexManager
from rag.retrieval import Retriever, RetrievedChunk
from rag.rerank import Reranker
from rag.tracing import Tracer

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Structured result returned by RAGPipeline.query()."""
    request_id: str
    answer: str
    sources: List[dict]          # [{doc_id, filename, pages, chunk_id, score}]
    retrieved_chunks: List[RetrievedChunk]
    error: Optional[str] = None


_SYSTEM_PROMPT = """\
You are a precise research assistant. Answer the user's question using ONLY the
provided document context. If the answer is not in the context, say so honestly.
Keep your answer concise but complete. Include citation tags exactly as they appear
in the context (e.g. [report.pdf p.5]) when referencing specific information.
"""


class RAGPipeline:
    """
    Production RAG pipeline:
      1. ingest()  — extract pages, chunk, embed, update FAISS index
      2. query()   — retrieve → (rerank) → generate with citations
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or load_config()
        self.config.ensure_dirs()

        # Load embedder once
        logger.info("Loading embedder: %s", self.config.embedding_model)
        self.embedder = SentenceTransformer(self.config.embedding_model)

        # Index manager — attempt to load persisted index
        self.index = IndexManager(self.config)
        loaded = self.index.load()
        if not loaded:
            logger.info("No existing index found; will build on first ingest.")

        # Retriever
        self.retriever = Retriever(self.index, self.config, self.embedder)

        # Reranker (lazy)
        self.reranker = Reranker(self.config) if self.config.use_rerank else None

        # OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set — add it to your .env file")
        self.client = OpenAI(api_key=api_key)

        # Tracer (session scoped)
        self.tracer = Tracer(self.config)

        logger.info("RAGPipeline ready.")

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest(
        self,
        source: Union[str, Path, bytes, IO],
        filename: str = "document.pdf",
        force_reindex: bool = False,
    ) -> dict:
        """
        Ingest a PDF document into the index.

        Args:
            source:         PDF file path, bytes, or file-like object.
            filename:       Human-readable name stored in chunk metadata.
            force_reindex:  Ignore hash check and always re-embed.

        Returns:
            dict with keys: doc_id, filename, chunks_added, skipped.
        """
        t0 = time.perf_counter()

        # Compute file hash for change detection
        file_hash = compute_file_hash(source)

        # Extract pages
        pages = extract_pages(source, filename=filename)
        doc_id = pages[0].doc_id

        # Check if reindex is needed
        if not force_reindex and not self.index.needs_reindex(doc_id, file_hash):
            logger.info("Document '%s' unchanged — skipping re-index.", filename)
            return {
                "doc_id": doc_id,
                "filename": filename,
                "chunks_added": 0,
                "skipped": True,
            }

        # Remove old chunks for this doc if it existed
        if doc_id in self.index._doc_hashes:
            self.index.remove_doc(doc_id)

        # Chunk
        chunks = chunk_pages(pages, self.config, embedder=self.embedder)
        logger.info("Chunked '%s' → %d chunks", filename, len(chunks))

        # Update file hash in doc_hashes store
        for c in chunks:
            self.index._doc_hashes[doc_id] = file_hash

        # Update index (append mode; build from scratch if first doc)
        self.index.update(chunks, self.embedder)

        # Persist to disk
        self.index.save()

        elapsed = time.perf_counter() - t0
        logger.info(
            "Ingested '%s': %d chunks in %.2fs", filename, len(chunks), elapsed
        )
        return {
            "doc_id": doc_id,
            "filename": filename,
            "chunks_added": len(chunks),
            "skipped": False,
        }

    # ── Query ────────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        *,
        k: Optional[int] = None,
        use_rerank: Optional[bool] = None,
    ) -> QueryResult:
        """
        Run the full RAG query pipeline.

        Args:
            question:    User's natural-language question.
            k:           Override top-k retrieval count.
            use_rerank:  Override rerank toggle.

        Returns:
            QueryResult with answer, sources, and request_id.
        """
        if not self.index.is_ready:
            return QueryResult(
                request_id="",
                answer="No documents have been indexed yet. Please ingest a PDF first.",
                sources=[],
                retrieved_chunks=[],
                error="index_empty",
            )

        request_id = self.tracer.start_request()
        timings: dict = {}
        error: Optional[str] = None

        # Retrieval
        t0 = time.perf_counter()
        k_val = k or self.config.top_k
        use_rerank_val = use_rerank if use_rerank is not None else self.config.use_rerank

        try:
            retrieved = self.retriever.retrieve(question, k=k_val)
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc)
            return QueryResult(
                request_id=request_id,
                answer="Retrieval failed.",
                sources=[],
                retrieved_chunks=[],
                error=str(exc),
            )
        timings["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # Reranking (optional)
        if use_rerank_val and retrieved:
            t1 = time.perf_counter()
            try:
                reranker = self.reranker or Reranker(self.config)
                retrieved = reranker.rerank(question, retrieved, top_k=k_val)
            except Exception as exc:
                logger.warning("Reranking failed (continuing without): %s", exc)
            timings["rerank_ms"] = round((time.perf_counter() - t1) * 1000, 1)

        # Build context with inline citations
        context_parts = []
        for chunk in retrieved:
            context_parts.append(f"{chunk.citation}\n{chunk.text}")
        context = "\n\n---\n\n".join(context_parts)

        # Build prompt
        prompt = (
            f"{_SYSTEM_PROMPT}\n\n"
            f"Context:\n\n{context}\n\n"
            f"---\n\nQuestion: {question}"
        )

        # Generate
        t2 = time.perf_counter()
        answer = ""
        try:
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Context:\n\n{context}\n\n---\n\nQuestion: {question}",
                    },
                ],
                temperature=self.config.openai_temperature,
                max_tokens=self.config.openai_max_tokens,
            )
            answer = response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("Generation failed: %s", exc)
            error = str(exc)
            answer = f"Generation failed: {exc}"
        timings["generation_ms"] = round((time.perf_counter() - t2) * 1000, 1)

        # Structured sources
        sources = [c.to_source_dict() for c in retrieved]

        # Trace
        retrieval_config = {
            "k": k_val,
            "use_rerank": use_rerank_val,
            "embedding_model": self.config.embedding_model,
        }
        self.tracer.record(
            request_id=request_id,
            query=question,
            retrieval_config=retrieval_config,
            retrieved_chunks=retrieved,
            timings=timings,
            model=self.config.openai_model,
            answer=answer,
            prompt=prompt,
            error=error,
        )

        return QueryResult(
            request_id=request_id,
            answer=answer,
            sources=sources,
            retrieved_chunks=retrieved,
            error=error,
        )

    @property
    def is_ready(self) -> bool:
        """True if the index has at least one vector."""
        return self.index.is_ready
