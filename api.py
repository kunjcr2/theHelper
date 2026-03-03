"""
api.py — FastAPI REST API for theHelper RAG pipeline.

Endpoints:
  GET  /health    → system status
  POST /ingest    → upload a PDF and index it
  POST /query     → ask a question against the indexed docs

Run:
  uvicorn api:app --reload
"""

import os
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

load_dotenv()

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="theHelper API",
    description="Local-first PDF RAG pipeline",
    version="2.0.0",
)

# ── Pipeline — loaded once at startup, reused for every request ───────────────

_pipeline = None

@app.on_event("startup")
def startup():
    global _pipeline
    from rag.pipeline import RAGPipeline
    _pipeline = RAGPipeline()


def get_pipeline():
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready yet.")
    return _pipeline


# ── Schemas ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = None
    use_rerank: Optional[bool] = None


class SourceItem(BaseModel):
    doc_id: str
    filename: str
    pages: str
    chunk_id: str
    score: float


class QueryResponse(BaseModel):
    request_id: str
    answer: str
    sources: List[SourceItem]


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    chunks_added: int
    skipped: bool
    message: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Return pipeline readiness and index stats."""
    pipeline = get_pipeline()
    return {
        "status": "ok",
        "index_ready": pipeline.is_ready,
        "total_vectors": pipeline.index.total_vectors,
        "embedding_model": pipeline.config.embedding_model,
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    force: bool = Query(False, description="Re-index even if file is unchanged"),
):
    """Upload a PDF and add it to the local FAISS index."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    pipeline = get_pipeline()
    try:
        result = pipeline.ingest(content, filename=file.filename, force_reindex=force)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    msg = (
        "Already indexed — no changes made."
        if result["skipped"]
        else f"Indexed {result['chunks_added']} chunks."
    )
    return IngestResponse(**result, message=msg)


@app.post("/query", response_model=QueryResponse)
def query(body: QueryRequest):
    """Ask a question. Returns an answer with inline citations and source list."""
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    pipeline = get_pipeline()
    if not pipeline.is_ready:
        raise HTTPException(
            status_code=503,
            detail="No documents indexed yet. POST a PDF to /ingest first.",
        )

    try:
        result = pipeline.query(
            body.question,
            k=body.k,
            use_rerank=body.use_rerank,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

    return QueryResponse(
        request_id=result.request_id,
        answer=result.answer,
        sources=[SourceItem(**s) for s in result.sources],
    )
