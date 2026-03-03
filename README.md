# theHelper — Production RAG API

> Local-first PDF Q&A over a REST API: FAISS index, cross-encoder reranking, JSONL observability, eval suite, and CI.

- About architecture: [ARCHITECTURE.md](ARCHITECTURE.md)

---

## Architecture

| Component | Technology | Where it runs |
|-----------|------------|---------------|
| PDF parsing | `pypdf` | Local |
| Embeddings | `all-MiniLM-L6-v2` | Local |
| Chunking | `RecursiveCharacterTextSplitter` | Local |
| Vector index | FAISS (persisted to disk) | Local — `data/faiss.index` |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local (on by default) |
| Observability | JSONL traces + request artifacts | Local — `data/` |
| Generation | OpenAI `gpt-4o-mini` | API |

---

## Setup

```bash
git clone https://github.com/kunjcr2/theHelper.git
cd theHelper
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-your-key" > .env
```

---

## Running

```bash
uvicorn api:app --reload
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)
```

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Index status + vector count |
| `POST` | `/ingest` | Upload a PDF and index it |
| `POST` | `/query` | Ask a question, get answer + citations |

### Ingest

```bash
curl -X POST http://localhost:8000/ingest -F "file=@report.pdf"

# Force re-index even if file hasn't changed
curl -X POST "http://localhost:8000/ingest?force=true" -F "file=@report.pdf"
```

### Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key findings?", "k": 5}'
```

**Request body:**
```json
{ "question": "...", "k": 5, "use_rerank": true }
```

**Response:**
```json
{
  "request_id": "uuid",
  "answer": "The main finding is ... [report.pdf p.3]",
  "sources": [
    { "filename": "report.pdf", "pages": "3", "score": 0.91, "chunk_id": "...", "doc_id": "..." }
  ]
}
```

---

## Observability

Every query is automatically traced locally — no external services.

| Artifact | Path |
|----------|------|
| FAISS index | `data/faiss.index` |
| Chunk metadata | `data/metadata.json` |
| Query traces | `data/traces/<YYYY-MM-DD>.jsonl` |
| Request artifacts | `data/artifacts/<request_id>/` |
| Eval reports | `eval/reports/<run_id>.json` |

Each trace record contains: timestamp, session_id, request_id, query, retrieval config, retrieved chunk refs + scores, latency timings, model used, answer length, errors.

---

## Eval

```bash
python -m eval.run eval/datasets/sample.json
```

Reports saved to `eval/reports/` as both JSON and Markdown.

---

## Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
theHelper/
├── api.py              ← FastAPI entry point
├── rag/
│   ├── config.py       ← all config knobs
│   ├── ingest.py       ← PDF → per-page text + metadata
│   ├── chunking.py     ← recursive & semantic chunking
│   ├── index.py        ← FAISS + metadata store
│   ├── retrieval.py    ← cosine similarity search
│   ├── rerank.py       ← cross-encoder reranker
│   ├── tracing.py      ← JSONL traces + artifact files
│   └── pipeline.py     ← orchestrates everything
├── eval/
│   ├── metrics.py
│   ├── run.py
│   └── datasets/sample.json
├── tests/
├── data/               ← gitignored (index, traces, artifacts)
└── .github/workflows/ci.yml
```

---

## License

MIT — *kunjcr2@gmail.com*
