# theHelper — Production RAG API

> Local-first PDF Q&A over a REST API: FAISS index, JSONL observability, cross-encoder reranking, eval suite, and CI.

---

## Architecture

| Component | Technology | Where it runs |
|-----------|------------|---------------|
| PDF parsing | `pypdf` | Local |
| Embeddings | `all-MiniLM-L6-v2` | Local |
| Vector index | FAISS (persisted to disk) | Local — `data/faiss.index` |
| Chunking | Recursive / Semantic | Local |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local (optional) |
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

## Running the API

```bash
uvicorn api:app --reload
# → http://localhost:8000
# → http://localhost:8000/docs   (Swagger UI)
```

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Index status, vector count |
| `POST` | `/ingest` | Upload a PDF (`multipart/form-data`) |
| `POST` | `/query` | Ask a question, get answer + citations |

### Ingest

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@report.pdf"
```

### Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key findings?", "k": 5, "use_mmr": true}'
```

**Request body:**
```json
{ "question": "...", "k": 5, "use_mmr": false, "use_rerank": false }
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

## CLI

```bash
python -m rag.cli ingest path/to/doc.pdf
python -m rag.cli query "What are the main findings?" --k 8 --mmr
python -m rag.cli eval eval/datasets/sample.json
```

---

## Observability

| Artifact | Path |
|----------|------|
| FAISS index | `data/faiss.index` |
| Chunk metadata | `data/metadata.json` |
| Query traces | `data/traces/<YYYY-MM-DD>.jsonl` |
| Request artifacts | `data/artifacts/<request_id>/` |
| Eval reports | `eval/reports/<run_id>.json` |

---

## Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
theHelper/
├── api.py             ← FastAPI app (entry point)
├── rag/
│   ├── config.py
│   ├── ingest.py
│   ├── chunking.py
│   ├── index.py
│   ├── retrieval.py
│   ├── rerank.py
│   ├── tracing.py
│   ├── pipeline.py
│   └── cli.py
├── eval/
│   ├── metrics.py
│   ├── run.py
│   └── datasets/sample.json
├── tests/
├── data/              (gitignored)
└── .github/workflows/ci.yml
```

---

## License

MIT — *kunjcr2@gmail.com*
