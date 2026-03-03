# theHelper — Architecture & Pipeline Deep Dive

This document explains exactly how the system works, from booting the server to getting an answer back, tied to the actual files in this repo.

---

## Folder map

```
theHelper/
├── api.py              ← HTTP layer (FastAPI). Only talks to RAGPipeline.
├── rag/
│   ├── config.py       ← Single dataclass that holds every tunable knob.
│   ├── ingest.py       ← PDF → list of cleaned pages, each with metadata.
│   ├── chunking.py     ← Pages → smaller text chunks, each with a unique ID.
│   ├── index.py        ← Stores vectors in FAISS. Saves/loads from disk.
│   ├── retrieval.py    ← Takes a question, returns the most relevant chunks.
│   ├── rerank.py       ← Re-scores chunks using a cross-encoder model.
│   ├── tracing.py      ← Writes one log line per query + artifact files.
│   └── pipeline.py     ← Wires all of the above together. The main entry point.
├── eval/
│   ├── metrics.py      ← Functions to score retrieval and generation quality.
│   ├── run.py          ← Runs a dataset through the pipeline, saves a report.
│   └── datasets/
│       └── sample.json ← Example evaluation questions.
└── tests/
    ├── test_chunking.py
    ├── test_index.py
    └── test_tracing.py
```

---

## 1. Server startup

**File:** `api.py`

```
uvicorn api:app --reload
```

When the server starts, FastAPI fires `@app.on_event("startup")`, which calls:

```python
_pipeline = RAGPipeline()
```

`RAGPipeline.__init__` (in `pipeline.py`) does three things:
1. Loads the SentenceTransformer embedding model (`all-MiniLM-L6-v2`) into memory.
2. Tries to load a previously saved FAISS index from `data/faiss.index` and `data/metadata.json`.
3. Creates a `Tracer` instance with a UUID for this server session.

Everything stays in a global `_pipeline` variable. One load, reused for every request.

---

## 2. Configuration

**File:** `rag/config.py`

Everything tunable lives in `RAGConfig`, a Python dataclass:

```
chunk_size              = 500       how big each text chunk is (characters)
chunk_overlap           = 100       how much two adjacent chunks share
semantic_chunking       = False     use embedding-based splits instead of character splits
top_k                   = 5         how many chunks to retrieve per query
use_rerank              = True      run cross-encoder after FAISS (on by default)
embedding_model         = all-MiniLM-L6-v2
reranker_model          = cross-encoder/ms-marco-MiniLM-L-6-v2
openai_model            = gpt-4o-mini
data_dir                = ./data/
```

All of these can be overridden via environment variables (e.g. `RAG_TOP_K=10`) or passed directly when constructing `RAGPipeline`.

---

## 3. Ingestion — POST /ingest

**Files:** `api.py` → `pipeline.py` → `ingest.py` → `chunking.py` → `index.py`

### 3a. Receive the file

`api.py` receives the uploaded PDF as raw bytes via `UploadFile`. It passes those bytes to `pipeline.ingest(content, filename)`.

### 3b. Hash the file

`ingest.py: compute_file_hash(source)` computes a SHA-256 of the file bytes. This hash becomes the `doc_id`. It is used to detect whether this exact file has already been indexed.

```
file bytes  →  SHA-256  →  doc_id (hex string, e.g. "a3f9c2...")
```

If `index.needs_reindex(doc_id, file_hash)` returns False, ingestion stops here and returns `skipped: true`.

### 3c. Extract text per page

`ingest.py: extract_pages(source, filename)` opens the PDF with `pypdf` and reads each page individually.

For each page it creates a `PageDoc`:

```python
PageDoc(
    doc_id          = "a3f9c2...",  # same SHA-256 as the file
    source_filename = "report.pdf",
    page_num        = 3,            # 1-indexed
    text            = "cleaned text..."
)
```

Cleaning strips control characters and collapses whitespace. Blank pages are dropped.

### 3d. Chunk the pages

`chunking.py: chunk_pages(pages, config)` splits each page's text into smaller pieces called chunks.

**Default (recursive):** Uses LangChain's `RecursiveCharacterTextSplitter`. It tries to split on `\n\n`, then `\n`, then `. `, then spaces, then characters — always respecting `chunk_size` and `chunk_overlap`.

**Optional (semantic):** Also uses `RecursiveCharacterTextSplitter` but at a small size (200 chars) to get sentence-level pieces first. It then embeds each sentence and merges adjacent ones unless their cosine similarity drops below a threshold.

Each piece becomes a `Chunk` dataclass:

```python
Chunk(
    chunk_id        = SHA-256 of the text content,
    doc_id          = "a3f9c2...",
    source_filename = "report.pdf",
    page_start      = 3,
    page_end        = 3,
    text            = "The main argument is...",
    hash            = same as chunk_id,
    created_at      = "2026-03-02T17:00:00Z",
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
)
```

The `chunk_id` is deterministic — same text always produces the same ID.

### 3e. Embed and store in FAISS

`index.py: IndexManager.update(chunks, embedder)` takes the chunk list and:

1. Encodes the text of each chunk into a 384-dimensional vector using the SentenceTransformer embedder.
2. L2-normalises the vectors (converts them to unit vectors so cosine similarity = dot product, which is what FAISS `IndexFlatIP` computes).
3. Adds the vectors to the FAISS index with `index.add(vectors)`.
4. Stores the chunk metadata in `_store` (a dict: `chunk_id → chunk dict`).
5. Records the chunk's position in `_id_list` (a list where index = FAISS row number).

The three data structures work together:

```
FAISS index   →  row numbers (integers)
_id_list      →  maps row number to chunk_id
_store        →  maps chunk_id to full chunk dict (text, pages, filename, etc.)
```

### 3f. Persist to disk

`index.save()` writes:
- `data/faiss.index`  — the FAISS index in binary format
- `data/metadata.json` — `_id_list`, `_store`, `_doc_hashes`, embedding model name

On the next server start, `index.load()` reads these files back. If the stored embedding model doesn't match the current config, `load()` returns False and a full rebuild is triggered automatically.

---

## 4. Querying — POST /query

**Files:** `api.py` → `pipeline.py` → `retrieval.py` → `rerank.py` → OpenAI → `tracing.py`

### 4a. Embed the question

`retrieval.py: Retriever.retrieve(question, k)` encodes the question string into a 384-dimensional vector using the same SentenceTransformer.

### 4b. Search FAISS

`index.search(query_vector, k)` runs:

```python
scores, indices = faiss_index.search(query_vector, k)
```

FAISS returns two numpy arrays:
- `indices` — row numbers of the k most similar vectors
- `scores` — their inner-product scores (cosine similarity, since vectors are normalised)

The code maps each row number → `chunk_id` → full chunk dict, and returns a list of `RetrievedChunk` objects with citation tags pre-built:

```
[report.pdf p.3]
```

### 4c. Rerank

`rerank.py: Reranker.rerank(question, chunks)` sends each `(question, chunk_text)` pair through a cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`). This model is more accurate than cosine similarity because it sees both texts at once. The chunks are re-sorted by the cross-encoder score.

The model is lazy-loaded — downloaded only when the first query runs, not at startup.

### 4d. Build context and call OpenAI

`pipeline.py` joins the top chunks into a context string, with citation tags:

```
[report.pdf p.3]
The main argument is...

---

[report.pdf p.7]
Further evidence suggests...
```

This goes into the OpenAI `gpt-4o-mini` chat completion call alongside a system prompt instructing the model to only use information from the context and to include the citation tags in its answer.

### 4e. Trace

`tracing.py: Tracer.record(...)` runs after every query, regardless of success or failure:

- Appends one JSON line to `data/traces/<YYYY-MM-DD>.jsonl` containing: timestamp, session_id, request_id, query, retrieval config, retrieved chunk IDs + scores, latency timings (retrieval, rerank, generation in ms), model name, answer length, error if any.
- Writes three files to `data/artifacts/<request_id>/`:
  - `prompt.txt` — the full text sent to OpenAI
  - `context.json` — list of chunks used (chunk_id, text, citation)
  - `response.json` — the answer, model, timestamp, error

### 4f. Return

`api.py` returns:

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

## 5. Evaluation

**Files:** `eval/run.py`, `eval/metrics.py`

```bash
python -m eval.run eval/datasets/sample.json
```

The runner loads a JSON dataset of questions (with optional expected source IDs and ground truth answers), runs each through `RAGPipeline.query`, and computes:

- **hit-rate@k** — was at least one expected source in the top-k results?
- **recall@k** — what fraction of expected sources were found?
- **faithfulness** — LLM judge: is the answer grounded in the retrieved context? (0–1)
- **relevance** — LLM judge: does the answer address the question? (0–1)

A JSON + Markdown report is saved to `eval/reports/<run_id>.[json|md]`. The CI pipeline fails if hit-rate@5 < 0.5 or recall@5 < 0.4.

---

## 6. Incremental indexing

When you POST the same PDF again:

1. `compute_file_hash` computes the SHA-256.
2. `index.needs_reindex(doc_id, file_hash)` checks `_doc_hashes`. If the hash matches → skip.
3. If the file changed, `index.remove_doc(doc_id)` soft-deletes the old chunks from `_store` (they stay in FAISS vectors but are invisible to search since `_store` acts as the filter).
4. New chunks are embedded and appended.

This means you never re-embed documents that haven't changed, even if you restart the server and upload a folder of PDFs.
