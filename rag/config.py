"""
rag/config.py — Configuration dataclass + loader.
Reads from environment variables / .env with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent  # project root


@dataclass
class RAGConfig:
    # ── Embedding ────────────────────────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size: int = 500
    chunk_overlap: int = 100
    semantic_chunking: bool = False          # off by default (slower)
    semantic_breakpoint_threshold: float = 0.85  # cosine similarity cutoff

    # ── Retrieval ────────────────────────────────────────────────────────────
    top_k: int = 5

    # ── Reranking ────────────────────────────────────────────────────────────
    use_rerank: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Storage paths ────────────────────────────────────────────────────────
    data_dir: Path = field(default_factory=lambda: BASE_DIR / "data")

    @property
    def faiss_path(self) -> Path:
        return self.data_dir / "faiss.index"

    @property
    def metadata_path(self) -> Path:
        return self.data_dir / "metadata.json"

    @property
    def traces_dir(self) -> Path:
        return self.data_dir / "traces"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    # ── OpenAI generation ────────────────────────────────────────────────────
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.2
    openai_max_tokens: int = 600

    def ensure_dirs(self) -> None:
        """Create all runtime directories if they don't exist."""
        for d in [self.data_dir, self.traces_dir, self.artifacts_dir,
                  BASE_DIR / "eval" / "reports"]:
            d.mkdir(parents=True, exist_ok=True)


def load_config(**overrides) -> RAGConfig:
    """
    Load config from environment, then apply any keyword overrides.

    Environment variables (all optional):
        RAG_EMBEDDING_MODEL, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP,
        RAG_SEMANTIC_CHUNKING, RAG_TOP_K, RAG_USE_MMR, RAG_USE_RERANK,
        RAG_OPENAI_MODEL, RAG_OPENAI_TEMPERATURE, RAG_OPENAI_MAX_TOKENS,
        RAG_DATA_DIR
    """
    cfg = RAGConfig(
        embedding_model=os.getenv("RAG_EMBEDDING_MODEL", RAGConfig.embedding_model),
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", str(RAGConfig.chunk_size))),
        chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", str(RAGConfig.chunk_overlap))),
        semantic_chunking=os.getenv("RAG_SEMANTIC_CHUNKING", "false").lower() == "true",
        top_k=int(os.getenv("RAG_TOP_K", str(RAGConfig.top_k))),
        use_rerank=os.getenv("RAG_USE_RERANK", "true").lower() == "true",
        openai_model=os.getenv("RAG_OPENAI_MODEL", RAGConfig.openai_model),
        openai_temperature=float(os.getenv("RAG_OPENAI_TEMPERATURE",
                                           str(RAGConfig.openai_temperature))),
        openai_max_tokens=int(os.getenv("RAG_OPENAI_MAX_TOKENS",
                                        str(RAGConfig.openai_max_tokens))),
        data_dir=Path(os.getenv("RAG_DATA_DIR", str(BASE_DIR / "data"))),
    )
    # Apply any programmatic overrides
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.ensure_dirs()
    return cfg
