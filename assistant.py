"""
theHelper - AI Research Assistant
Hybrid RAG system: FAISS + Transformers for retrieval/summarization, OpenAI for Q&A.
"""

import os
import re
import hashlib
from io import BytesIO
from typing import List, Optional

import faiss
import numpy as np
import PyPDF2
from openai import OpenAI
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline
from sentence_transformers import SentenceTransformer

load_dotenv()


class FAISSVectorStore:
    """FAISS-based vector store for fast similarity search."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine sim with normalized vectors)
        self.texts: List[str] = []
    
    def add(self, texts: List[str], embeddings: np.ndarray):
        """Add texts and embeddings to the store."""
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        self.texts.extend(texts)
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[str]:
        """Find k most similar texts using FAISS."""
        if self.index.ntotal == 0:
            return []
        
        # Normalize query
        query = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query)
        
        k = min(k, self.index.ntotal)
        _, indices = self.index.search(query, k)
        
        return [self.texts[i] for i in indices[0] if i < len(self.texts)]
    
    def clear(self):
        """Clear all stored data."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.texts = []


class Assistant:
    """
    Hybrid RAG Assistant:
    - SentenceTransformer for embeddings
    - FAISS for fast vector search
    - BART for summarization (local, no API)
    - OpenAI GPT for question answering only
    """
    
    def __init__(self):
        """Initialize models."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_api_key_here":
            raise ValueError("Please set OPENAI_API_KEY in .env file")
        
        self.client = OpenAI(api_key=api_key)
        
        # Local models
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Loading summarization model...")
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=-1  # CPU
        )
        
        # FAISS vector store (384 = dimension of all-MiniLM-L6-v2)
        self.vector_store = FAISSVectorStore(dimension=384)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self._current_doc_hash: Optional[str] = None
        self._full_text: str = ""
        print("Assistant ready!")
    
    def _get_doc_hash(self, content: bytes) -> str:
        """Generate hash for document caching."""
        return hashlib.md5(content).hexdigest()
    
    def extract_text(self, pdf_file) -> str:
        """Extract text from PDF."""
        try:
            if isinstance(pdf_file, BytesIO):
                pdf_file.seek(0)
            
            reader = PyPDF2.PdfReader(pdf_file, strict=False)
            pages = []
            
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)
                    text = re.sub(r'\s+', ' ', text)
                    pages.append(text.strip())
            
            return "\n\n".join(pages)
        except Exception as e:
            raise ValueError(f"Error reading PDF: {e}")
    
    def process_document(self, pdf_file) -> str:
        """Process PDF: extract text, chunk, and build FAISS index."""
        if isinstance(pdf_file, BytesIO):
            pdf_file.seek(0)
            content = pdf_file.read()
            pdf_file.seek(0)
        else:
            with open(pdf_file, 'rb') as f:
                content = f.read()
            pdf_file = BytesIO(content)
        
        doc_hash = self._get_doc_hash(content)
        
        if doc_hash == self._current_doc_hash:
            return self._full_text
        
        text = self.extract_text(pdf_file)
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF")
        
        self._full_text = text
        
        # Chunk and embed
        chunks = self.splitter.split_text(text)
        embeddings = self.embedding_model.encode(chunks, convert_to_numpy=True)
        
        # Build FAISS index
        self.vector_store.clear()
        self.vector_store.add(chunks, embeddings)
        self._current_doc_hash = doc_hash
        
        return text
    
    def get_context(self, question: str, k: int = 5) -> str:
        """Retrieve relevant context using FAISS."""
        query_embedding = self.embedding_model.encode(question, convert_to_numpy=True)
        relevant_chunks = self.vector_store.search(query_embedding, k=k)
        return "\n\n---\n\n".join(relevant_chunks)
    
    def get_summary(self, pdf_file) -> str:
        """Generate summary using BART (local, no API)."""
        text = self.process_document(pdf_file)
        
        max_chars = 3000
        text_chunk = text[:max_chars]
        
        result = self.summarizer(
            text_chunk,
            max_length=150,
            min_length=50,
            do_sample=False
        )
        
        return result[0]['summary_text']
    
    def ask(self, question: str, pdf_file=None) -> str:
        """Answer question using OpenAI with FAISS-retrieved context."""
        if pdf_file:
            self.process_document(pdf_file)
        
        if not self.vector_store.texts:
            return "Please upload a PDF document first."
        
        context = self.get_context(question, k=5)
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are a helpful research assistant. Answer questions based on the provided document context.

Rules:
- Only use information from the context
- If the answer isn't in the context, say so
- Be concise but thorough"""
                },
                {
                    "role": "user",
                    "content": f"Context:\n\n{context}\n\n---\n\nQuestion: {question}"
                }
            ],
            temperature=0.2,
            max_tokens=500
        )
        
        return response.choices[0].message.content


def check_api_key() -> bool:
    """Check if OpenAI API key is configured."""
    api_key = os.getenv("OPENAI_API_KEY")
    return api_key is not None and api_key != "your_api_key_here"


if __name__ == "__main__":
    print("theHelper - AI Research Assistant")
    print("=" * 40)
    
    if check_api_key():
        print("✓ OpenAI API key configured")
        assistant = Assistant()
    else:
        print("✗ Please set OPENAI_API_KEY in .env file")
